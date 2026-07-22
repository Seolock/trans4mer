"""
===============================================================================
 파일: models/vit_encoder.py
 목적:
    Vision Transformer(ViT) 이미지 인코더를 scratch로 구현한다. pretrained
    이나 완성된 Vision Backbone(torchvision.models, timm, CLIP, DINO, MAE 등)을
    일절 사용하지 않고, Patch Embedding / Position Embedding / Multi-Head
    Self-Attention / Feed-Forward / LayerNorm / Residual을 모두 직접 조립한다.

 역할:
    이미지 하나를 patch 시퀀스로 잘라 Transformer 블록으로 인코딩하고,
    디코더의 Image Cross-Attention이 어텐션할 patch feature 시퀀스를
    만든다. CLS 토큰만 쓰지 않고 patch feature 전체를 반환한다. 텍스트
    Transformer와 동일한 프리미티브(MultiHeadAttention / PositionwiseFeedForward
    / LayerNorm)를 재사용해 코드 스타일을 맞춘다.

 입력 / 출력:
    forward(image):
        image: (batch, channels, image_size, image_size)  실수 텐서
        ->     (batch, num_patches, d_model)  patch feature 시퀀스
               num_patches = (image_size / patch_size) ** 2
               (예: 224 / 16 -> 14x14 = 196 patch)

 구현 세부사항:
    - Patch Embedding은 kernel=stride=patch_size인 Conv2d 하나로 구현한다
      (겹치지 않는 패치를 잘라 각각을 image_embed_dim 벡터로 투영하는 것과
      동일하다). 결과 (B, D, H/P, W/P)를 (B, N, D)로 펼친다.
    - Position Embedding은 학습 가능한 (1, N, image_embed_dim) 파라미터다.
    - 각 블록은 Pre-LN이다: x = x + Dropout(Attn(LN(x))),
      x = x + Dropout(FFN(LN(x))). 이미지에는 causal/padding 마스크가 없다.
    - 인코더 내부 폭(image_embed_dim)이 모델 폭(d_model)과 달라도 되도록,
      마지막에 선택적 선형 투영으로 d_model에 맞춘다 — Image Cross-Attention이
      디코더와 같은 폭을 요구하기 때문이다.
    - 모든 가중치는 Transformer가 init_xavier로 random 초기화한다.
===============================================================================
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from config.config import Config
from models.feed_forward import PositionwiseFeedForward
from models.layer_norm import LayerNorm
from models.multi_head_attention import MultiHeadAttention


class _ViTEncoderLayer(nn.Module):
    """ViT 인코더 블록 하나: (마스크 없는) self-attention + feed-forward, Pre-LN.

    텍스트 EncoderLayer와 같은 데이터 흐름이지만 이미지 전용 폭
    (image_embed_dim / image_heads / image_ffn_dim)을 사용하며, 시퀀스가
    고정 길이(패치 개수)라 어텐션 마스크가 필요 없다.

    Args:
        config: 전체 프로젝트 설정 (multimodal / model / attention 사용).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        mm, m, a = config.multimodal, config.model, config.attention
        self.self_attention = MultiHeadAttention(
            d_model=mm.image_embed_dim,
            n_heads=mm.image_heads,
            attention_dropout=m.attention_dropout,
            qkv_bias=a.qkv_bias,
            attention_scaling=a.attention_scaling,
            attention_type=a.attention_type,
            causal=False,
            store_attention=a.store_attention,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=mm.image_embed_dim,
            dim_feedforward=mm.image_ffn_dim,
            dropout=m.dropout,
            activation=m.activation,
            bias=m.bias,
        )
        self.attention_norm = LayerNorm(mm.image_embed_dim, eps=m.layer_norm_eps, bias=m.bias)
        self.feed_forward_norm = LayerNorm(mm.image_embed_dim, eps=m.layer_norm_eps, bias=m.bias)
        self.residual_dropout = nn.Dropout(m.residual_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Pre-LN self-attention + feed-forward를 실행한다.

        Args:
            x: ``(batch, num_patches, image_embed_dim)`` patch 표현.

        Returns:
            ``(batch, num_patches, image_embed_dim)`` 정제된 표현.
        """
        # ======================= Self-Attention (Pre-LN) =======================
        residual = x
        normed = self.attention_norm(x)
        attended = self.self_attention(query=normed, key=normed, value=normed, mask=None)
        x = residual + self.residual_dropout(attended)

        # ======================== Feed Forward (Pre-LN) ========================
        residual = x
        normed = self.feed_forward_norm(x)
        hidden = self.feed_forward(normed)
        x = residual + self.residual_dropout(hidden)
        return x


class ViTEncoder(nn.Module):
    """이미지를 patch feature 시퀀스로 인코딩하는 scratch Vision Transformer.

    Args:
        config: 전체 프로젝트 설정 (multimodal 섹션이 이미지 하이퍼파라미터를
            제공하고, 출력 폭은 model.d_model에 맞춘다).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        mm, m = config.multimodal, config.model
        self.patch_size = mm.patch_size
        grid = mm.image_size // mm.patch_size
        self.num_patches = grid * grid

        # -------------------------------------------------- Patch Embedding
        # kernel=stride=patch_size인 Conv2d: 겹치지 않는 패치를 잘라 각각을
        # image_embed_dim 벡터로 투영. (B, C, H, W) -> (B, D, H/P, W/P).
        self.patch_embedding = nn.Conv2d(
            mm.image_channels,
            mm.image_embed_dim,
            kernel_size=mm.patch_size,
            stride=mm.patch_size,
        )

        # -------------------------------------------------- Position Embedding
        # 학습 가능한 절대 위치 임베딩 (CLS 토큰 없음 — patch feature 전체 사용).
        # 최종 random 초기화는 Transformer.init_xavier가 담당한다.
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, mm.image_embed_dim)
        )
        self.embedding_dropout = nn.Dropout(m.embedding_dropout)

        # -------------------------------------------------- Transformer 블록
        self.layers = nn.ModuleList(
            [_ViTEncoderLayer(config) for _ in range(mm.image_layers)]
        )
        self.final_norm = LayerNorm(mm.image_embed_dim, eps=m.layer_norm_eps, bias=m.bias)

        # -------------------------------------------------- d_model로 투영
        # Image Cross-Attention은 디코더와 같은 폭(d_model)을 요구한다.
        self.projection: nn.Module = (
            nn.Linear(mm.image_embed_dim, m.d_model)
            if mm.image_embed_dim != m.d_model
            else nn.Identity()
        )

    def forward(self, image: Tensor) -> Tensor:
        """이미지 배치를 patch feature 시퀀스로 인코딩한다.

        Args:
            image: ``(batch, channels, image_size, image_size)``.

        Returns:
            ``(batch, num_patches, d_model)`` patch feature 시퀀스.
        """
        # ===================== 1) Patch Embedding =====================
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        patches = self.patch_embedding(image)
        patches = patches.flatten(2).transpose(1, 2)  # (B, D, gh*gw) -> (B, N, D)

        # ===================== 2) Position Embedding =====================
        x = self.embedding_dropout(patches + self.position_embedding)

        # ===================== 3) Transformer 블록 스택 =====================
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)

        # ===================== 4) d_model로 투영 =====================
        # (B, N, image_embed_dim) -> (B, N, d_model)
        return self.projection(x)
