"""
===============================================================================
 파일: models/decoder_layer.py
 목적:
    세 개의 서브레이어를 가진 Transformer 디코더 레이어 하나:
      1. 타겟 prefix에 대한 causal 마스킹된 self-attention,
      2. 인코더 출력("memory")에 대한 cross-attention,
      3. position-wise feed-forward network,
    각각 residual connection과 layer normalization으로 감싸져 있다.

 역할:
    디코더 스택에서 반복되는 단위. 서브레이어 1은 각 타겟 위치가 이전
    타겟 위치들만 참고할 수 있게 한다(mask_future); 서브레이어 2는 소스
    정보가 디코더로 들어오는 지점이다 (쿼리는 디코더에서, 키/값은 인코더
    memory에서 온다).

 입력 / 출력:
    x          : (batch, tgt_len, d_model)   디코더 측 표현
    memory     : (batch, src_len, d_model)   인코더 출력
    tgt_mask   : (batch, n_heads, tgt_len, tgt_len)에 브로드캐스트 가능한
                 불리언 — causal 마스크와 타겟-패딩 마스크가 결합된 것.
    memory_mask: (batch, n_heads, tgt_len, src_len)에 브로드캐스트 가능한
                 불리언 — cross-attention을 위한 소스-패딩 마스크.
    출력       : (batch, tgt_len, d_model)

 구현 세부사항:
    - 모든 계산(LayerNorm, Attention, Dropout, Residual 덧셈, FFN)은
      helper 메서드에 숨기지 않고 forward() 안에 직접 작성되어 있어,
      forward()만 읽어도 세 서브레이어의 전체 흐름이 보인다.
    - 정규화 배치("pre" vs "post")는 encoder_layer.py와 정확히 동일하다;
      두 데이터 흐름 방정식은 그 파일을 참고하라.
===============================================================================
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from config.config import Config
from models.feed_forward import PositionwiseFeedForward
from models.layer_norm import LayerNorm
from models.multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """디코더 레이어 하나: 마스킹된 self-attention, cross-attention, feed-forward.

    Args:
        config: 전체 프로젝트 설정 (model, attention 섹션 사용됨).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        m, a = config.model, config.attention
        self.norm_first = m.norm_style == "pre"

        def build_attention(causal: bool) -> MultiHeadAttention:
            """두 어텐션 모듈은 causal 여부를 제외한 모든 설정을 공유한다."""
            return MultiHeadAttention(
                d_model=m.d_model,
                n_heads=m.n_heads,
                attention_dropout=m.attention_dropout,
                qkv_bias=a.qkv_bias,
                attention_scaling=a.attention_scaling,
                attention_type=a.attention_type,
                causal=causal,
                store_attention=a.store_attention,
            )

        # self-attention은 둘 중 하나의 플래그가 요청하면 causal이 된다;
        # 디코더는 추가로 명시적인 causal 마스크를 만들기 때문에
        # (models/transformer.py), 여기서 `a.causal`은 주로 디코더 전용
        # 재사용을 위한 것이다.
        self.self_attention = build_attention(causal=a.causal)
        self.cross_attention = build_attention(causal=False)
        self.feed_forward = PositionwiseFeedForward(
            d_model=m.d_model,
            dim_feedforward=m.dim_feedforward,
            dropout=m.dropout,
            activation=m.activation,
            bias=m.bias,
        )
        self.self_attention_norm = LayerNorm(m.d_model, eps=m.layer_norm_eps, bias=m.bias)
        self.cross_attention_norm = LayerNorm(m.d_model, eps=m.layer_norm_eps, bias=m.bias)
        self.feed_forward_norm = LayerNorm(m.d_model, eps=m.layer_norm_eps, bias=m.bias)
        self.residual_dropout = nn.Dropout(m.residual_dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """설정된 정규화 배치에 따라 세 서브레이어를 실행한다.

        전체 데이터 흐름이 이 함수 안에 직접 작성되어 있다:
        residual 저장 -> (LayerNorm) -> 서브레이어 -> Dropout ->
        residual 덧셈 -> (LayerNorm) 을 세 서브레이어에 대해 반복.

        Args:
            x: ``(batch, tgt_len, d_model)`` 타겟 측 표현.
            memory: ``(batch, src_len, d_model)`` 인코더 출력.
            tgt_mask: causal + 타겟-패딩이 결합된 마스크.
            memory_mask: cross-attention을 위한 소스-패딩 마스크.

        Returns:
            ``(batch, tgt_len, d_model)`` 정제된 타겟 표현.
        """
        if self.norm_first:
            # ====================== Self-Attention (Pre-LN) ======================
            # LayerNorm -> 마스킹된 Self-Attention -> Dropout -> Residual 덧셈.
            # 타겟 prefix 내부에서만 정보를 섞는다 (미래 위치는 tgt_mask로 차단).
            residual = x
            normed = self.self_attention_norm(x)
            attended = self.self_attention(query=normed, key=normed, value=normed, mask=tgt_mask)
            x = residual + self.residual_dropout(attended)

            # ====================== Cross-Attention (Pre-LN) =====================
            # LayerNorm -> Cross-Attention -> Dropout -> Residual 덧셈.
            # 쿼리는 디코더에서, 키/값은 인코더 memory에서 온다.
            residual = x
            normed = self.cross_attention_norm(x)
            attended = self.cross_attention(
                query=normed, key=memory, value=memory, mask=memory_mask
            )
            x = residual + self.residual_dropout(attended)

            # ======================= Feed Forward (Pre-LN) =======================
            # LayerNorm -> FFN -> Dropout -> Residual 덧셈.
            residual = x
            normed = self.feed_forward_norm(x)
            hidden = self.feed_forward(normed)
            x = residual + self.residual_dropout(hidden)
        else:
            # ====================== Self-Attention (Post-LN) =====================
            # 마스킹된 Self-Attention -> Dropout -> Residual 덧셈 -> LayerNorm.
            residual = x
            attended = self.self_attention(query=x, key=x, value=x, mask=tgt_mask)
            x = self.self_attention_norm(residual + self.residual_dropout(attended))

            # ====================== Cross-Attention (Post-LN) ====================
            # Cross-Attention -> Dropout -> Residual 덧셈 -> LayerNorm.
            residual = x
            attended = self.cross_attention(query=x, key=memory, value=memory, mask=memory_mask)
            x = self.cross_attention_norm(residual + self.residual_dropout(attended))

            # ======================= Feed Forward (Post-LN) ======================
            # FFN -> Dropout -> Residual 덧셈 -> LayerNorm.
            residual = x
            hidden = self.feed_forward(x)
            x = self.feed_forward_norm(residual + self.residual_dropout(hidden))
        return x
