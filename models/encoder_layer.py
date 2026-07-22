"""
===============================================================================
 파일: models/encoder_layer.py
 목적:
    Transformer 인코더 레이어 하나: self-attention + feed-forward를
    각각 residual connection과 layer normalization으로 감싼 구조.

 역할:
    인코더 스택에서 반복되는 단위. 설정(`model.norm_style`)에 따라 두
    가지 정규화 배치를 모두 지원한다:
      - "pre"  (Pre-LN):  x = x + Dropout(SubLayer(LN(x)))   — 학습이 안정적
      - "post" (Post-LN): x = LN(x + Dropout(SubLayer(x)))   — 원 논문 방식

 입력 / 출력:
    x       : (batch, src_len, d_model)
    src_mask: (batch, n_heads, src_len, src_len)에 브로드캐스트 가능한
              선택적 불리언 마스크; True = 어텐션 가능.
    출력    : (batch, src_len, d_model)

 구현 세부사항:
    - 모든 계산(LayerNorm, Attention, Dropout, Residual 덧셈, FFN)은
      helper 메서드에 숨기지 않고 forward() 안에 직접 작성되어 있어,
      forward()만 읽어도 전체 데이터 흐름이 보인다.
    - `residual_dropout`은 각 서브레이어의 *출력*이 residual 스트림에
      들어가기 전에 정규화한다 (attention_dropout과 FFN 내부 dropout은
      별도로 설정됨).
    - 이 레이어는 두 개의 LayerNorm을 가지고 있다 (서브레이어당 하나씩);
      Pre-LN에 필요한 스택 레벨의 최종 정규화는 models/encoder.py에
      있다.
===============================================================================
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from config.config import Config
from models.feed_forward import PositionwiseFeedForward
from models.layer_norm import LayerNorm
from models.multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """인코더 레이어 하나: self-attention 서브레이어 + feed-forward 서브레이어.

    모든 하이퍼파라미터는 공유된 :class:`Config`로부터 읽어오므로, 레이어들이
    어디서든 동일하게 생성되고 YAML과 항상 일치한다.

    Args:
        config: 전체 프로젝트 설정 (model, attention 섹션 사용됨).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        m, a = config.model, config.attention
        self.norm_first = m.norm_style == "pre"

        self.self_attention = MultiHeadAttention(
            d_model=m.d_model,
            n_heads=m.n_heads,
            attention_dropout=m.attention_dropout,
            qkv_bias=a.qkv_bias,
            attention_scaling=a.attention_scaling,
            attention_type=a.attention_type,
            causal=a.causal,
            store_attention=a.store_attention,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=m.d_model,
            dim_feedforward=m.dim_feedforward,
            dropout=m.dropout,
            activation=m.activation,
            bias=m.bias,
        )
        self.attention_norm = LayerNorm(m.d_model, eps=m.layer_norm_eps, bias=m.bias)
        self.feed_forward_norm = LayerNorm(m.d_model, eps=m.layer_norm_eps, bias=m.bias)
        self.residual_dropout = nn.Dropout(m.residual_dropout)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """설정된 정규화 배치에 따라 두 서브레이어를 실행한다.

        전체 데이터 흐름이 이 함수 안에 직접 작성되어 있다:
        residual 저장 -> (LayerNorm) -> 서브레이어 -> Dropout ->
        residual 덧셈 -> (LayerNorm).

        Args:
            x: ``(batch, src_len, d_model)`` 소스 표현.
            src_mask: 선택적 패딩 마스크 (True = 실제 토큰).

        Returns:
            ``(batch, src_len, d_model)`` 정제된 표현.
        """
        if self.norm_first:
            # ======================= Self-Attention (Pre-LN) =======================
            # LayerNorm -> Self-Attention -> Dropout -> Residual 덧셈.
            # Pre-LN은 서브레이어의 *입력*을 정규화하고 residual 경로는
            # 정규화하지 않은 채로 유지한다.
            residual = x
            normed = self.attention_norm(x)
            attended = self.self_attention(query=normed, key=normed, value=normed, mask=src_mask)
            x = residual + self.residual_dropout(attended)

            # ======================== Feed Forward (Pre-LN) ========================
            # LayerNorm -> FFN -> Dropout -> Residual 덧셈.
            residual = x
            normed = self.feed_forward_norm(x)
            hidden = self.feed_forward(normed)
            x = residual + self.residual_dropout(hidden)
        else:
            # ======================= Self-Attention (Post-LN) ======================
            # Self-Attention -> Dropout -> Residual 덧셈 -> LayerNorm.
            # Post-LN은 residual 덧셈 *이후에* 정규화한다 (원 논문 방식).
            residual = x
            attended = self.self_attention(query=x, key=x, value=x, mask=src_mask)
            x = self.attention_norm(residual + self.residual_dropout(attended))

            # ======================== Feed Forward (Post-LN) =======================
            # FFN -> Dropout -> Residual 덧셈 -> LayerNorm.
            residual = x
            hidden = self.feed_forward(x)
            x = self.feed_forward_norm(residual + self.residual_dropout(hidden))
        return x
