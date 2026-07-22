"""
===============================================================================
 파일: models/encoder.py
 목적:
    Transformer 인코더 스택: N개의 동일한 EncoderLayer를 순서대로 적용하고,
    Pre-LN 설정에서 필요한 최종 LayerNorm을 더한다.

 역할:
    임베딩된 소스 토큰을 디코더가 cross-attention으로 어텐션할 "memory"로
    변환한다. 모든 소스 위치는 다른 모든 (패딩이 아닌) 소스 위치를 볼 수
    있다 — 인코딩은 양방향(bidirectional)이다.

 입력 / 출력:
    x       : (batch, src_len, d_model)  임베딩된 소스 토큰
    src_mask: (batch, n_heads, src_len, src_len)에 브로드캐스트 가능한
              선택적 불리언; True = 어텐션 가능.
    출력    : (batch, src_len, d_model)  인코더 memory

 구현 세부사항:
    - 레이어들은 (가중치를 공유하지 않는) 독립적인 인스턴스이며, 파라미터가
      올바르게 등록되도록 nn.ModuleList에 담긴다.
    - Pre-LN 스택은 마지막에 LayerNorm을 하나 더 추가한다: Pre-LN은
      정규화되지 않은 residual 스트림을 유지하기 때문에, 출력을 맨 위에서
      한 번 정규화해주지 않으면 깊이가 깊어질수록 활성값이 커진다.
      Post-LN은 이미 매 레이어 출구에서 정규화하므로 여기에 최종 norm이
      추가되지 않는다.
===============================================================================
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from config.config import Config
from models.encoder_layer import EncoderLayer
from models.layer_norm import LayerNorm


class Encoder(nn.Module):
    """``num_encoder_layers``개의 인코더 레이어를 쌓은 스택.

    Args:
        config: 전체 프로젝트 설정.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        m = config.model
        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(m.num_encoder_layers)]
        )
        # Pre-LN에서만 최종 norm을 둔다 (이유는 파일 상단 설명 참고).
        self.final_norm: Optional[LayerNorm] = (
            LayerNorm(m.d_model, eps=m.layer_norm_eps, bias=m.bias)
            if m.norm_style == "pre"
            else None
        )

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """임베딩된 소스 시퀀스 배치를 인코딩한다.

        Args:
            x: ``(batch, src_len, d_model)`` 임베딩된 소스 토큰.
            src_mask: 선택적 소스-패딩 마스크 (True = 실제 토큰).

        Returns:
            ``(batch, src_len, d_model)`` 인코더 memory.
        """
        # ===================== 1) 인코더 레이어 스택 =====================
        # N개의 동일한 레이어를 순서대로 통과한다 (각 레이어 내부:
        # Self-Attention -> Feed Forward, residual + LayerNorm 포함).
        for layer in self.layers:
            x = layer(x, src_mask)

        # ===================== 2) 최종 LayerNorm (Pre-LN 전용) =====================
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x
