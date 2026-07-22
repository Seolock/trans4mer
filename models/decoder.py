"""
===============================================================================
 파일: models/decoder.py
 목적:
    Transformer 디코더 스택: N개의 동일한 DecoderLayer를 순서대로 적용하고,
    Pre-LN 설정에서 필요한 최종 LayerNorm을 더한다.

 역할:
    타겟 측 표현을 자기회귀적으로(autoregressively) 구성한다. 각 레이어는
    먼저 (causal 마스킹된) 타겟 prefix 내에서 정보를 섞은 뒤,
    cross-attention을 통해 인코더 memory로부터 소스 정보를 끌어온다.

 입력 / 출력:
    x          : (batch, tgt_len, d_model)  임베딩된 타겟 토큰
    memory     : (batch, src_len, d_model)  인코더 출력
    tgt_mask   : (batch, n_heads, tgt_len, tgt_len)에 브로드캐스트 가능한
                 불리언
    memory_mask: (batch, n_heads, tgt_len, src_len)에 브로드캐스트 가능한
                 불리언
    출력       : (batch, tgt_len, d_model)

 구현 세부사항:
    models/encoder.py와 동일한 구조를 따른다: 독립적인 레이어들을
    nn.ModuleList에 담고, norm_style == "pre"일 때만 스택 레벨 LayerNorm
    하나를 둔다.
===============================================================================
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from config.config import Config
from models.decoder_layer import DecoderLayer
from models.layer_norm import LayerNorm


class Decoder(nn.Module):
    """``num_decoder_layers``개의 디코더 레이어를 쌓은 스택.

    Args:
        config: 전체 프로젝트 설정.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        m = config.model
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(m.num_decoder_layers)]
        )
        self.final_norm: Optional[LayerNorm] = (
            LayerNorm(m.d_model, eps=m.layer_norm_eps, bias=m.bias)
            if m.norm_style == "pre"
            else None
        )

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """임베딩된 타겟 prefix 배치를 미리 계산된 인코더 memory에 대해 디코딩한다.

        Args:
            x: ``(batch, tgt_len, d_model)`` 임베딩된 타겟 토큰.
            memory: ``(batch, src_len, d_model)`` 인코더 출력.
            tgt_mask: causal + 타겟-패딩이 결합된 마스크.
            memory_mask: cross-attention을 위한 소스-패딩 마스크.

        Returns:
            ``(batch, tgt_len, d_model)`` 디코더 상태 (프로젝션 전).
        """
        # ===================== 1) 디코더 레이어 스택 =====================
        # N개의 동일한 레이어를 순서대로 통과한다 (각 레이어 내부:
        # 마스킹된 Self-Attention -> Cross-Attention -> Feed Forward,
        # residual + LayerNorm 포함).
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)

        # ===================== 2) 최종 LayerNorm (Pre-LN 전용) =====================
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x
