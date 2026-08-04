"""
===============================================================================
 파일: models/caption_head.py
 목적:
    이미지 인코더를 직접 학습시키기 위한 **캡셔닝 보조 태스크(auxiliary
    captioning task)** 전용 경량 디코더.

 역할:
    이미지 memory만 보고 타겟 문장(= 그 이미지의 캡션)을 생성하도록 학습해,
    이미지 인코더에 강한 gradient 신호를 흘려보낸다. 번역 디코더의 fusion이
    잔차 형태(``text + lam*image``)라 텍스트 경로만으로도 번역이 어느 정도
    되기 때문에, 이 보조 손실이 없으면 이미지 인코더가 "번역에 쓸모 있는
    시각 피처"를 배우기 어렵다.

    학습 전용이다 — 추론(translate/beam_search/evaluate)에서는 사용되지
    않으며, Trainer만 :meth:`Transformer.caption_logits`를 통해 호출한다.

    이 모듈은 ``multimodal.caption_share_decoder=false``(기본)일 때만
    생성된다. true면 번역 디코더를 공유하므로(텍스트 cross-attention만
    건너뜀) 이 헤드는 아예 만들어지지 않는다 — 두 방식의 차이는
    config/config.py의 caption_share_decoder 설명을 참고.

 입력 / 출력:
    forward(tgt_embedded, image_memory, tgt_mask):
        tgt_embedded: (batch, tgt_len, d_model)      임베딩된 타겟 prefix
        image_memory: (batch, num_patches, d_model)  이미지 인코더 출력
        tgt_mask    : (batch, 1, tgt_len, tgt_len)   causal + 패딩 마스크
        ->            (batch, tgt_len, d_model)      캡션 디코더 상태
    어휘집 logits은 Transformer가 자신의 generator를 적용해 얻는다.

 구현 세부사항:
    - 기존 :class:`DecoderLayer`를 그대로 재사용하되, ``use_image=False``로
      강제한 config 사본으로 생성한다. 그러면 레이어 내부에 image
      cross-attention/fusion이 만들어지지 않고 "self-attn + cross-attn +
      FFN" 구조가 되는데, 이때 **cross-attention의 memory 자리에
      image_memory를 넣어** 이미지에 어텐션하게 한다.
    - 이미지 패치에는 패딩이 없으므로 memory_mask는 항상 None이다.
    - 임베딩과 generator는 소유하지 않는다 — 번역 쪽 ``tgt_embedding`` /
      ``generator``를 공유해 파라미터 증가를 최소화하고, 시각 신호가
      임베딩/출력층까지 흐르게 한다.
    - Pre-LN이면 스택 끝에 LayerNorm을 둔다 (models/decoder.py와 동일 규칙).
===============================================================================
"""

from __future__ import annotations

import copy
from typing import Optional

from torch import Tensor, nn

from config.config import Config
from models.decoder_layer import DecoderLayer
from models.layer_norm import LayerNorm


class CaptionHead(nn.Module):
    """이미지 memory만 어텐션하는 캡셔닝 보조 디코더 (학습 전용).

    Args:
        config: 전체 프로젝트 설정 (multimodal.caption_layers가 깊이를 결정).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        m = config.model

        # DecoderLayer를 "텍스트-only" 형태로 만들기 위한 config 사본:
        # use_image=False면 레이어가 image cross-attention/fusion을 만들지
        # 않으므로, 남는 cross-attention 하나를 이미지 전용으로 쓸 수 있다.
        layer_config = copy.deepcopy(config)
        layer_config.multimodal.use_image = False

        self.layers = nn.ModuleList(
            [DecoderLayer(layer_config) for _ in range(config.multimodal.caption_layers)]
        )
        # Pre-LN 스택은 마지막에 한 번 정규화한다 (models/decoder.py와 동일).
        self.final_norm: Optional[LayerNorm] = (
            LayerNorm(m.d_model, bias=m.bias)
            if m.norm_style == "pre"
            else None
        )

    def forward(
        self,
        tgt_embedded: Tensor,
        image_memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """이미지 memory에 어텐션하며 캡션 디코더 상태를 계산한다.

        Args:
            tgt_embedded: ``(batch, tgt_len, d_model)`` 임베딩된 타겟 prefix.
            image_memory: ``(batch, num_patches, d_model)`` 이미지 인코더 출력.
            tgt_mask: causal + 타겟-패딩이 결합된 마스크.

        Returns:
            ``(batch, tgt_len, d_model)`` 캡션 디코더 상태
            (generator를 적용하면 어휘집 logits).
        """
        x = tgt_embedded
        for layer in self.layers:
            # cross-attention의 memory 자리에 이미지를 넣는다.
            # 이미지 패치에는 패딩이 없으므로 memory_mask는 None.
            x = layer(x, image_memory, tgt_mask=tgt_mask, memory_mask=None)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x
