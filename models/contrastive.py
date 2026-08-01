"""
===============================================================================
 파일: models/contrastive.py
 목적:
    이미지 인코더 출력과 텍스트 인코더 출력을 같은 의미 공간으로 끌어당기는
    **배치 내 대조 손실(CLIP-style symmetric InfoNCE)**.

 역할:
    같은 (문장, 이미지) 쌍은 가깝게, 배치 안의 다른 쌍끼리는 멀게 만든다.
    번역 손실은 디코더의 image cross-attention/fusion을 거쳐 이미지 인코더에
    간접적인 신호만 주는데, fusion이 잔차 형태(``text + lam*image``)라
    텍스트 경로만으로도 번역이 되기 때문에 두 인코더의 출력이 같은 공간에
    놓인다는 보장이 없다. 이 손실이 그 정렬을 직접 강제한다.

    학습 전용이다 — 추론(translate/beam_search/evaluate)에서는 사용되지
    않으며, Trainer만 :meth:`Transformer.contrastive_loss`를 통해 호출한다.

 입력 / 출력:
    forward(text_pooled, image_pooled):
        text_pooled : (batch, d_model)  풀링된 텍스트 인코더 출력
        image_pooled: (batch, d_model)  풀링된 이미지 인코더 출력
        ->            스칼라 손실 (대칭 InfoNCE)

 구현 세부사항:
    - 모달리티별 **선형 투영 헤드**를 거친 뒤 L2 정규화해서 비교한다. 인코더
      출력을 직접 제약하지 않으므로 번역 품질을 해칠 위험이 낮고, CNN 인코더
      출력이 LayerNorm을 거치지 않는 비대칭(vit는 거침)도 여기서 흡수된다.
    - temperature는 CLIP과 동일하게 학습 가능한 ``logit_scale``로 두고
      ``log(1/T)``로 초기화한다. 0-dim 스칼라라 ``init_xavier``
      (``parameter.dim() > 1``)가 건드리지 않으므로 이 초기값이 살아남는다.
    - ``exp()`` 후 100으로 clamp한다 (CLIP과 동일). 정규화된 벡터의 내적은
      [-1, 1]이므로 logits이 [-100, 100]에 갇혀 fp16(max 65504)에서도 안전하고,
      autocast가 ``cross_entropy``를 fp32로 승격시키므로 AMP에서 문제없다.
    - 배치 크기가 곧 negative 개수다. gradient 누적은 negative pool을 늘려주지
      않는다 (micro-batch 안에서만 대조된다).
===============================================================================
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config.config import Config


class ContrastiveHead(nn.Module):
    """이미지/텍스트 표현을 공유 공간으로 투영해 대조 손실을 계산한다 (학습 전용).

    Args:
        config: 전체 프로젝트 설정 (multimodal.contrastive_dim /
            contrastive_temperature, model.d_model 사용).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        m, mm = config.model, config.multimodal

        # 모달리티별 투영. L2 정규화가 뒤따르므로 bias는 의미가 없다.
        self.text_projection = nn.Linear(m.d_model, mm.contrastive_dim, bias=False)
        self.image_projection = nn.Linear(m.d_model, mm.contrastive_dim, bias=False)

        # CLIP 방식 학습 가능 temperature: log(1/T)로 초기화하고 exp 후 사용.
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / mm.contrastive_temperature))
        )

    def forward(self, text_pooled: Tensor, image_pooled: Tensor) -> Tensor:
        """대칭 InfoNCE 손실을 계산한다.

        Args:
            text_pooled: ``(batch, d_model)`` 풀링된 텍스트 인코더 출력.
            image_pooled: ``(batch, d_model)`` 풀링된 이미지 인코더 출력.

        Returns:
            스칼라 손실. 완전 무작위 정렬이면 ``log(batch)`` 근처이고,
            완벽히 정렬되면 0에 가까워진다.
        """
        text = F.normalize(self.text_projection(text_pooled), dim=-1)
        image = F.normalize(self.image_projection(image_pooled), dim=-1)

        # (batch, batch) 유사도 행렬 — 대각선이 정답 쌍이다.
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * text @ image.t()
        labels = torch.arange(logits.size(0), device=logits.device)

        # 대칭: text->image와 image->text 두 방향의 평균.
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        )
