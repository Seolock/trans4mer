"""
===============================================================================
 파일: models/fusion.py
 목적:
    디코더의 두 cross-attention 출력(텍스트 cross-attention과 이미지
    cross-attention)을 하나의 표현으로 융합(fusion)하는 모듈들.

 역할:
    Multimodal 디코더 레이어에서 Text Cross-Attention과 Image
    Cross-Attention의 결과를 합쳐 residual 스트림으로 돌려보낸다. 융합
    방식은 설정(multimodal.fusion_type)으로 교체 가능하도록 레지스트리로
    관리한다 (models/multi_head_attention.py의 ATTENTION_REGISTRY와 동일한
    패턴).

 입력 / 출력 (모든 융합 모듈 공통):
    text_output  : (batch, tgt_len, d_model)  텍스트 cross-attention 출력
    image_output : (batch, tgt_len, d_model)  이미지 cross-attention 출력
    출력         : (batch, tgt_len, d_model)  융합된 표현

 구현 세부사항:
    - SumFusion    : 파라미터 없이 두 출력을 더한다.
    - WeightedFusion: λ*text + (1-λ)*image. λ는 고정값이거나, 학습 가능한
      파라미터(sigmoid로 (0,1) 유지)일 수 있다.
    - GateFusion   : [text; image]로부터 게이트를 학습해 원소별로 섞는다.
    - 새 융합 방식은 (text, image) -> fused forward를 가진 nn.Module을
      만들어 FUSION_REGISTRY에 등록하면 설정에서 바로 선택된다.
===============================================================================
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SumFusion(nn.Module):
    """가장 단순한 융합: 두 출력을 그대로 더한다 (파라미터 없음).

    ``output = text_output + image_output``
    """

    def forward(self, text_output: Tensor, image_output: Tensor) -> Tensor:
        """두 cross-attention 출력을 더한다.

        Args:
            text_output: ``(batch, tgt_len, d_model)``.
            image_output: ``(batch, tgt_len, d_model)``.

        Returns:
            ``(batch, tgt_len, d_model)``.
        """
        return text_output + image_output


class WeightedFusion(nn.Module):
    """가중 합 융합: ``λ * text + (1 - λ) * image``.

    ``learnable=False``면 λ는 ``lambda_init`` 고정값이다. ``learnable=True``면
    λ는 학습 가능한 스칼라 파라미터이며, sigmoid를 통과시켜 항상 (0, 1)에
    머무르게 한다 (초기값은 sigmoid(logit) = lambda_init가 되도록 역산).

    Args:
        lambda_init: 텍스트 쪽 초기 가중치 λ (0~1).
        learnable: λ를 학습할지 여부.
    """

    def __init__(self, lambda_init: float = 0.5, learnable: bool = False) -> None:
        super().__init__()
        self.learnable = learnable
        if learnable:
            # sigmoid(logit) == lambda_init가 되도록 초기 logit을 역산한다.
            eps = 1e-6
            clamped = min(max(lambda_init, eps), 1.0 - eps)
            logit = torch.log(torch.tensor(clamped / (1.0 - clamped)))
            self.lambda_logit = nn.Parameter(logit)
        else:
            # 학습되지 않는 상수 버퍼: 모듈과 함께 이동하지만 gradient는 없다.
            self.register_buffer("lambda_value", torch.tensor(float(lambda_init)))

    def forward(self, text_output: Tensor, image_output: Tensor) -> Tensor:
        """λ로 가중한 두 출력의 볼록 결합을 계산한다.

        Args:
            text_output: ``(batch, tgt_len, d_model)``.
            image_output: ``(batch, tgt_len, d_model)``.

        Returns:
            ``(batch, tgt_len, d_model)``.
        """
        if self.learnable:
            lam = torch.sigmoid(self.lambda_logit)
        else:
            lam = self.lambda_value
        return lam * text_output + (1.0 - lam) * image_output


class GateFusion(nn.Module):
    """게이트 융합: ``[text; image]``로부터 원소별 게이트를 학습한다.

    ``gate = sigmoid(Linear([text; image]))``
    ``output = gate * text + (1 - gate) * image``

    게이트가 d_model 차원별로 계산되므로, 모델이 각 특징 차원마다 텍스트와
    이미지 중 무엇을 얼마나 신뢰할지 스스로 조절할 수 있다.

    Args:
        d_model: cross-attention 출력의 폭.
        bias: 게이트 선형 레이어의 bias 항 사용 여부.
    """

    def __init__(self, d_model: int, bias: bool = True) -> None:
        super().__init__()
        # 입력은 두 출력을 이어 붙인 (..., 2*d_model), 출력은 (..., d_model) 게이트.
        self.gate = nn.Linear(2 * d_model, d_model, bias=bias)

    def forward(self, text_output: Tensor, image_output: Tensor) -> Tensor:
        """학습된 게이트로 두 출력을 원소별로 섞는다.

        Args:
            text_output: ``(batch, tgt_len, d_model)``.
            image_output: ``(batch, tgt_len, d_model)``.

        Returns:
            ``(batch, tgt_len, d_model)``.
        """
        # (B, L, d_model) 두 개 -> (B, L, 2*d_model) -> 게이트 (B, L, d_model).
        combined = torch.cat([text_output, image_output], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        return gate * text_output + (1.0 - gate) * image_output


def build_fusion(fusion_type: str, d_model: int, config: "object") -> nn.Module:
    """설정으로부터 융합 전략을 선택하는 팩토리 함수.

    Args:
        fusion_type: "sum", "weighted" 또는 "gate".
        d_model: cross-attention 출력의 폭 (gate 융합에 필요).
        config: multimodal 설정 (weighted 융합의 λ 관련 필드 사용).

    Returns:
        ``(text_output, image_output) -> fused`` forward를 가진 nn.Module.

    Raises:
        ValueError: 등록되지 않은 fusion_type일 경우.
    """
    if fusion_type == "sum":
        return SumFusion()
    if fusion_type == "weighted":
        return WeightedFusion(
            lambda_init=config.fusion_lambda,
            learnable=config.fusion_learnable_lambda,
        )
    if fusion_type == "gate":
        return GateFusion(d_model)
    raise ValueError(
        f"Unknown fusion_type '{fusion_type}'. Available: ['sum', 'weighted', 'gate']"
    )
