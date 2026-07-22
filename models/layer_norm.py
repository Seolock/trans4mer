"""
===============================================================================
 파일: models/layer_norm.py
 목적:
    Layer Normalization을 처음부터 직접 구현 (nn.LayerNorm 사용하지 않음).

 역할:
    각 위치의 특징 벡터를 평균 0, 분산 1로 정규화한 뒤, 학습된 아핀 변환을
    적용한다. 설정된 정규화 방식(pre-LN vs post-LN)에 따라 매 서브레이어의
    앞 또는 뒤에서 사용되며, pre-LN 인코더/디코더 스택의 마지막
    정규화로도 쓰인다.

 입력 / 출력:
    입력 : (..., d_model) 실수 텐서 — 정규화는 마지막 차원에 대해 수행.
    출력 : (..., d_model) 실수 텐서, 입력과 동일한 형태.

 구현 세부사항:
    - 통계량은 배치가 아니라 위치(토큰)별로 계산되며, 이것이 바로
      LayerNorm이 시퀀스 길이와 배치 크기에 독립적인 이유다.
    - 분산은 편향(biased) 추정량(N으로 나눔)을 사용해 torch.nn.LayerNorm과
      동일하게 맞춘다.
    - ``eps``는 수치 안정성을 위해 sqrt 안쪽에 더해진다; ``bias``는
      설정으로 비활성화할 수 있다 (일부 최신 아키텍처는 bias를 생략함).
===============================================================================
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """학습 가능한 아핀 변환을 포함한, 마지막 차원 기준 layer normalization.

    ``y = (x - mean) / sqrt(var + eps) * weight + bias``

    Args:
        d_model: 정규화 대상(마지막) 차원의 크기.
        eps: 수치 안정성을 위해 분산에 더하는 작은 상수.
        bias: False면 덧셈 ``bias`` 파라미터를 생략하고 곱셈 게인
            ``weight``만 학습한다.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        """``x``를 마지막 차원 기준으로 정규화한다.

        Args:
            x: ``(..., d_model)`` 형태의 텐서.

        Returns:
            동일한 형태의 정규화된 텐서.
        """
        # 위치별 통계: 각 토큰이 독립적으로 정규화된다.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        if self.bias is not None:
            return normalized * self.weight + self.bias
        return normalized * self.weight

    def extra_repr(self) -> str:
        return f"d_model={self.weight.numel()}, eps={self.eps}, bias={self.bias is not None}"
