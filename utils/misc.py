"""
===============================================================================
 파일: utils/misc.py
 목적:
    특정 서브시스템에 속하지 않는, 여러 곳에서 두루 쓰이는 작은 헬퍼들.

 역할:
    - 디바이스 선택 (CUDA -> Apple MPS -> CPU).
    - 파라미터 개수 세기 / 보기 좋게 포맷팅.
    - 중첩된 배치를 디바이스로 옮기기.
    - 학습 루프에서 쓰는 이동 평균(running-average) 미터.

 입력 / 출력:
    개별 함수 docstring 참고; 모든 헬퍼는 순수 함수이거나 사소하게
    상태를 가진다 (AverageMeter).

 구현 세부사항:
    AverageMeter는 가중치가 적용된(WEIGHTED) 업데이트를 지원한다: trainer는
    각 배치의 손실을 패딩이 아닌 토큰 개수로 가중치를 매기므로, epoch
    평균이 배치별 평균이 아니라 진짜 토큰별 평균이 된다.
===============================================================================
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


def get_device(prefer: str | None = None) -> torch.device:
    """사용 가능한 최선의 디바이스를 고르거나 (명시적 선호를 존중한다).

    Args:
        prefer: 선택적으로 명시할 디바이스 문자열 ("cuda", "mps", "cpu").

    Returns:
        :class:`torch.device`.
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """모델 파라미터 개수를 센다.

    Args:
        model: 임의의 nn.Module.
        trainable_only: ``requires_grad``인 파라미터만 센다.

    Returns:
        전체 파라미터 개수. 참고: 묶인(tied) 가중치(공유된 Parameter
        객체)는 한 번만 세는데, 이것이 정직한 숫자다.
    """
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        if id(p) in seen:  # 이미 센 묶인 파라미터는 건너뛴다
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def format_count(n: int) -> str:
    """사람이 읽기 좋은 개수 표현, 예: 12_345_678 -> '12.3M'."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def move_to_device(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    """배치 dict 안의 모든 텐서를 ``device``로 옮긴다.

    Args:
        batch: 이름을 텐서에 매핑한 것 (collator가 만들어냄).
        device: 대상 디바이스.

    Returns:
        모든 텐서가 옮겨진 새 dict (non_blocking은 고정된(pinned) 메모리와
        함께 사용될 때 비동기 host-to-GPU 복사를 가능하게 함).
    """
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


class AverageMeter:
    """가중치가 적용된 이동 평균 (예: epoch 전체의 토큰별 손실)."""

    def __init__(self) -> None:
        self.total = 0.0
        self.weight = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        """``value``를 주어진 ``weight``로 누적한다."""
        self.total += value * weight
        self.weight += weight

    @property
    def average(self) -> float:
        """현재 가중 평균 (업데이트가 없으면 0.0)."""
        return self.total / self.weight if self.weight > 0 else 0.0
