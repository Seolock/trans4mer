"""
===============================================================================
 파일: trainer/scheduler.py
 목적:
    선형 warm-up을 포함하는 학습률 스케줄, 설정으로부터 만들어짐:
    "noam", "cosine", "linear", "constant".

 역할:
    `build_scheduler()`는 Trainer가 한 번 호출하며, 매 OPTIMIZER 스텝마다
    한 번씩 스텝된다 (배치마다가 아님 — gradient 누적이 이미 여러
    micro-batch를 하나의 옵티마이저 스텝으로 합쳤기 때문).

 입력 / 출력:
    입력 : 옵티마이저, 최적화 설정, 이번 실행에서 계획된 전체 옵티마이저
           스텝 수.
    출력 : 기본 학습률(`optimization.learning_rate` = 최대(peak) LR)에
           곱해질 배율을 내는 torch.optim.lr_scheduler.LambdaLR.

 구현 세부사항:
    모든 스케줄은 곱셈 배율 f(step)으로 표현된다:
      - constant: warm-up 0 -> 1, 이후 1로 유지.
      - linear  : warm-up 0 -> 1, 이후 min_lr/base_lr까지 선형 decay.
      - cosine  : warm-up 0 -> 1, 이후 min_lr/base_lr까지 half-cosine
                  decay. (`cosine_decay: false`면 warm-up 이후 1로 고정.)
      - noam    : "Attention Is All You Need"의 역제곱근(inverse-sqrt)
                  스케줄을, 최댓값(step == warmup_steps 시점)이 1이 되도록
                  정규화한 것 — 이렇게 하면 다른 모든 스케줄러와 마찬가지로
                  learning_rate가 "최대 LR"이라는 의미를 유지하며, 원 논문의
                  d_model에서 유도된 크기와 다르다.
    0^(-0.5)를 피하기 위해 스텝은 1부터 센다.
===============================================================================
"""

from __future__ import annotations

import math
from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from config.config import OptimizationConfig


def _constant_factory(warmup: int) -> Callable[[int], float]:
    """warm-up 후 그대로 유지."""

    def factor(step: int) -> float:
        step = max(1, step)
        if step < warmup:
            return step / max(1, warmup)
        return 1.0

    return factor


def _linear_factory(warmup: int, total_steps: int, floor: float) -> Callable[[int], float]:
    """warm-up 후 ``floor``까지 선형 decay."""

    def factor(step: int) -> float:
        step = max(1, step)
        if step < warmup:
            return step / max(1, warmup)
        remaining = max(0, total_steps - step)
        span = max(1, total_steps - warmup)
        return floor + (1.0 - floor) * (remaining / span)

    return factor


def _cosine_factory(
    warmup: int, total_steps: int, floor: float, decay: bool
) -> Callable[[int], float]:
    """warm-up 후 ``floor``까지 half-cosine decay (decay가 아니면 고정)."""

    def factor(step: int) -> float:
        step = max(1, step)
        if step < warmup:
            return step / max(1, warmup)
        if not decay:
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return floor + (1.0 - floor) * cosine

    return factor


def _noam_factory(warmup: int) -> Callable[[int], float]:
    """정규화된 Noam / 역제곱근 스케줄 (warm-up 시점에서 최대 배율 = 1)."""

    def factor(step: int) -> float:
        step = max(1, step)
        # 원본: min(step^-0.5, step * warmup^-1.5); sqrt(warmup)을 곱해
        # 최댓값을 정확히 1.0으로 재조정한다.
        return (warmup**0.5) * min(step**-0.5, step * warmup**-1.5)

    return factor


def build_scheduler(
    optimizer: Optimizer, config: OptimizationConfig, total_steps: int
) -> LambdaLR:
    """설정된 LR 스케줄을 구성한다.

    Args:
        optimizer: 학습률이 조정될 옵티마이저.
        config: `optimization` 설정 섹션.
        total_steps: 전체 실행에서 계획된 옵티마이저 스텝 수 (decay하는
            스케줄이 min_lr에 도달할 지점을 알기 위해 사용).

    Returns:
        :class:`LambdaLR`; 옵티마이저 스텝마다 한 번씩 ``.step()``을
        호출하라.
    """
    warmup = max(1, config.warmup_steps)
    # decay는 최대값 대비 비율로 표현된 min_lr에서 바닥을 친다.
    floor = min(1.0, config.min_lr / max(config.learning_rate, 1e-12))

    if config.scheduler == "constant":
        factor = _constant_factory(warmup)
    elif config.scheduler == "linear":
        factor = _linear_factory(warmup, total_steps, floor)
    elif config.scheduler == "cosine":
        factor = _cosine_factory(warmup, total_steps, floor, decay=config.cosine_decay)
    elif config.scheduler == "noam":
        factor = _noam_factory(warmup)
    else:
        raise ValueError(f"Unknown scheduler '{config.scheduler}'")

    return LambdaLR(optimizer, lr_lambda=factor)
