"""
===============================================================================
 파일: utils/visualization.py
 목적:
    학습된 모델을 살펴보기 위한 Matplotlib 헬퍼: 어텐션 히트맵과 간단한
    학습 곡선 플롯.

 역할:
    어텐션 맵은 설정에서 `attention.store_attention: true`일 때
    MultiHeadAttention이 캡처한다 (각 모듈이 그때
    (batch, n_heads, query_len, key_len) 형태의 `module.last_attention`을
    노출함). 한 예제의 맵을 이 헬퍼들에 넘겨서 토큰 대 토큰 그리드를
    렌더링한다.

 입력 / 출력:
    plot_attention_heads : (n_heads, query_len, key_len) 텐서/배열 ->
                           헤드마다 하나씩 히트맵이 있는 figure.
    plot_training_curves : {"name": [values...]} -> 하나의 라인 차트.
    `save_path`가 주어지면 figure가 디스크에 저장되며, 항상 반환도 된다.

 구현 세부사항:
    헤드리스(headless) 학습 서버에서도 플로팅이 동작하도록 비대화형
    "Agg" 백엔드를 강제한다; 노트북에서 보고 싶다면 직접 plt.show()를
    호출하라.
===============================================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # 헤드리스 환경에서 안전; pyplot import보다 먼저 와야 함
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def _to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    """텐서나 배열을 받아; CPU 상의 float numpy 배열을 반환한다."""
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def plot_attention_heads(
    attention: Tensor | np.ndarray,
    x_tokens: Sequence[str],
    y_tokens: Sequence[str],
    title: str = "Attention",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """어텐션 헤드마다 하나씩 히트맵을 그려 그리드로 렌더링한다.

    Args:
        attention: 단일 예제에 대한 ``(n_heads, query_len, key_len)`` 형태의
            어텐션 가중치 (배치 차원은 미리 인덱싱해서 빼야 함).
        x_tokens: key 측 토큰 문자열 (열), 길이는 ``key_len``.
        y_tokens: query 측 토큰 문자열 (행), 길이는 ``query_len``.
        title: figure 제목.
        save_path: 지정하면 그 경로에 PNG로 저장한다 (디렉터리는 자동 생성).

    Returns:
        matplotlib figure.
    """
    weights = _to_numpy(attention)
    if weights.ndim != 3:
        raise ValueError(f"Expected (n_heads, query_len, key_len), got shape {weights.shape}")

    n_heads = weights.shape[0]
    # 헤드 개수에 맞춰 가능한 한 정사각형에 가까운 subplot 그리드를 만든다.
    cols = math.ceil(math.sqrt(n_heads))
    rows = math.ceil(n_heads / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    fig.suptitle(title)

    for head in range(rows * cols):
        ax = axes[head // cols][head % cols]
        if head >= n_heads:
            ax.axis("off")
            continue
        ax.imshow(weights[head], aspect="auto", cmap="viridis")
        ax.set_title(f"head {head}", fontsize=9)
        ax.set_xticks(range(len(x_tokens)))
        ax.set_xticklabels(x_tokens, rotation=90, fontsize=7)
        ax.set_yticks(range(len(y_tokens)))
        ax.set_yticklabels(y_tokens, fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    curves: dict[str, Sequence[float]],
    title: str = "Training curves",
    xlabel: str = "epoch",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """하나 이상의 이름이 붙은 지표 시계열을 하나의 축에 그린다.

    Args:
        curves: 시리즈 이름을 epoch/step별 값 리스트에 매핑한 것.
        title: figure 제목.
        xlabel: x축 라벨.
        save_path: 지정하면 그 경로에 PNG로 저장한다.

    Returns:
        matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, values in curves.items():
        ax.plot(range(1, len(values) + 1), list(values), label=name, marker="o", markersize=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
