"""
===============================================================================
 파일: models/utils.py
 목적:
    모든 모델 컴포넌트가 공유하는 자잘하지만 재사용 가능한 헬퍼들.

 역할:
    - 활성화 함수 레지스트리 (이름 -> nn.Module 팩토리).
    - 어텐션 마스크 생성 (패딩 마스크와 causal 마스크).
    - 가중치 초기화.

 입력 / 출력 (프로젝트 전체에서 사용하는 마스크 규칙):
    어텐션 점수 (batch, n_heads, query_len, key_len)에 브로드캐스트 가능한
    불리언 마스크:
        True  -> 해당 위치를 어텐션할 수 있음
        False -> 해당 위치는 마스킹됨
    - make_pad_mask   : (batch, seq_len)        -> (batch, 1, 1, seq_len)
    - make_causal_mask: 정수 길이                -> (1, 1, length, length)
    - combine_masks   : 여러 개의 브로드캐스트 가능한 마스크의 논리 AND.

 구현 세부사항:
    마스킹된 위치는 softmax 전에 dtype 최솟값으로 채워진다 (-inf가 아님).
    이는 혼합 정밀도(mixed precision)에서도 수치적으로 안전하며, 어떤 행이
    완전히 마스킹되더라도 gradient를 오염시키는 NaN 대신 유한한(균등한)
    분포를 만들어낸다.
===============================================================================
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor, nn

# 지원하는 활성화 함수 레지스트리. 새로 추가하려면 이름 -> 생성자 매핑을
# 여기에 등록하면 YAML 설정에서 바로 쓸 수 있다.
ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def get_activation(name: str) -> nn.Module:
    """설정에 지정된 이름으로 활성화 모듈을 생성한다.

    Args:
        name: :data:`ACTIVATIONS`의 키 (대소문자 구분 없음).

    Returns:
        새로 생성된 활성화 ``nn.Module``.

    Raises:
        ValueError: 등록되지 않은 이름일 경우.
    """
    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Available: {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[key]()


def make_pad_mask(token_ids: Tensor, pad_id: int) -> Tensor:
    """토큰 id 배치로부터 key-padding 마스크를 만든다.

    Args:
        token_ids: ``(batch, seq_len)`` 형태의 정수 텐서.
        pad_id: 패딩 토큰의 id.

    Returns:
        ``(batch, 1, 1, seq_len)`` 형태의 불리언 텐서 — 실제 토큰이면 True.
        두 개의 singleton 차원은 헤드와 쿼리 위치에 걸쳐 브로드캐스트된다.
    """
    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(length: int, device: torch.device) -> Tensor:
    """하삼각(lower-triangular, 미래 위치 마스킹) 마스크를 만든다.

    위치 ``i``는 ``j <= i``인 위치만 어텐션할 수 있어, teacher forcing 중
    디코더가 미래 토큰을 미리 엿보는 것을 막는다.

    Args:
        length: 정사각형 마스크의 시퀀스 길이.
        device: 마스크를 할당할 디바이스.

    Returns:
        ``(1, 1, length, length)`` 형태의 불리언 텐서.
    """
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    return causal.unsqueeze(0).unsqueeze(1)


def combine_masks(*masks: Optional[Tensor]) -> Optional[Tensor]:
    """브로드캐스트 가능한 여러 마스크를 논리 AND로 결합하며 None은 건너뛴다.

    모든 입력이 ``None``이면 (즉 "마스킹 없음"이면) ``None``을 반환한다.
    """
    result: Optional[Tensor] = None
    for mask in masks:
        if mask is None:
            continue
        result = mask if result is None else result & mask
    return result


def init_xavier(model: nn.Module) -> None:
    """``model``의 모든 가중치 행렬에 Xavier-uniform 초기화를 적용한다.

    벡터(bias, LayerNorm 게인)는 기본 초기화를 그대로 유지한다 — Xavier는
    fan-in/fan-out이 있는 행렬에만 의미가 있기 때문이다.
    """
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
