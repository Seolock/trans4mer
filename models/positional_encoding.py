"""
===============================================================================
 파일: models/positional_encoding.py
 목적:
    토큰 임베딩에 위치 정보를 주입한다. self-attention은 순열에 불변
    (permutation-invariant)이므로, 이것이 없으면 모델은 단어 순서를 전혀
    알 수 없다.

 역할:
    설정(`model.positional_encoding_type`)으로 선택 가능한 세 가지 방식을
    제공한다:
      - "sinusoidal": 고정된 sin/cos 특징 (원 논문 방식, 파라미터 없음,
        max_len까지 어떤 길이로도 확장 가능).
      - "learned"   : 학습 가능한 위치-임베딩 테이블 (BERT/GPT 스타일).
      - "none"      : 항등 함수 (ablation / 상대 위치 변형 실험용).

 입력 / 출력:
    입력 : (batch, seq_len, d_model) 토큰 임베딩.
    출력 : (batch, seq_len, d_model) 위치 정보가 더해진 임베딩.

 구현 세부사항:
    - Sinusoidal: PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
                  PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
      한 번만 (1, max_len, d_model) 버퍼로 미리 계산해두고, 체크포인트에
      저장하는 대신 설정으로부터 재구성할 수 있도록 non-persistent
      버퍼로 등록한다 (그래도 .to(device) 시 함께 이동함).
    - Learned: 절대 위치로 인덱싱하는 평범한 nn.Embedding; max_len을 넘는
      길이로는 일반화되지 않으므로 시퀀스 길이를 검증한다.
===============================================================================
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """"Attention Is All You Need"의 고정 sinusoidal 위치 인코딩.

    각 차원 쌍 (2i, 2i+1)은 기하급수적으로 증가하는 파장을 가진 사인파를
    형성하며, 이 특징들의 선형 조합을 통해 모델이 상대적 오프셋에
    어텐션할 수 있게 해준다.

    Args:
        d_model: 임베딩 차원.
        max_len: 미리 계산해둘 최대 시퀀스 길이.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        # position: (max_len, 1); div_term: (ceil(d_model/2),)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        # d_model이 홀수여도 동작하도록 div_term을 잘라준다.
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        # 버퍼(Parameter 아님): 모듈과 함께 이동하지만 학습되지는 않음.
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """``x``에 위치 특징을 더한다.

        Args:
            x: ``(batch, seq_len, d_model)`` 형태의 임베딩.

        Returns:
            동일한 형태의 ``x + PE[:seq_len]``.
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)} "
                "— increase model.max_seq_length in the config."
            )
        return x + self.pe[:, :seq_len]


class LearnedPositionalEncoding(nn.Module):
    """학습 가능한 절대 위치 임베딩 (BERT/GPT 스타일).

    Args:
        d_model: 임베딩 차원.
        max_len: 테이블에 담을 위치 슬롯 개수.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """``x``에 학습된 위치 벡터를 더한다.

        Args:
            x: ``(batch, seq_len, d_model)`` 형태의 임베딩.

        Returns:
            동일한 형태의 ``x + position_embedding``.
        """
        seq_len = x.size(1)
        if seq_len > self.embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds the learned position table "
                f"({self.embedding.num_embeddings}) — increase model.max_seq_length."
            )
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


def build_positional_encoding(pe_type: str, d_model: int, max_len: int) -> nn.Module:
    """설정으로부터 positional-encoding 전략을 선택하는 팩토리 함수.

    Args:
        pe_type: "sinusoidal", "learned" 또는 "none".
        d_model: 임베딩 차원.
        max_len: 지원하는 최대 시퀀스 길이.

    Returns:
        임베딩 -> 위치 정보가 반영된 임베딩으로 매핑하는 forward를 가진
        모듈. ("none"은 ``nn.Identity``를 반환한다 — 위치가 그냥 더해지지
        않음.)
    """
    if pe_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len)
    if pe_type == "learned":
        return LearnedPositionalEncoding(d_model, max_len)
    if pe_type == "none":
        return nn.Identity()
    raise ValueError(f"Unknown positional_encoding_type '{pe_type}'")
