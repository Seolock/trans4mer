"""
===============================================================================
 파일: models/embedding.py
 목적:
    토큰 임베딩과 Transformer의 "입력 레이어"를 구성하는 부분
    (토큰 임베딩 + positional encoding + 임베딩 dropout).

 역할:
    - `TokenEmbedding`: id -> 밀집 벡터 조회, 원 논문처럼 선택적으로
      sqrt(d_model)만큼 스케일링 가능 (임베딩 크기를 sinusoidal 위치
      특징과 비슷한 스케일로 맞춰줌).
    - `TransformerEmbedding`: 인코더 입력과 디코더 입력에 적용되는 전체
      파이프라인. 두 인스턴스가 하나의 `TokenEmbedding`을 공유해서
      `share_embedding`을 구현할 수 있으며, 이 밑바탕 가중치 행렬이
      바로 출력 프로젝션과 묶이는(tie) 대상이다.

 입력 / 출력:
    입력 : (batch, seq_len) int64 토큰 id.
    출력 : (batch, seq_len, d_model) 실수 임베딩.

 구현 세부사항:
    - `padding_idx`는 패딩 벡터를 0으로 만들고 gradient를 얼려서, 패딩
      위치가 아무 신호도 전달하지 않게 한다.
    - 전역 Xavier 초기화가 임베딩 테이블을 덮어쓰기 때문에, Transformer는
      초기화 후 `reset_parameters()`를 호출해 fairseq 방식(normal(0,
      d_model^-0.5))으로 임베딩을 다시 초기화하고 패딩 행을 0으로 복원한다.
===============================================================================
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """선택적 sqrt(d_model) 스케일링을 지원하는 어휘집 임베딩 조회.

    Args:
        vocab_size: 임베딩 테이블의 행 개수.
        d_model: 임베딩 차원.
        pad_id: 패딩-토큰 인덱스 (해당 벡터는 0으로 고정되고 학습되지 않음).
        scale: True면 임베딩에 ``sqrt(d_model)``을 곱한다.
    """

    def __init__(self, vocab_size: int, d_model: int, pad_id: int, scale: bool = True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pad_id = pad_id
        self.scale_factor = math.sqrt(d_model) if scale else 1.0

    @property
    def weight(self) -> nn.Parameter:
        """원본 임베딩 행렬 — 출력 프로젝션 weight tying에 사용된다."""
        return self.embedding.weight

    def reset_padding_vector(self) -> None:
        """패딩 행을 다시 0으로 만든다 (전역 재초기화 이후에 호출)."""
        with torch.no_grad():
            self.embedding.weight[self.pad_id].fill_(0.0)

    def reset_parameters(self) -> None:
        """토큰 임베딩을 fairseq 방식으로 초기화한다.

        전역 Xavier 초기화(models/utils.py:init_xavier)가 임베딩 테이블까지
        덮어쓰므로, Transformer가 그 이후에 이 메서드를 호출해 임베딩만
        fairseq의 ``normal_(0, d_model^-0.5)``로 다시 초기화하고 패딩 행을
        0으로 되돌린다. Linear/Conv 등 다른 행렬은 Xavier를 그대로 유지한다.
        """
        with torch.no_grad():
            nn.init.normal_(
                self.embedding.weight, mean=0.0, std=self.embedding.embedding_dim**-0.5
            )
        self.reset_padding_vector()

    def forward(self, token_ids: Tensor) -> Tensor:
        """임베딩을 조회(하고 스케일링)한다.

        Args:
            token_ids: ``(batch, seq_len)`` 형태의 int64 id.

        Returns:
            ``(batch, seq_len, d_model)`` 형태의 실수 임베딩.
        """
        return self.embedding(token_ids) * self.scale_factor


class TransformerEmbedding(nn.Module):
    """완전한 입력 레이어: 토큰 임베딩 + positional encoding + dropout.

    토큰-임베딩 모듈은 (여기서 직접 생성하지 않고) 주입받는 방식이라서,
    `share_embedding`이 켜졌을 때 인코더와 디코더가 하나의 테이블을 공유할
    수 있다.

    Args:
        token_embedding: (공유될 수도 있는) :class:`TokenEmbedding`.
        positional_encoding: 위치 정보를 더하는 모듈
            (models/positional_encoding.py 참고).
        dropout: 임베딩-dropout 확률.
    """

    def __init__(
        self,
        token_embedding: TokenEmbedding,
        positional_encoding: nn.Module,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.positional_encoding = positional_encoding
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: Tensor) -> Tensor:
        """토큰 id 배치를 임베딩한다.

        Args:
            token_ids: ``(batch, seq_len)`` 형태의 int64 id.

        Returns:
            ``(batch, seq_len, d_model)`` 형태의 위치 정보 반영 임베딩.
        """
        # ===================== 1) 토큰 임베딩 =====================
        # id 조회 (+ 설정 시 sqrt(d_model) 스케일링): (B, L) -> (B, L, d_model)
        embedded = self.token_embedding(token_ids)

        # ===================== 2) 위치 인코딩 =====================
        # sinusoidal / learned / none 중 설정된 방식으로 위치 정보를 더한다.
        embedded = self.positional_encoding(embedded)

        # ===================== 3) 임베딩 Dropout =====================
        return self.dropout(embedded)
