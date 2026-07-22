"""
===============================================================================
 파일: models/feed_forward.py
 목적:
    Position-wise feed-forward network (FFN) — 모든 인코더/디코더
    레이어의 두 번째 서브레이어.

 역할:
    같은 두-레이어 MLP를 모든 위치에 독립적으로 적용한다:
        FFN(x) = W2 · act(W1 · x + b1) + b2
    어텐션이 위치들 *사이에서* 정보를 섞는 반면, FFN은 각 위치의 표현을
    독자적으로 변환하며, 더 넓은 은닉 공간(dim_feedforward, 보통
    d_model의 4배)으로 확장했다가 다시 원래 크기로 프로젝션한다.

 입력 / 출력:
    입력 : (batch, seq_len, d_model)
    출력 : (batch, seq_len, d_model)

 구현 세부사항:
    - 활성화 함수는 models/utils.py의 레지스트리에서 조회되므로,
      relu -> gelu -> silu로 바꾸는 것은 설정 한 줄만 바꾸면 된다.
    - dropout은 활성화와 다운-프로젝션 사이에 위치한다 (원 논문의 참조
      구현들이 사용하는 배치).
===============================================================================
"""

from __future__ import annotations

from torch import Tensor, nn

from models.utils import get_activation


class PositionwiseFeedForward(nn.Module):
    """설정 가능한 활성화 함수를 가진 두-레이어 position-wise MLP.

    Args:
        d_model: 입력/출력 폭.
        dim_feedforward: 은닉(확장된) 폭.
        dropout: 활성화 이후에 적용하는 dropout.
        activation: 레지스트리의 활성화 함수 이름 ("relu", "gelu", ...).
        bias: 두 선형 레이어가 bias 항을 쓰는지 여부.
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.up_projection = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.down_projection = nn.Linear(dim_feedforward, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """각 위치를 독립적으로 변환한다.

        Args:
            x: ``(batch, seq_len, d_model)``.

        Returns:
            ``(batch, seq_len, d_model)``.
        """
        # ===================== 1) 확장 프로젝션 =====================
        # (B, L, d_model) -> (B, L, dim_feedforward)
        hidden = self.up_projection(x)

        # ===================== 2) 비선형 활성화 =====================
        hidden = self.activation(hidden)

        # ===================== 3) Dropout =====================
        hidden = self.dropout(hidden)

        # ===================== 4) 축소 프로젝션 =====================
        # (B, L, dim_feedforward) -> (B, L, d_model)
        return self.down_projection(hidden)
