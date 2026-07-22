"""
===============================================================================
 파일: models/multi_head_attention.py
 목적:
    Scaled dot-product attention과 multi-head attention을 직접 구현
    (torch.nn.MultiheadAttention, F.scaled_dot_product_attention 사용하지
    않음) — 계산의 모든 단계가 눈에 보이도록.

 역할:
    Transformer의 핵심 혼합(mixing) 연산. 하나의 모듈이 세 가지 용도 모두를
    담당한다:
      - 인코더 self-attention  (query = key = value = 인코더 상태)
      - 디코더 self-attention  (causal 마스킹 적용)
      - 인코더-디코더 cross-attention (query = 디코더, key/value = memory)

 입력 / 출력:
    query : (batch, query_len, d_model)
    key   : (batch, key_len,   d_model)
    value : (batch, key_len,   d_model)
    mask  : 선택적 불리언, (batch, n_heads, query_len, key_len)에 브로드캐스트
            가능; True = 어텐션 가능 (규칙은 models/utils.py 참고).
    출력  : (batch, query_len, d_model)

 구현 세부사항:
    - 프로젝션 -> 헤드 분리 -> 어텐션 -> 헤드 결합 -> 출력 프로젝션의
      모든 단계가 helper 메서드 없이 forward() 안에 직접 작성되어 있다.
    - 헤드는 하나의 큰 프로젝션을 reshape해서 만든다: (B, L, d_model) ->
      (B, n_heads, L, d_head). 모든 헤드가 한 번의 matmul로 계산된다.
    - 마스킹된 위치는 dtype 최솟값을 채운다 (AMP에서도 안전; 한 행이
      전부 마스킹되어도 NaN 대신 균등 분포로 softmax된다).
    - `attention_scaling`은 True(1/sqrt(d_head)), False(스케일링 없음)
      또는 float(사용자 지정 스케일)이 될 수 있으며 설정으로 제어된다.
    - `causal=True`면 모듈이 스스로 subsequent 마스크를 만들어서, 이
      블록들을 디코더 전용 언어 모델에서 재사용할 수 있게 해준다.
    - 새 어텐션 메커니즘은 서브클래스를 만들고 ATTENTION_REGISTRY에
      등록하는 방식으로 추가한다; 설정의 `attention_type`이 어떤 항목을
      쓸지 선택한다.
===============================================================================
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor, nn

from models.utils import combine_masks, make_causal_mask


class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(Q K^T * scale + mask) V.

    Args:
        dropout: 어텐션 확률 맵에 적용하는 dropout.
        scale: 원본 점수에 곱하는 배율 (보통 ``1/sqrt(d_head)``).
    """

    def __init__(self, dropout: float = 0.0, scale: float = 1.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scale = scale

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """헤드별 텐서에 대해 어텐션을 계산한다.

        Args:
            query: ``(batch, n_heads, query_len, d_head)``.
            key:   ``(batch, n_heads, key_len,   d_head)``.
            value: ``(batch, n_heads, key_len,   d_head)``.
            mask:  ``(batch, n_heads, query_len, key_len)``에 브로드캐스트
                   가능한 선택적 불리언 마스크; True = 어텐션 가능.

        Returns:
            ``(context, attention_weights)`` 튜플. 형태는 각각
            ``(batch, n_heads, query_len, d_head)``,
            ``(batch, n_heads, query_len, key_len)``.
        """
        # ===================== 어텐션 점수 =====================
        # 모든 쿼리와 모든 키 사이의 유사도: (B, H, Lq, Lk).
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # ===================== 마스킹 =====================
        if mask is not None:
            # -inf가 아니라 dtype 최솟값을 써서, 행 전체가 마스킹되어도
            # softmax가 유한하게 나오고 float16 autocast에서 오버플로를
            # 피할 수 있다.
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        # ===================== Softmax + Dropout =====================
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # ===================== 가중 평균 =====================
        # value들의 가중 평균: (B, H, Lq, d_head).
        context = torch.matmul(attention, value)
        return context, attention


# 설정의 `attention.attention_type`을 구현체에 매핑하는 레지스트리.
# 예를 들어 linear나 local attention을 추가하려면 (dropout, scale)과
# 같은 생성자를 가진 클래스를 만들고 여기에 등록하면 된다.
ATTENTION_REGISTRY: dict[str, type[nn.Module]] = {
    "scaled_dot_product": ScaledDotProductAttention,
}


class MultiHeadAttention(nn.Module):
    """프로젝션, 스케일링, 마스킹을 설정으로 제어할 수 있는 multi-head attention.

    ``d_model``을 ``n_heads``개의 독립적인 부분공간으로 나누고, 각각에서
    어텐션을 수행한 뒤 결과를 이어 붙여 프로젝션한다. 여러 헤드 덕분에
    모델은 동시에 여러 종류의 관계를 어텐션할 수 있다.

    Args:
        d_model: 모델 폭 (``n_heads``로 나눠떨어져야 함).
        n_heads: 어텐션 헤드 개수.
        attention_dropout: 어텐션 확률 맵에 적용하는 dropout.
        qkv_bias: Q/K/V 및 출력 프로젝션의 bias 항.
        attention_scaling: True -> ``1/sqrt(d_head)``; False -> 1.0;
            float -> 해당 사용자 지정 값.
        attention_type: :data:`ATTENTION_REGISTRY`의 키.
        causal: True면 모듈 스스로 causal 마스킹을 강제한다
            (:meth:`forward`에 전달된 마스크와 결합됨).
        store_attention: True면 마지막(detach된) 어텐션 맵을
            ``self.last_attention``에 저장한다 (시각화/디버깅용).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        attention_scaling: Union[bool, float] = True,
        attention_type: str = "scaled_dot_product",
        causal: bool = False,
        store_attention: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.store_attention = store_attention
        self.last_attention: Optional[Tensor] = None

        # (bool | float) 설정 값으로부터 점수 스케일을 결정한다.
        if isinstance(attention_scaling, bool):
            scale = self.d_head**-0.5 if attention_scaling else 1.0
        else:
            scale = float(attention_scaling)

        if attention_type not in ATTENTION_REGISTRY:
            raise ValueError(
                f"Unknown attention_type '{attention_type}'. "
                f"Available: {sorted(ATTENTION_REGISTRY)}"
            )
        self.attention = ATTENTION_REGISTRY[attention_type](
            dropout=attention_dropout, scale=scale
        )

        # Q/K/V를 별도로 프로젝션 (하나의 합쳐진 행렬보다 명확하며,
        # 어차피 cross-attention은 Q와 K/V에 서로 다른 입력이 필요함).
        self.w_q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.w_k = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.w_v = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.w_o = nn.Linear(d_model, d_model, bias=qkv_bias)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """multi-head attention을 실행한다.

        프로젝션부터 출력까지 모든 단계가 이 함수 안에 직접 작성되어 있다.

        Args:
            query: ``(batch, query_len, d_model)``.
            key:   ``(batch, key_len,   d_model)``.
            value: ``(batch, key_len,   d_model)``.
            mask:  ``(batch, n_heads, query_len, key_len)``에 브로드캐스트
                   가능한 선택적 불리언 마스크; True = 어텐션 가능.

        Returns:
            ``(batch, query_len, d_model)`` 형태의 어텐션이 반영된 표현.
        """
        batch_size = query.size(0)
        query_len, key_len = query.size(1), key.size(1)

        # ===================== 1) Q/K/V 선형 프로젝션 =====================
        # (B, L, d_model) -> (B, L, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # ===================== 2) 헤드 분리 =====================
        # (B, L, d_model) -> (B, L, n_heads, d_head) -> (B, n_heads, L, d_head)
        q = q.view(batch_size, query_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, key_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, key_len, self.n_heads, self.d_head).transpose(1, 2)

        # ===================== 3) Causal 마스크 (옵션) =====================
        # 모듈 내부에서 선택적으로 causal 제약을 강제한다. 마스크는
        # 오른쪽 정렬(key_len - query_len만큼 오프셋)되어 있어서,
        # query_len < key_len인 증분(incremental) 디코딩에서도 계속
        # 올바르게 동작한다.
        if self.causal:
            offset = key_len - query_len
            causal = make_causal_mask(key_len, query.device)[:, :, offset:, :]
            mask = combine_masks(mask, causal)

        # ===================== 4) Scaled Dot-Product Attention =====================
        # 모든 헤드에서 한 번에 어텐션을 계산.
        # context: (B, n_heads, Lq, d_head), attention: (B, n_heads, Lq, Lk)
        context, attention = self.attention(q, k, v, mask)

        if self.store_attention:
            # detach해서 시각화가 autograd 그래프를 붙잡고 있지 않도록 한다.
            self.last_attention = attention.detach()

        # ===================== 5) 헤드 결합 =====================
        # (B, n_heads, Lq, d_head) -> (B, Lq, n_heads, d_head) -> (B, Lq, d_model)
        # transpose 이후 view()를 하려면 contiguous()가 필요하다.
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, self.n_heads * self.d_head)
        )

        # ===================== 6) 출력 프로젝션 =====================
        return self.w_o(context)
