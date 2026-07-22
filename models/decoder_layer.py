"""
===============================================================================
 파일: models/decoder_layer.py
 목적:
    Transformer 디코더 레이어 하나. 텍스트-only 모드에서는 세 서브레이어:
      1. 타겟 prefix에 대한 causal 마스킹된 self-attention,
      2. 인코더 출력("memory")에 대한 (텍스트) cross-attention,
      3. position-wise feed-forward network.
    Multimodal(use_image) 모드에서는 텍스트 cross-attention과 나란히
    이미지 인코더 출력에 대한 Image Cross-Attention을 하나 더 두고, 두
    출력을 Fusion 모듈로 합친다:
      1. Self-Attention
      2. Text Cross-Attention  +  Image Cross-Attention  ->  Fusion
      3. Feed Forward
    각 서브레이어는 residual connection과 layer normalization으로 감싸진다.

 역할:
    디코더 스택에서 반복되는 단위. self-attention은 각 타겟 위치가 이전
    타겟 위치들만 참고하게 한다(mask_future). Text Cross-Attention은 소스
    문장 정보가, Image Cross-Attention은 이미지 정보가 디코더로 들어오는
    지점이다 (두 어텐션 모두 쿼리는 디코더 hidden state, 키/값은 각각
    텍스트 memory / 이미지 memory에서 온다). Fusion이 둘을 하나의 표현으로
    합쳐 residual 스트림으로 돌려보낸다.

 입력 / 출력:
    x           : (batch, tgt_len, d_model)   디코더 측 표현
    memory      : (batch, src_len, d_model)   텍스트 인코더 출력
    tgt_mask    : (batch, n_heads, tgt_len, tgt_len)에 브로드캐스트 가능한
                  불리언 — causal 마스크와 타겟-패딩 마스크가 결합된 것.
    memory_mask : (batch, n_heads, tgt_len, src_len)에 브로드캐스트 가능한
                  불리언 — 텍스트 cross-attention을 위한 소스-패딩 마스크.
    image_memory: (batch, num_patches, d_model) 이미지 인코더 출력
                  (use_image가 아니거나 이미지가 없으면 None).
    image_mask  : 이미지 cross-attention용 선택적 마스크 (패치 패딩이
                  있을 때만; 보통 None — 모든 패치가 유효).
    출력        : (batch, tgt_len, d_model)

 구현 세부사항:
    - 모든 계산(LayerNorm, Attention, Dropout, Residual 덧셈, Fusion, FFN)은
      helper 메서드에 숨기지 않고 forward() 안에 직접 작성되어 있어,
      forward()만 읽어도 전체 흐름이 보인다.
    - 정규화 배치("pre" vs "post")는 encoder_layer.py와 정확히 동일하다.
    - Text/Image 두 cross-attention은 동일한 쿼리(정규화된 디코더 hidden
      state)를 공유하며, cross_attention_norm 하나를 함께 사용한다 —
      두 어텐션의 쿼리가 모두 "디코더 hidden state"이기 때문이다. Fusion
      이후의 residual/LayerNorm은 기존 Transformer와 동일하게 유지된다.
    - use_image=false면 image_cross_attention/fusion을 아예 만들지 않아
      텍스트-only Transformer와 state_dict가 완전히 동일하다.
===============================================================================
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from config.config import Config
from models.feed_forward import PositionwiseFeedForward
from models.fusion import build_fusion
from models.layer_norm import LayerNorm
from models.multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """디코더 레이어 하나: self-attention, (텍스트[+이미지]) cross-attention, feed-forward.

    Args:
        config: 전체 프로젝트 설정 (model, attention, multimodal 섹션 사용됨).
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        m, a, mm = config.model, config.attention, config.multimodal
        self.norm_first = m.norm_style == "pre"
        self.use_image = mm.use_image

        def build_attention(causal: bool) -> MultiHeadAttention:
            """어텐션 모듈들은 causal 여부를 제외한 모든 설정을 공유한다."""
            return MultiHeadAttention(
                d_model=m.d_model,
                n_heads=m.n_heads,
                attention_dropout=m.attention_dropout,
                qkv_bias=a.qkv_bias,
                attention_scaling=a.attention_scaling,
                attention_type=a.attention_type,
                causal=causal,
                store_attention=a.store_attention,
            )

        # self-attention은 둘 중 하나의 플래그가 요청하면 causal이 된다;
        # 디코더는 추가로 명시적인 causal 마스크를 만들기 때문에
        # (models/transformer.py), 여기서 `a.causal`은 주로 디코더 전용
        # 재사용을 위한 것이다.
        self.self_attention = build_attention(causal=a.causal)
        self.cross_attention = build_attention(causal=False)  # 텍스트 cross-attention
        self.feed_forward = PositionwiseFeedForward(
            d_model=m.d_model,
            dim_feedforward=m.dim_feedforward,
            dropout=m.dropout,
            activation=m.activation,
            bias=m.bias,
        )
        self.self_attention_norm = LayerNorm(m.d_model, eps=m.layer_norm_eps, bias=m.bias)
        self.cross_attention_norm = LayerNorm(m.d_model, eps=m.layer_norm_eps, bias=m.bias)
        self.feed_forward_norm = LayerNorm(m.d_model, eps=m.layer_norm_eps, bias=m.bias)
        self.residual_dropout = nn.Dropout(m.residual_dropout)

        # ---------------------------------------------- Multimodal 확장 (옵션)
        # use_image일 때만 이미지 cross-attention과 fusion을 만든다; 그래야
        # 텍스트-only 체크포인트의 state_dict가 그대로 유지된다.
        if self.use_image:
            self.image_cross_attention = build_attention(causal=False)
            self.fusion = build_fusion(mm.fusion_type, m.d_model, mm)

    def _cross_attend(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        image_memory: Optional[Tensor],
        image_mask: Optional[Tensor],
    ) -> Tensor:
        """Text (+ Image) cross-attention을 수행하고 결과를 Fusion으로 합친다.

        두 어텐션 모두 같은 쿼리(정규화/미정규화된 디코더 hidden state)를
        쓰고, 텍스트는 인코더 memory를, 이미지는 이미지 memory를 키/값으로
        어텐션한다. use_image가 아니거나 image_memory가 없으면 텍스트
        cross-attention 결과를 그대로 반환한다 (텍스트-only와 동일).

        Args:
            query: ``(batch, tgt_len, d_model)`` cross-attention 쿼리.
            memory: ``(batch, src_len, d_model)`` 텍스트 인코더 출력.
            memory_mask: 텍스트 cross-attention 소스-패딩 마스크.
            image_memory: ``(batch, num_patches, d_model)`` 이미지 인코더 출력 또는 None.
            image_mask: 이미지 cross-attention 마스크 또는 None.

        Returns:
            ``(batch, tgt_len, d_model)`` (융합된) cross-attention 출력.
        """
        text_output = self.cross_attention(
            query=query, key=memory, value=memory, mask=memory_mask
        )
        if not (self.use_image and image_memory is not None):
            return text_output
        image_output = self.image_cross_attention(
            query=query, key=image_memory, value=image_memory, mask=image_mask
        )
        return self.fusion(text_output, image_output)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        image_memory: Optional[Tensor] = None,
        image_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """설정된 정규화 배치에 따라 세 서브레이어를 실행한다.

        전체 데이터 흐름이 이 함수 안에 직접 작성되어 있다:
        residual 저장 -> (LayerNorm) -> 서브레이어 -> Dropout ->
        residual 덧셈 -> (LayerNorm) 을 세 서브레이어에 대해 반복. 여기서
        cross-attention 서브레이어는 Text/Image 두 어텐션 + Fusion으로
        확장된다 (:meth:`_cross_attend`).

        Args:
            x: ``(batch, tgt_len, d_model)`` 타겟 측 표현.
            memory: ``(batch, src_len, d_model)`` 텍스트 인코더 출력.
            tgt_mask: causal + 타겟-패딩이 결합된 마스크.
            memory_mask: 텍스트 cross-attention을 위한 소스-패딩 마스크.
            image_memory: ``(batch, num_patches, d_model)`` 이미지 인코더 출력 또는 None.
            image_mask: 이미지 cross-attention 마스크 또는 None.

        Returns:
            ``(batch, tgt_len, d_model)`` 정제된 타겟 표현.
        """
        if self.norm_first:
            # ====================== Self-Attention (Pre-LN) ======================
            # LayerNorm -> 마스킹된 Self-Attention -> Dropout -> Residual 덧셈.
            # 타겟 prefix 내부에서만 정보를 섞는다 (미래 위치는 tgt_mask로 차단).
            residual = x
            normed = self.self_attention_norm(x)
            attended = self.self_attention(query=normed, key=normed, value=normed, mask=tgt_mask)
            x = residual + self.residual_dropout(attended)

            # ============ Text (+ Image) Cross-Attention + Fusion (Pre-LN) =======
            # LayerNorm -> [Text Cross-Attn (+ Image Cross-Attn -> Fusion)] ->
            # Dropout -> Residual 덧셈. 두 어텐션은 같은 정규화된 쿼리를 쓰며,
            # 쿼리는 디코더에서, 키/값은 각각 텍스트/이미지 memory에서 온다.
            residual = x
            normed = self.cross_attention_norm(x)
            attended = self._cross_attend(normed, memory, memory_mask, image_memory, image_mask)
            x = residual + self.residual_dropout(attended)

            # ======================= Feed Forward (Pre-LN) =======================
            # LayerNorm -> FFN -> Dropout -> Residual 덧셈.
            residual = x
            normed = self.feed_forward_norm(x)
            hidden = self.feed_forward(normed)
            x = residual + self.residual_dropout(hidden)
        else:
            # ====================== Self-Attention (Post-LN) =====================
            # 마스킹된 Self-Attention -> Dropout -> Residual 덧셈 -> LayerNorm.
            residual = x
            attended = self.self_attention(query=x, key=x, value=x, mask=tgt_mask)
            x = self.self_attention_norm(residual + self.residual_dropout(attended))

            # =========== Text (+ Image) Cross-Attention + Fusion (Post-LN) =======
            # [Text Cross-Attn (+ Image Cross-Attn -> Fusion)] -> Dropout ->
            # Residual 덧셈 -> LayerNorm.
            residual = x
            attended = self._cross_attend(x, memory, memory_mask, image_memory, image_mask)
            x = self.cross_attention_norm(residual + self.residual_dropout(attended))

            # ======================= Feed Forward (Post-LN) ======================
            # FFN -> Dropout -> Residual 덧셈 -> LayerNorm.
            residual = x
            hidden = self.feed_forward(x)
            x = self.feed_forward_norm(residual + self.residual_dropout(hidden))
        return x
