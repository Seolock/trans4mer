"""
===============================================================================
 파일: models/transformer.py
 목적:
    이 패키지의 구성 요소들로 조립한 완전한 sequence-to-sequence
    Transformer (torch.nn.Transformer는 어디에도 쓰지 않음).

 역할:
    다음을 소유하고 연결한다:
      - 소스/타겟 임베딩 파이프라인 (가중치 공유 옵션 포함),
      - 인코더와 디코더 스택,
      - 어휘집으로의 최종 선형 프로젝션 ("generator", 선택적으로 디코더
        임베딩과 가중치 공유(tie) 가능),
      - 원본 토큰 id로부터 마스크(패딩 + causal) 생성.

 입력 / 출력:
    forward(src, tgt):
        src: (batch, src_len) int64 소스 토큰 id (오른쪽 패딩)
        tgt: (batch, tgt_len) int64 타겟-prefix id (BOS로 시작)
        ->   (batch, tgt_len, vocab_size) 정규화되지 않은 logits
    encode()/decode()는 추론을 위해 두 부분을 각각 노출하며, 이때 소스는
    한 번만 인코딩되고 디코더는 스텝별로 호출된다.

 구현 세부사항:
    - forward()에는 마스크 생성 -> 인코더 -> 디코더 -> 프로젝션의 전체
      데이터 흐름이 helper 없이 직접 작성되어 있다; encode()/decode()는
      추론(빔 서치 / greedy)이 두 절반을 따로 실행하기 위한 공개 API다.
    - 가중치 공유 (fairseq 스타일 시맨틱):
        share_embedding         -> 인코더와 디코더가 하나의 토큰 테이블 공유
        share_decoder_embedding -> 디코더 입력 임베딩을 출력 프로젝션과
                                   묶음 (tie_output_projection을 함의)
        tie_output_projection   -> generator.weight가 디코더 임베딩
                                   가중치와 *같은* Parameter 객체가 됨
    - Xavier 초기화가 모든 행렬에 걸쳐 실행된 뒤, 임베딩 패딩 행을 다시
      0으로 만든다 (그렇지 않으면 Xavier가 패딩 토큰에도 신호를 줄 것임).
    - 마스크는 프로젝트 전역 규칙(True = 어텐션 가능)을 따른다;
      models/utils.py 참고.
===============================================================================
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from config.config import Config
from models.decoder import Decoder
from models.embedding import TokenEmbedding, TransformerEmbedding
from models.encoder import Encoder
from models.positional_encoding import build_positional_encoding
from models.utils import combine_masks, init_xavier, make_causal_mask, make_pad_mask


class Transformer(nn.Module):
    """sequence-to-sequence 작업을 위한 완전한 인코더-디코더 Transformer.

    모든 아키텍처 선택(깊이, 폭, 정규화 방식, 공유, 마스킹, 위치 인코딩 등)은
    :class:`Config`로부터 읽어오므로 어떤 것도 하드코딩되어 있지 않다.

    Args:
        config: 전체 프로젝트 설정.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        m = config.model
        self.pad_id = m.pad_token_id
        # 디코더 self-attention은 명시적으로 비활성화하지 않는 한 causal이다.
        self.mask_future = config.attention.mask_future or config.attention.causal

        # ------------------------------------------------------- 임베딩
        # 레거시 `learned_position_embedding` 플래그가 타입을 override한다.
        pe_type = "learned" if m.learned_position_embedding else m.positional_encoding_type

        # 언어별 분리 어휘집 지원: src/tgt 크기가 따로 설정되지 않으면
        # 공용 vocab_size를 사용한다 (기존 동작과 동일).
        src_vocab_size = m.src_vocab_size if m.src_vocab_size is not None else m.vocab_size
        tgt_vocab_size = m.tgt_vocab_size if m.tgt_vocab_size is not None else m.vocab_size

        src_token_embedding = TokenEmbedding(
            src_vocab_size, m.d_model, pad_id=self.pad_id, scale=m.scale_embedding
        )
        # `share_embedding`: 두 번째 테이블을 만드는 대신 디코더 측에서도
        # *같은* 모듈(즉 같은 가중치)을 재사용한다. 어휘집 크기가 다르면
        # 공유가 불가능하다.
        if m.share_embedding:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError(
                    "share_embedding=true requires equal src/tgt vocab sizes "
                    f"(got {src_vocab_size} vs {tgt_vocab_size})"
                )
            tgt_token_embedding = src_token_embedding
        else:
            tgt_token_embedding = TokenEmbedding(
                tgt_vocab_size, m.d_model, pad_id=self.pad_id, scale=m.scale_embedding
            )
        self.src_embedding = TransformerEmbedding(
            src_token_embedding,
            build_positional_encoding(pe_type, m.d_model, m.max_seq_length),
            dropout=m.embedding_dropout,
        )
        self.tgt_embedding = TransformerEmbedding(
            tgt_token_embedding,
            build_positional_encoding(pe_type, m.d_model, m.max_seq_length),
            dropout=m.embedding_dropout,
        )

        # ------------------------------------------------------------ 스택
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # -------------------------------------------------------- generator
        # d_model에서 어휘집 logits로의 최종 프로젝션. 가중치가 묶여있을
        # 때는 그 가중치가 말 그대로 디코더 임베딩 행렬이므로(bias는
        # 꺼야 함), 그래야 이 매핑이 임베딩 조회의 정확한 전치가 유지된다.
        # 출력 차원은 항상 타겟 어휘집 크기다.
        tie_generator = m.tie_output_projection or m.share_decoder_embedding
        self.generator = nn.Linear(m.d_model, tgt_vocab_size, bias=m.bias and not tie_generator)

        # ---------------------------------------------------------- 초기화
        init_xavier(self)
        src_token_embedding.reset_padding_vector()
        tgt_token_embedding.reset_padding_vector()
        if tie_generator:
            # 초기화 *이후에* 대입해서 tying이 유지되도록 한다; 이제 두
            # 모듈이 같은 Parameter 객체를 가지며 gradient가 합산된다.
            self.generator.weight = tgt_token_embedding.weight

    # ------------------------------------------------------------------ 마스크
    def make_source_mask(self, src: Tensor) -> Tensor:
        """인코더 self-attention과 cross-attention을 위한 패딩 마스크.

        Args:
            src: ``(batch, src_len)`` 소스 id.

        Returns:
            불리언 마스크 ``(batch, 1, 1, src_len)``; True = 실제 토큰.
        """
        return make_pad_mask(src, self.pad_id)

    def make_target_mask(self, tgt: Tensor) -> Tensor:
        """디코더 self-attention을 위한 패딩 + causal 결합 마스크.

        Args:
            tgt: ``(batch, tgt_len)`` 타겟-prefix id.

        Returns:
            불리언 마스크 ``(batch, 1, tgt_len, tgt_len)``.
        """
        pad_mask = make_pad_mask(tgt, self.pad_id)
        if not self.mask_future:
            return pad_mask
        causal = make_causal_mask(tgt.size(1), tgt.device)
        return combine_masks(pad_mask, causal)

    # ---------------------------------------------------------------- 절반씩
    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """소스를 임베딩하고 memory로 인코딩한다.

        Args:
            src: ``(batch, src_len)`` 소스 id.
            src_mask: 미리 계산된 소스 마스크, 또는 None이면 여기서 생성.

        Returns:
            ``(batch, src_len, d_model)`` 인코더 memory.
        """
        if src_mask is None:
            src_mask = self.make_source_mask(src)
        return self.encoder(self.src_embedding(src), src_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """타겟 prefix를 임베딩하고 미리 계산된 memory에 대해 디코딩한다.

        Args:
            tgt: ``(batch, tgt_len)`` 타겟-prefix id.
            memory: ``(batch, src_len, d_model)`` 인코더 출력.
            tgt_mask: causal + 패딩이 결합된 마스크, 또는 None이면 여기서 생성.
            memory_mask: cross-attention을 위한 소스-패딩 마스크.

        Returns:
            ``(batch, tgt_len, d_model)`` 디코더 상태. 어휘집 logits을
            얻으려면 :attr:`generator`를 적용하라.
        """
        if tgt_mask is None:
            tgt_mask = self.make_target_mask(tgt)
        return self.decoder(self.tgt_embedding(tgt), memory, tgt_mask, memory_mask)

    # ---------------------------------------------------------------- forward
    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """학습과 평가를 위한 teacher-forced forward pass.

        마스크 생성부터 어휘집 프로젝션까지 전체 데이터 흐름이 이 함수
        안에 직접 작성되어 있다. (:meth:`encode` / :meth:`decode`는 추론이
        두 절반을 따로 실행하기 위한 공개 API로 유지된다.)

        Args:
            src: ``(batch, src_len)`` 소스 id.
            tgt: ``(batch, tgt_len)`` 디코더 입력 id — 타겟을 오른쪽으로
                한 칸 민 것, 즉 ``[BOS, y1, ..., y_{n-1}]``. 위치 ``t``는
                ``y_t``를 예측하므로, 손실은 출력을 ``[y1, ..., y_n(EOS)]``와
                비교한다 (datasets/collate.py 참고).

        Returns:
            ``(batch, tgt_len, vocab_size)`` 정규화되지 않은 logits.
        """
        # ===================== 1) 마스크 생성 =====================
        # 소스 패딩 마스크: (B, 1, 1, src_len), True = 실제 토큰.
        src_mask = make_pad_mask(src, self.pad_id)
        # 타겟 마스크: 패딩 마스크에 (설정 시) causal 마스크를 AND로 결합
        # -> (B, 1, tgt_len, tgt_len). 미래 위치를 미리 엿보지 못하게 한다.
        tgt_mask = make_pad_mask(tgt, self.pad_id)
        if self.mask_future:
            tgt_mask = combine_masks(tgt_mask, make_causal_mask(tgt.size(1), tgt.device))

        # ===================== 2) 인코더 =====================
        # 소스 임베딩(토큰 + 위치 + dropout) -> 인코더 스택 -> memory.
        src_embedded = self.src_embedding(src)
        memory = self.encoder(src_embedded, src_mask)

        # ===================== 3) 디코더 =====================
        # 타겟 임베딩 -> (마스킹된 self-attn + cross-attn + FFN) 스택.
        # cross-attention은 소스 패딩 마스크를 재사용한다: (B, 1, 1, src_len)이
        # 모든 tgt 쿼리 위치에 걸쳐 브로드캐스트된다.
        tgt_embedded = self.tgt_embedding(tgt)
        decoded = self.decoder(tgt_embedded, memory, tgt_mask, memory_mask=src_mask)

        # ===================== 4) 어휘집 프로젝션 =====================
        # (B, tgt_len, d_model) -> (B, tgt_len, vocab_size)
        return self.generator(decoded)
