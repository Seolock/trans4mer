"""
===============================================================================
 파일: datasets/tokenizer.py
 목적:
    특수 토큰을 포함하고, 원본 텍스트로부터 학습 가능하며 JSON으로
    직렬화 가능한, 외부 의존성 없는 단어 단위(word-level) 토크나이저.

 역할:
    소스와 타겟 양쪽 모두를 위해 텍스트 <-> 정수 id 시퀀스를 변환한다
    (하나의 공유된 어휘집을 사용하며, 이 덕분에 `share_embedding` /
    weight tying이 유효해진다). 학습 스크립트가 최초 실행 시 학습 TSV로부터
    토크나이저를 만들고, 이후에는 저장된 JSON을 재사용한다.

 입력 / 출력:
    encode : str -> list[int]     (선택적으로 BOS/EOS로 감쌈)
    decode : list[int] -> str     (선택적으로 특수 토큰 생략)

 구현 세부사항:
    - 특수 토큰은 낮은 고정 id를 차지한다: <pad>=0, <unk>=1, <bos>=2,
      <eos>=3. 고정된 id 덕분에 나머지 코드베이스(마스크, 손실
      ignore_index, 디코딩 루프)가 별도의 간접 참조 없이 이 값들에
      의존할 수 있다.
    - 어휘집은 빈도순으로 정렬되며, 크기 상한(`max_size`)과 빈도 필터링
      (`min_freq`)을 걸 수 있다 — 둘 다 설정으로 제어된다.
    - 공백 기준의 단어 단위 분리는 교육적 목적으로 클래스를 단순하게
      유지한다; 서브워드 토크나이저(BPE 등)로 동일한 인터페이스 뒤에서
      교체할 수 있다.
===============================================================================
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


class Tokenizer:
    """특수 토큰 처리와 JSON 영속화를 지원하는 단어 단위 토크나이저.

    Args:
        tokens: 맨 앞에 네 개의 특수 토큰을 포함한 전체 정렬 어휘집
            (:meth:`train` 또는 :meth:`load`가 만들어내는 형태).
        lowercase: 분리하기 전 텍스트를 소문자로 바꿀지 여부.
    """

    def __init__(self, tokens: list[str], lowercase: bool = True) -> None:
        if list(tokens[: len(SPECIAL_TOKENS)]) != list(SPECIAL_TOKENS):
            raise ValueError(f"Vocabulary must start with the special tokens {SPECIAL_TOKENS}")
        self.lowercase = lowercase
        self.id_to_token: list[str] = list(tokens)
        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}

    # ------------------------------------------------------------ 속성
    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    def __len__(self) -> int:
        return self.vocab_size

    # -------------------------------------------------------------- 인코딩
    def tokenize(self, text: str) -> list[str]:
        """원본 텍스트를 표층 토큰으로 분리한다 (공백 기준 단어 단위)."""
        if self.lowercase:
            text = text.lower()
        return text.strip().split()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """텍스트를 토큰 id로 변환하며, 모르는 단어는 <unk>로 매핑한다.

        Args:
            text: 원본 입력 문자열.
            add_bos: 시퀀스 시작(BOS) id를 앞에 붙일지 여부.
            add_eos: 시퀀스 종료(EOS) id를 뒤에 붙일지 여부.

        Returns:
            정수 토큰 id 리스트.
        """
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in self.tokenize(text)]
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """토큰 id를 공백으로 이어 붙인 문자열로 되돌린다.

        Args:
            ids: 정수 id의 이터러블.
            skip_special: 출력에서 pad/unk/bos/eos 토큰을 제거할지 여부.

        Returns:
            복원된 텍스트.
        """
        specials = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        words = [
            self.id_to_token[i]
            for i in ids
            if 0 <= i < self.vocab_size and not (skip_special and i in specials)
        ]
        return " ".join(words)

    # -------------------------------------------------------------- 학습
    @classmethod
    def train(
        cls,
        files: Iterable[str | Path],
        lowercase: bool = True,
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ) -> "Tokenizer":
        """TSV 파일들("소스<TAB>타겟"이 한 줄씩)로부터 어휘집을 만든다.

        두 컬럼 모두 하나의 공유 어휘집에 기여한다. 토큰은 빈도순으로
        정렬되며(결정적 결과를 위해 동점은 알파벳 순으로 정렬).

        Args:
            files: 스캔할 TSV 파일 경로들.
            lowercase: 카운트하기 전 텍스트를 소문자로 바꿀지 여부.
            min_freq: 이보다 드문 토큰은 버린다.
            max_size: 전체 어휘집 크기 상한 (특수 토큰 포함).

        Returns:
            학습된 :class:`Tokenizer`.
        """
        counter: Counter[str] = Counter()
        for path in files:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    for column in line.rstrip("\n").split("\t"):
                        text = column.lower() if lowercase else column
                        counter.update(text.strip().split())
        # 결정적인 순서: 빈도 내림차순, 그다음 토큰 오름차순.
        ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        tokens = [tok for tok, freq in ranked if freq >= min_freq]
        if max_size is not None:
            tokens = tokens[: max(0, max_size - len(SPECIAL_TOKENS))]
        return cls(list(SPECIAL_TOKENS) + tokens, lowercase=lowercase)

    # ----------------------------------------------------------- 영속화
    def save(self, path: str | Path) -> None:
        """어휘집을 JSON 파일로 직렬화한다."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"lowercase": self.lowercase, "tokens": self.id_to_token}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        """:meth:`save`가 저장한 토크나이저를 복원한다."""
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(payload["tokens"], lowercase=payload["lowercase"])
