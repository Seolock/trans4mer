"""
===============================================================================
 파일: vocab.py
 목적:
    BPE 서브워드를 위한 언어별 어휘집(Vocabulary) 클래스.
    토큰 <-> 정수 id 매핑과 텍스트 파일 영속화를 담당한다.

 역할:
    preprocess.py가 train.bpe.{lang}으로부터 언어별로 하나씩 만들어
    vocab.en / vocab.de 로 저장하고, dataset.py / train.py /
    inference/translator.py 가 로드해서 사용한다.

 입력 / 출력:
    encode : list[str] (BPE 토큰) -> list[int]   (미등록 토큰은 <unk>)
    decode : list[int] -> list[str]              (특수 토큰 선택적 생략)
    save/load : 한 줄에 토큰 하나인 UTF-8 텍스트 파일.

 구현 세부사항:
    - 특수 토큰은 낮은 고정 id를 차지한다:
        <pad>=0, <unk>=1, <s>=2(BOS), </s>=3(EOS)
      고정 id 덕분에 손실의 ignore_index, 마스크, 디코딩 루프가 어휘집
      내용과 무관하게 동작한다. 소스/타겟 어휘집이 서로 달라도 pad id는
      항상 0으로 같다.
    - 빈도 내림차순(동점은 토큰 오름차순)으로 정렬해 결정적인 결과를
      보장하며, 크기 상한(max_size)과 최소 빈도(min_freq)를 지원한다.
===============================================================================
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


class Vocab:
    """특수 토큰을 포함한 서브워드 어휘집.

    Args:
        tokens: 맨 앞에 네 개의 특수 토큰을 포함한 전체 정렬 토큰 리스트
            (:meth:`build` 또는 :meth:`load`가 만들어내는 형태).
    """

    def __init__(self, tokens: list[str]) -> None:
        if list(tokens[: len(SPECIAL_TOKENS)]) != list(SPECIAL_TOKENS):
            raise ValueError(f"Vocabulary must start with the special tokens {SPECIAL_TOKENS}")
        self.id_to_token: list[str] = list(tokens)
        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(tokens)}

    # ------------------------------------------------------------ 속성
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
        return len(self.id_to_token)

    # -------------------------------------------------------- 인코딩/디코딩
    def encode(self, tokens: Sequence[str]) -> list[int]:
        """BPE 토큰 시퀀스를 id 시퀀스로 변환한다 (미등록 -> <unk>).

        Args:
            tokens: BPE가 적용된 토큰 리스트.

        Returns:
            정수 id 리스트 (BOS/EOS는 붙이지 않음 — Dataset이 담당).
        """
        return [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> list[str]:
        """id 시퀀스를 BPE 토큰 리스트로 되돌린다.

        Args:
            ids: 정수 id의 이터러블.
            skip_special: pad/unk/bos/eos 토큰을 결과에서 제거할지 여부.

        Returns:
            BPE 토큰 리스트 (" ".join 후 utils.text.remove_bpe로 복원).
        """
        specials = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        return [
            self.id_to_token[i]
            for i in ids
            if 0 <= i < len(self.id_to_token) and not (skip_special and i in specials)
        ]

    # -------------------------------------------------------------- 구축
    @classmethod
    def build(
        cls,
        files: Iterable[str | Path],
        max_size: Optional[int] = None,
        min_freq: int = 1,
    ) -> "Vocab":
        """BPE가 적용된 텍스트 파일(들)로부터 어휘집을 구축한다.

        Args:
            files: 공백 구분 BPE 토큰이 담긴 파일 경로들
                (보통 해당 언어의 train.bpe.{lang} 하나).
            max_size: 어휘집 크기 상한 (특수 토큰 포함).
            min_freq: 이보다 드문 서브워드는 <unk>로 처리된다.

        Returns:
            구축된 :class:`Vocab`.
        """
        counter: Counter[str] = Counter()
        for path in files:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    counter.update(line.split())
        # 결정적인 순서: 빈도 내림차순, 동점은 토큰 오름차순.
        ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        tokens = [tok for tok, freq in ranked if freq >= min_freq]
        if max_size is not None:
            tokens = tokens[: max(0, max_size - len(SPECIAL_TOKENS))]
        return cls(list(SPECIAL_TOKENS) + tokens)

    # ----------------------------------------------------------- 영속화
    def save(self, path: str | Path) -> None:
        """어휘집을 한 줄에 토큰 하나인 텍스트 파일로 저장한다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.id_to_token) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        """:meth:`save`가 저장한 어휘집을 복원한다.

        Raises:
            FileNotFoundError: 어휘집 파일이 없을 때
                (preprocess.py를 먼저 실행해야 함).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {path} — run `python preprocess.py` first."
            )
        with open(path, "r", encoding="utf-8") as fh:
            tokens = [line.rstrip("\n") for line in fh if line.rstrip("\n")]
        return cls(tokens)
