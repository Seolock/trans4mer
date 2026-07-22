"""
===============================================================================
 파일: datasets/dataset.py
 목적:
    TSV("소스<TAB>타겟", 한 줄에 한 쌍)로 저장된 sequence-to-sequence
    쌍을 위한 map-style PyTorch Dataset.

 역할:
    생성 시점에 코퍼스 전체를 한 번 읽고 토큰화한다 (메모리에 다 들어가는
    코퍼스에는 적합; 아주 큰 데이터셋은 동일한 인터페이스 뒤에서 지연
    읽기(lazy reading)로 바꿀 수 있음). 각 아이템은 패딩되지 않은 id
    시퀀스 쌍이며, 패딩과 teacher-forcing shift는 나중에
    datasets/collate.py에서 처리된다.

 입력 / 출력:
    __getitem__(i) -> {
        "src": list[int]  소스 id, max_seq_length로 잘림,
        "tgt": list[int]  [BOS ... EOS]가 포함된 타겟 id, shift된 디코더
                          입력이 여전히 max_seq_length에 맞도록 잘림,
    }

 구현 세부사항:
    - 형식이 잘못된 줄(탭 없음 / 한쪽이 빈 경우)은 건너뛰며 개수를
      `num_skipped`에 기록한다, 그래서 지저분한 데이터가 학습을 절대
      중단시키지 않는다.
    - truncation은 타겟 쪽에서 EOS가 살아남도록 보장한다: 종결자가 아니라
      중간 내용을 자른다.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from datasets.tokenizer import Tokenizer


class Seq2SeqDataset(Dataset):
    """(소스, 타겟) id 시퀀스로 이루어진 TSV 기반 병렬 코퍼스.

    Args:
        path: 한 줄에 한 쌍의 "소스<TAB>타겟"이 담긴 TSV 파일.
        tokenizer: 두 컬럼 모두에 사용되는 공유 :class:`Tokenizer`.
        max_seq_length: 시퀀스 길이의 상한 (모델 제약).
    """

    def __init__(self, path: str | Path, tokenizer: Tokenizer, max_seq_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples: list[dict[str, list[int]]] = []
        self.num_skipped = 0

        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if "\t" not in line:
                    self.num_skipped += 1
                    continue
                source_text, target_text = line.split("\t", 1)
                example = self._build_example(source_text, target_text)
                if example is None:
                    self.num_skipped += 1
                else:
                    self.examples.append(example)

        if not self.examples:
            raise ValueError(f"No usable examples found in {path}")

    def _build_example(self, source_text: str, target_text: str) -> dict[str, list[int]] | None:
        """한 쌍을 토큰화한다; 어느 한쪽이 비어 있으면 None을 반환한다."""
        src = self.tokenizer.encode(source_text)[: self.max_seq_length]
        # 타겟은 teacher forcing을 위해 BOS/EOS를 포함한다. shift된
        # (길이 - 1)짜리 디코더 입력도 max_seq_length에 맞도록 여유를 둔다.
        tgt = self.tokenizer.encode(target_text, add_bos=True, add_eos=True)
        if len(tgt) > self.max_seq_length:
            tgt = tgt[: self.max_seq_length - 1] + [self.tokenizer.eos_id]
        if not src or len(tgt) <= 2:  # BOS/EOS만 있다는 것은 타겟이 비었다는 뜻
            return None
        return {"src": src, "tgt": tgt}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """한 쌍의 (패딩되지 않은) id 시퀀스를 반환한다."""
        return self.examples[index]
