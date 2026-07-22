"""
===============================================================================
 파일: dataset.py
 목적:
    Multi30k BPE 파이프라인을 위한 PyTorch Dataset과 DataLoader 구성.

 역할:
    - `TranslationDataset`: preprocess.py가 만든 토큰 id 파일
      ({split}.ids.{src}, {split}.ids.{tgt})을 줄 단위로 짝지어 로드하고,
      타겟에 <s>/</s>를 붙인다.
    - `collate_fn`: 배치 패딩 + teacher-forcing shift. 기존
      datasets/collate.py 의 Seq2SeqCollator 를 그대로 재사용한다
      (동일한 배치 형식을 만들기 때문) — 패딩과 shift 규칙은 그 파일에
      상세히 문서화되어 있다.
    - `build_split_dataloader`: 활성 언어쌍의 산출물 디렉터리에서 임의의
      split 이름 하나에 대한 DataLoader를 생성한다 (test.py가 test2016 /
      test2017 / testcoco / test2018을 각각 개별적으로 평가할 때 사용).
    - `build_dataloaders`: train / valid_split / test_split 세 역할의
      DataLoader를 한 번에 생성한다 (내부적으로 build_split_dataloader를
      역할마다 호출). train.py가 학습 루프에 사용한다.

 입력 / 출력:
    __getitem__(i) -> {"src": list[int], "tgt": list[int]}   (패딩 없음)
    DataLoader 배치 -> {
        "src"       : (batch, max_src_len)      int64, 오른쪽 패딩
        "tgt_input" : (batch, max_tgt_len - 1)  int64  [<s>, y1, ..., yn]
        "tgt_output": (batch, max_tgt_len - 1)  int64  [y1, ..., yn, </s>]
    }
    -> 기존 Trainer / evaluate() 가 기대하는 형식과 정확히 동일하다.

 구현 세부사항:
    - id 파일이 없으면 "python preprocess.py 먼저 실행" 안내와 함께
      FileNotFoundError를 던진다.
    - 소스/타겟 줄 수가 다르면 데이터 정렬이 깨진 것이므로 즉시 에러.
    - 시퀀스는 model.max_seq_length로 잘리며, 타겟은 </s>가 살아남도록
      중간을 자른다.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from config.config import Config
from datasets.collate import Seq2SeqCollator
from utils.data_paths import ids_path, pair_dir
from utils.logger import get_logger

logger = get_logger(__name__)


def _read_ids_file(path: str | Path) -> list[list[int]]:
    """공백 구분 정수 id 파일을 문장별 리스트로 읽는다.

    Args:
        path: preprocess.py가 만든 {split}.ids.{lang} 파일.

    Returns:
        문장마다 하나씩의 id 리스트 (빈 줄은 빈 리스트로 보존해
        소스/타겟 줄 정렬을 유지).

    Raises:
        FileNotFoundError: 파일이 없을 때 (전처리 미실행).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Token-id file not found: {path} — run `python preprocess.py` first."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return [[int(tok) for tok in line.split()] for line in fh]


class TranslationDataset(Dataset):
    """토큰 id 파일 쌍으로 이루어진 번역 데이터셋.

    소스 문장과 타겟 문장을 줄 단위로 짝지어 로드하고, 타겟에는
    teacher forcing을 위해 [BOS ... EOS]를 붙인다. 패딩과 입력/출력
    shift는 collate 단계(Seq2SeqCollator)에서 수행된다.

    Args:
        src_ids_path: 소스 id 파일 (예: data/multi30k/train.ids.en).
        tgt_ids_path: 타겟 id 파일 (예: data/multi30k/train.ids.de).
        bos_id: 타겟 문장 시작(<s>) id.
        eos_id: 타겟 문장 종료(</s>) id.
        max_seq_length: 시퀀스 길이 상한 (model.max_seq_length).
    """

    def __init__(
        self,
        src_ids_path: str | Path,
        tgt_ids_path: str | Path,
        bos_id: int,
        eos_id: int,
        max_seq_length: int,
    ) -> None:
        src_lines = _read_ids_file(src_ids_path)
        tgt_lines = _read_ids_file(tgt_ids_path)
        if len(src_lines) != len(tgt_lines):
            raise ValueError(
                f"Line count mismatch: {src_ids_path} has {len(src_lines)} lines "
                f"but {tgt_ids_path} has {len(tgt_lines)} — corpora are misaligned."
            )

        self.examples: list[dict[str, list[int]]] = []
        self.num_skipped = 0
        for src, tgt in zip(src_lines, tgt_lines):
            if not src or not tgt:  # 어느 한쪽이 빈 문장이면 건너뛴다
                self.num_skipped += 1
                continue
            # 소스: 길이 상한으로 자르기.
            src = src[:max_seq_length]
            # 타겟: [BOS ... EOS]를 붙이고, 넘치면 EOS가 살아남도록 자른다.
            tgt = [bos_id] + tgt + [eos_id]
            if len(tgt) > max_seq_length:
                tgt = tgt[: max_seq_length - 1] + [eos_id]
            self.examples.append({"src": src, "tgt": tgt})

        if not self.examples:
            raise ValueError(f"No usable sentence pairs in {src_ids_path} / {tgt_ids_path}")
        if self.num_skipped:
            logger.info("Skipped %d empty pairs in %s", self.num_skipped, Path(src_ids_path).name)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """(패딩 없는) 한 쌍의 id 시퀀스를 반환한다."""
        return self.examples[index]


# 배치 패딩 + teacher-forcing shift. 기존 구현을 그대로 재사용한다
# (자세한 규칙은 datasets/collate.py의 파일 헤더 참고).
collate_fn = Seq2SeqCollator


def build_split_dataloader(
    config: Config,
    split: str,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    shuffle: bool = False,
) -> DataLoader:
    """활성 언어쌍의 임의의 split 하나에 대한 DataLoader를 생성한다.

    id 파일은 활성 쌍의 산출물 디렉터리(bin_dir/{src}-{tgt}/)에서 읽는다.
    `split`은 preprocess.py가 만든 어떤 이름이든 될 수 있다 (train, val,
    test2016, test2017, testcoco, test2018).

    Args:
        config: 전체 설정 (dataset.bin_dir / src_lang / tgt_lang,
            training.batch_size / num_workers, model.max_seq_length 사용).
        split: 로드할 split 이름 (예: "test2017").
        pad_id: 패딩 id (양쪽 어휘집 모두 0).
        bos_id: 타겟 BOS id.
        eos_id: 타겟 EOS id.
        shuffle: 배치 순서를 섞을지 여부 (학습용 train만 True로 준다).

    Returns:
        해당 split의 DataLoader (평가 용도이므로 기본은 셔플 없음,
        결정적 순서 유지).
    """
    d, t = config.dataset, config.training
    base_dir = pair_dir(d.bin_dir, d.src_lang, d.tgt_lang)
    dataset = TranslationDataset(
        src_ids_path=ids_path(base_dir, split, d.src_lang),
        tgt_ids_path=ids_path(base_dir, split, d.tgt_lang),
        bos_id=bos_id,
        eos_id=eos_id,
        max_seq_length=config.model.max_seq_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=t.batch_size,
        shuffle=shuffle,
        num_workers=t.num_workers,
        collate_fn=collate_fn(pad_id),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    logger.info("%s (%s-%s): %d pairs", split, d.src_lang, d.tgt_lang, len(dataset))
    return loader


def build_dataloaders(
    config: Config, pad_id: int, bos_id: int, eos_id: int
) -> dict[str, DataLoader]:
    """활성 언어쌍의 train/valid/test DataLoader를 한 번에 생성한다.

    어떤 split을 검증/테스트에 쓸지는 dataset.valid_split /
    dataset.test_split이 결정한다 (예: test_split=test2017). 내부적으로
    :func:`build_split_dataloader`를 역할마다 호출한다.

    Args:
        config: 전체 설정 (dataset.valid_split / test_split 포함;
            나머지는 build_split_dataloader와 동일).
        pad_id: 패딩 id (양쪽 어휘집 모두 0).
        bos_id: 타겟 BOS id.
        eos_id: 타겟 EOS id.

    Returns:
        {"train": ..., "valid": ..., "test": ...} DataLoader 매핑
        (키는 역할 이름 — Trainer/evaluate와의 계약을 유지한다).
        train만 셔플되며 평가는 결정적으로 유지된다.
    """
    d = config.dataset
    # 역할 이름 -> 실제 split 이름 매핑 (검증/테스트 split은 설정으로 선택).
    role_to_split = {"train": "train", "valid": d.valid_split, "test": d.test_split}
    return {
        role: build_split_dataloader(config, split, pad_id, bos_id, eos_id, shuffle=(role == "train"))
        for role, split in role_to_split.items()
    }
