"""
===============================================================================
 파일: dataset.py
 목적:
    Multi30k BPE 파이프라인을 위한 PyTorch Dataset과 DataLoader 구성.

 역할:
    - `TranslationDataset`: preprocess.py가 만든 토큰 id 파일
      ({split}.ids.{src}, {split}.ids.{tgt})을 줄 단위로 짝지어 로드하고,
      타겟에 <s>/</s>를 붙인다. Multimodal(use_image)이면 각 문장에 대응되는
      이미지를 함께 로드한다 — JPEG 즉석 로드(split별 {split}.txt) 또는
      preprocess_images.py가 만든 uint8 캐시(memmap, use_image_cache) 중 하나.
    - `collate_fn`: 배치 패딩 + teacher-forcing shift. 기존
      datasets/collate.py 의 Seq2SeqCollator 를 그대로 재사용한다
      (동일한 배치 형식을 만들기 때문) — 패딩과 shift 규칙은 그 파일에
      상세히 문서화되어 있다. 이미지가 있으면 배치에 stack해 넣어준다.
    - `build_split_dataloader`: 활성 언어쌍의 산출물 디렉터리에서 임의의
      split 이름 하나에 대한 DataLoader를 생성한다 (test.py가 test2016 /
      test2017 / testcoco / test2018을 각각 개별적으로 평가할 때 사용).
    - `build_dataloaders`: train / valid_split / test_split 세 역할의
      DataLoader를 한 번에 생성한다 (내부적으로 build_split_dataloader를
      역할마다 호출). train.py가 학습 루프에 사용한다.

 입력 / 출력:
    __getitem__(i) -> {"src": list[int], "tgt": list[int]}
                      (+ use_image면 "image": (C, H, W) 텐서)  (패딩 없음)
    DataLoader 배치 -> {
        "src"       : (batch, max_src_len)      int64, 오른쪽 패딩
        "tgt_input" : (batch, max_tgt_len - 1)  int64  [<s>, y1, ..., yn]
        "tgt_output": (batch, max_tgt_len - 1)  int64  [y1, ..., yn, </s>]
        "image"     : (batch, C, H, W)          float — use_image일 때만
    }
    -> 기존 Trainer / evaluate() 가 기대하는 형식과 정확히 동일하다
       (이미지 키는 추가될 뿐이며, move_to_device가 자동으로 디바이스로 옮김).

 구현 세부사항:
    - id 파일이 없으면 "python preprocess.py 먼저 실행" 안내와 함께
      FileNotFoundError를 던진다.
    - 소스/타겟 줄 수가 다르면 데이터 정렬이 깨진 것이므로 즉시 에러.
    - 이미지 리스트({split}.txt)는 코퍼스와 줄 단위 1:1 정렬이므로, 빈
      문장 쌍을 skip할 때 해당 이미지도 함께 제외해 정렬을 유지한다.
    - 시퀀스는 model.max_seq_length로 잘리며, 타겟은 </s>가 살아남도록
      중간을 자른다.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config.config import Config
from datasets.collate import Seq2SeqCollator
from utils.data_paths import ids_path, pair_dir
from utils.image import (
    build_image_transform,
    build_normalize_transform,
    load_image_tensor,
    resolve_image_path,
)
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


def _read_image_list(path: str | Path) -> list[str]:
    """split별 이미지 리스트 파일({split}.txt)을 줄 단위로 읽는다.

    Args:
        path: 이미지 파일명 리스트 (한 줄에 하나, "<name>.jpg" 또는
            testcoco처럼 "<name>.jpg#id").

    Returns:
        줄마다 하나씩의 원본 문자열 (경로 해석은 resolve_image_path가 담당).

    Raises:
        FileNotFoundError: 리스트 파일이 없을 때.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Image list not found: {path} — expected data/image/{{split}}.txt for MMT."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh]


class TranslationDataset(Dataset):
    """토큰 id 파일 쌍(+ 선택적 이미지)으로 이루어진 번역 데이터셋.

    소스 문장과 타겟 문장을 줄 단위로 짝지어 로드하고, 타겟에는
    teacher forcing을 위해 [BOS ... EOS]를 붙인다. Multimodal 모드에서는
    split별 이미지 리스트에서 각 문장에 대응되는 이미지를 함께 로드한다.
    패딩과 입력/출력 shift는 collate 단계(Seq2SeqCollator)에서 수행된다.

    Args:
        src_ids_path: 소스 id 파일 (예: data/data-bin/en-de/train.ids.en).
        tgt_ids_path: 타겟 id 파일 (예: data/data-bin/en-de/train.ids.de).
        bos_id: 타겟 문장 시작(<s>) id.
        eos_id: 타겟 문장 종료(</s>) id.
        max_seq_length: 시퀀스 길이 상한 (model.max_seq_length).
        image_list_path: split별 이미지 리스트 경로 (예: data/image/train.txt).
            JPEG 즉석 로드 경로에서 사용. None이면 텍스트-only 또는 캐시 모드.
        image_dir: 이미지 루트 (raw/ 하위에 실제 파일). JPEG 모드에서 필요.
        image_size: 정사각 리사이즈 크기 (multimodal.image_size). 캐시 모드에서
            헤더 검증에 사용.
        image_cache_path: preprocess_images.py가 만든 uint8 캐시 .npy 경로.
            주어지면 JPEG 디코딩 대신 memmap에서 읽는다 (image_list_path와
            둘 중 하나만 주면 됨).
    """

    def __init__(
        self,
        src_ids_path: str | Path,
        tgt_ids_path: str | Path,
        bos_id: int,
        eos_id: int,
        max_seq_length: int,
        image_list_path: Optional[str | Path] = None,
        image_dir: Optional[str | Path] = None,
        image_size: Optional[int] = None,
        image_cache_path: Optional[str | Path] = None,
    ) -> None:
        src_lines = _read_ids_file(src_ids_path)
        tgt_lines = _read_ids_file(tgt_ids_path)
        if len(src_lines) != len(tgt_lines):
            raise ValueError(
                f"Line count mismatch: {src_ids_path} has {len(src_lines)} lines "
                f"but {tgt_ids_path} has {len(tgt_lines)} — corpora are misaligned."
            )

        # ------------------------------------------------------ 이미지 설정
        # 두 경로: (A) JPEG 즉석 로드(image_list_path), (B) uint8 캐시(image_cache_path).
        self.use_image = image_list_path is not None or image_cache_path is not None
        self.use_cache = image_cache_path is not None
        image_names: list[str] = []
        if self.use_cache:
            # 캐시 헤더만 읽어 정렬/해상도를 검증한다 (데이터는 로드하지 않음).
            self.image_cache_path = Path(image_cache_path)
            try:
                header = np.load(self.image_cache_path, mmap_mode="r")
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"Image cache not found: {self.image_cache_path} — run "
                    f"`python preprocess_images.py` (or set multimodal.use_image_cache=false)."
                ) from error
            if header.shape[0] != len(src_lines):
                raise ValueError(
                    f"Image cache/corpus mismatch: {self.image_cache_path} has "
                    f"{header.shape[0]} rows but corpus has {len(src_lines)} — rebuild the cache."
                )
            if header.shape[2] != image_size or header.shape[3] != image_size:
                raise ValueError(
                    f"Image cache size mismatch: {self.image_cache_path} is "
                    f"{header.shape[2]}x{header.shape[3]} but multimodal.image_size={image_size} "
                    f"— rebuild with `python preprocess_images.py`."
                )
            del header  # 검증용 memmap을 닫는다 (실제 로드는 워커별로 지연 오픈).
            self._image_cache: Optional[np.memmap] = None  # 워커마다 개별 지연 오픈
            self.normalize_transform = build_normalize_transform()
        elif self.use_image:
            image_names = _read_image_list(image_list_path)
            if len(image_names) != len(src_lines):
                raise ValueError(
                    f"Image/corpus mismatch: {image_list_path} has {len(image_names)} lines "
                    f"but corpus has {len(src_lines)} — image list must align 1:1 with sentences."
                )
            self.image_dir = image_dir
            # 매 __getitem__마다 새로 만들지 않도록 transform을 한 번만 만든다.
            self.image_transform = build_image_transform(image_size)

        self.examples: list[dict[str, object]] = []
        self.num_skipped = 0
        for line_index, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
            if not src or not tgt:  # 어느 한쪽이 빈 문장이면 건너뛴다
                self.num_skipped += 1
                continue  # 이미지도 함께 건너뛰어 정렬을 유지한다
            # 소스: 길이 상한으로 자르기.
            src = src[:max_seq_length]
            # 타겟: [BOS ... EOS]를 붙이고, 넘치면 EOS가 살아남도록 자른다.
            tgt = [bos_id] + tgt + [eos_id]
            if len(tgt) > max_seq_length:
                tgt = tgt[: max_seq_length - 1] + [eos_id]
            example: dict[str, object] = {"src": src, "tgt": tgt}
            # 원본 줄 인덱스로 이미지를 짝짓는다 (skip된 줄은 이미지도 제외됨).
            if self.use_cache:
                example["image_row"] = line_index  # 캐시 memmap의 row 인덱스
            elif self.use_image:
                example["image_path"] = resolve_image_path(self.image_dir, image_names[line_index])
            self.examples.append(example)

        if not self.examples:
            raise ValueError(f"No usable sentence pairs in {src_ids_path} / {tgt_ids_path}")
        if self.num_skipped:
            logger.info("Skipped %d empty pairs in %s", self.num_skipped, Path(src_ids_path).name)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        """(패딩 없는) 한 쌍의 id 시퀀스를 반환한다 (use_image면 이미지 텐서 포함)."""
        example = self.examples[index]
        if not self.use_image:
            return example
        if self.use_cache:
            # memmap을 워커 프로세스마다 지연 오픈한다 (OS 페이지 캐시는 공유됨).
            if self._image_cache is None:
                self._image_cache = np.load(self.image_cache_path, mmap_mode="r")
            # 해당 row(uint8 (C,S,S))만 mmap에서 복사해 오고 로드 시점에 정규화한다.
            row = np.asarray(self._image_cache[example["image_row"]])
            image = self.normalize_transform(torch.from_numpy(row.copy()))
        else:
            # 이미지를 지연 로드한다 (전체를 메모리에 올리지 않음): (C, H, W) 텐서.
            image = load_image_tensor(example["image_path"], transform=self.image_transform)
        return {"src": example["src"], "tgt": example["tgt"], "image": image}


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
    d, t, mm = config.dataset, config.training, config.multimodal
    base_dir = pair_dir(d.bin_dir, d.src_lang, d.tgt_lang)
    # Multimodal 이미지 소스 선택:
    #   use_image_cache -> preprocess_images.py가 만든 uint8 캐시(memmap)에서 로드,
    #   그 외 use_image  -> split 이름과 동일한 이미지 리스트({split}.txt)로 JPEG 즉석 로드.
    image_list_path = None
    image_cache_path = None
    if mm.use_image:
        if mm.use_image_cache:
            image_cache_path = Path(mm.image_cache_dir) / f"{split}_{mm.image_size}.npy"
        else:
            image_list_path = Path(mm.image_dir) / f"{split}.txt"
    dataset = TranslationDataset(
        src_ids_path=ids_path(base_dir, split, d.src_lang),
        tgt_ids_path=ids_path(base_dir, split, d.tgt_lang),
        bos_id=bos_id,
        eos_id=eos_id,
        max_seq_length=config.model.max_seq_length,
        image_list_path=image_list_path,
        image_dir=mm.image_dir if mm.use_image else None,
        image_size=mm.image_size if mm.use_image else None,
        image_cache_path=image_cache_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=t.batch_size,
        shuffle=shuffle,
        num_workers=t.num_workers,
        collate_fn=collate_fn(pad_id),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        # 워커를 epoch 간 유지해 재시작 비용과 (캐시 모드의) memmap 재오픈을 줄인다.
        persistent_workers=t.num_workers > 0,
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
