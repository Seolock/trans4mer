"""
===============================================================================
 파일: datasets/datamodule.py
 목적:
    프로젝트의 데이터 관련 부분 전체를 소유하는 하나의 객체: 토크나이저,
    train/valid/test 분할에 대한 데이터셋과 DataLoader.

 역할:
    train.py / test.py에서 한 번 호출된다. `setup()`에서 다음을 수행한다:
      1. 토크나이저 JSON을 로드하거나, 파일이 없으면 학습 TSV로부터
         새로 학습한다 (model.vocab_size로 크기 상한).
      2. 분할마다 Seq2SeqDataset을 만든다.
      3. 공유 collator를 사용하는, 바로 순회 가능한 DataLoader들을
         노출한다.
    setup 이후, 호출자는 `tokenizer.vocab_size` / `pad_id`를 model
    config에 복사해서 모델이 항상 실제 어휘집에 맞게 만들어지도록 한다.

 입력 / 출력:
    입력 : 전체 Config (dataset, training, model 섹션 사용).
    출력 : datasets/collate.py에 문서화된 형태의 배치를 만들어내는
           `train_dataloader()`, `val_dataloader()`, `test_dataloader()`.

 구현 세부사항:
    - 학습 분할만 셔플되며, 평가는 결정적으로 유지된다.
    - `drop_last=False`를 모든 곳에서 사용 — gradient 누적이 있으면
      trainer가 마지막에 남는 부분 배치를 올바르게 처리한다.
    - CUDA를 사용할 수 있으면 pin_memory가 자동으로 활성화된다.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config.config import Config
from datasets.collate import Seq2SeqCollator
from datasets.dataset import Seq2SeqDataset
from datasets.tokenizer import Tokenizer
from utils.logger import get_logger

logger = get_logger(__name__)


class Seq2SeqDataModule:
    """모든 분할에 대한 토크나이저, 데이터셋, dataloader를 만들고 소유한다.

    Args:
        config: 전체 프로젝트 설정.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.tokenizer: Optional[Tokenizer] = None
        self.train_dataset: Optional[Seq2SeqDataset] = None
        self.valid_dataset: Optional[Seq2SeqDataset] = None
        self.test_dataset: Optional[Seq2SeqDataset] = None

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        """토크나이저를 로드/학습하고 모든 데이터셋 분할을 실체화한다."""
        self.tokenizer = self._load_or_train_tokenizer()
        d = self.config.dataset
        max_len = self.config.model.max_seq_length
        self.train_dataset = Seq2SeqDataset(d.train_path, self.tokenizer, max_len)
        self.valid_dataset = Seq2SeqDataset(d.valid_path, self.tokenizer, max_len)
        self.test_dataset = Seq2SeqDataset(d.test_path, self.tokenizer, max_len)
        logger.info(
            "Data ready — vocab: %d | train: %d | valid: %d | test: %d examples",
            self.tokenizer.vocab_size,
            len(self.train_dataset),
            len(self.valid_dataset),
            len(self.test_dataset),
        )

    def _load_or_train_tokenizer(self) -> Tokenizer:
        """저장된 토크나이저를 재사용하거나, 학습 코퍼스로부터 새로 학습한다."""
        d = self.config.dataset
        tokenizer_path = Path(d.tokenizer_path)
        if tokenizer_path.exists():
            logger.info("Loading tokenizer from %s", tokenizer_path)
            return Tokenizer.load(tokenizer_path)
        logger.info("Training tokenizer from %s ...", d.train_path)
        tokenizer = Tokenizer.train(
            files=[d.train_path],
            lowercase=d.lowercase,
            min_freq=d.min_freq,
            max_size=self.config.model.vocab_size,
        )
        tokenizer.save(tokenizer_path)
        logger.info("Saved tokenizer (%d tokens) to %s", tokenizer.vocab_size, tokenizer_path)
        return tokenizer

    # ------------------------------------------------------------ dataloader
    def _make_loader(self, dataset: Seq2SeqDataset, shuffle: bool) -> DataLoader:
        """모든 분할에서 공유하는 DataLoader 생성 로직."""
        assert self.tokenizer is not None, "call setup() first"
        t = self.config.training
        return DataLoader(
            dataset,
            batch_size=t.batch_size,
            shuffle=shuffle,
            num_workers=t.num_workers,
            collate_fn=Seq2SeqCollator(self.tokenizer.pad_id),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        """학습 분할에 대한 셔플된 loader."""
        assert self.train_dataset is not None, "call setup() first"
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """검증 분할에 대한 결정적인 loader."""
        assert self.valid_dataset is not None, "call setup() first"
        return self._make_loader(self.valid_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """테스트 분할에 대한 결정적인 loader."""
        assert self.test_dataset is not None, "call setup() first"
        return self._make_loader(self.test_dataset, shuffle=False)
