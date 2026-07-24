"""
===============================================================================
 파일: trainer/evaluator.py
 목적:
    dataloader에 대한 teacher-forced 평가: loss, perplexity, 토큰 정확도,
    (시퀀스) 정확도.

 역할:
    Trainer가 epoch 사이 검증에, test.py가 최종 test-set 리포트에
    사용한다. 인자를 미리 바인딩한 순수 `evaluate()` 함수와 `Evaluator`
    클래스 둘 다로 노출된다.

 입력 / 출력:
    입력 : model, dataloader (datasets/collate.py에서 온 배치), device,
           pad id.
    출력 : {"loss", "perplexity", "token_accuracy", "accuracy"} 실수값들.

 구현 세부사항:
    - loss는 정확한 corpus-level 토큰당 cross-entropy를 얻기 위해
      패딩이 아닌 타겟 토큰의 정확한 개수로 나눈 sum-reduction을
      사용한다 — 배치 크기가 가변적일 때 배치 평균 방식은 값을
      왜곡시킨다.
    - (학습에서 label smoothing을 쓰더라도) 여기서는 사용하지 않는다:
      그래야 리포트된 loss와 perplexity가 smoothing 설정과 무관하게
      비교 가능하다.
    - `evaluate`는 teacher-forced 지표만, `evaluate_bleu`는 생성 기반
      BLEU를 계산한다. fairseq의 기본 검증(teacher-forced loss)에
      `--eval-bleu`(생성 BLEU)를 얹는 구조와 동일하게, Trainer는 매 검증마다
      두 함수를 모두 호출한다 (evaluate_bleu는 greedy 생성이라 더 비쌈).
===============================================================================
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference.decode import greedy_decode
from trainer.metrics import perplexity, sequence_correct, token_correct
from utils.misc import move_to_device
from utils.text import remove_bpe
from vocab import Vocab


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    pad_id: int,
    progress: bool = False,
) -> dict[str, float]:
    """teacher-forced 평가를 실행하고 corpus-level 지표를 집계한다.

    Args:
        model: Transformer (또는 forward(src, tgt) -> logits를 가진 모델).
        dataloader: {"src", "tgt_input", "tgt_output"}을 만들어내는 loader.
        device: 실행할 디바이스.
        pad_id: loss와 모든 지표에서 제외할 패딩 id.
        progress: tqdm 진행바를 표시할지 여부 (test.py에서 사용; 학습 바를
            깔끔하게 유지하기 위해 검증 중에는 꺼둠).

    Returns:
        ``loss``(토큰당 nats), ``perplexity``, ``token_accuracy``,
        ``accuracy``(정확한 시퀀스 일치율) 키를 가진 dict.
    """
    was_training = model.training
    model.eval()

    # sum-reduction + 수동 정규화 = 정확한 토큰당 평균.
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    total_sequences = 0

    iterator = tqdm(dataloader, desc="evaluate", leave=False) if progress else dataloader
    for batch in iterator:
        batch = move_to_device(batch, device)
        # batch.get("image"): Multimodal이면 (B, C, H, W), 텍스트-only면 None.
        logits = model(batch["src"], batch.get("image"), batch["tgt_input"])
        targets = batch["tgt_output"]

        # (batch, seq, vocab) -> (batch*seq, vocab)으로 펼쳐서 loss를 계산.
        total_loss += criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()

        predictions = logits.argmax(dim=-1)
        tok_correct, tok_total = token_correct(predictions, targets, pad_id)
        seq_correct, seq_total = sequence_correct(predictions, targets, pad_id)
        correct_tokens += tok_correct
        total_tokens += tok_total
        correct_sequences += seq_correct
        total_sequences += seq_total

    if was_training:  # 호출자의 모드를 복원한다
        model.train()

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": perplexity(mean_loss),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "accuracy": correct_sequences / max(total_sequences, 1),
    }


@torch.no_grad()
def evaluate_bleu(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    tgt_vocab: Vocab,
    max_length: int,
    min_length: int = 1,
    progress: bool = False,
) -> float:
    """생성 기반 valid BLEU를 계산한다 (greedy 디코딩 + sacreBLEU).

    가설은 greedy로 생성하고, 참조는 배치의 ``tgt_output`` id를 그대로
    detok한다. hyp/ref 모두 동일한 BPE-detok(소문자) 공간에서 비교되는
    self-contained "내부 valid BLEU"다. test.py가 원본 참조 파일로 계산하는
    최종 sacreBLEU와 수치가 약간 다를 수 있으나, 학습 진행을 추적하는
    일관된 신호다.

    Args:
        model: Transformer (encode/decode/generator를 가진 모델).
        dataloader: {"src", "tgt_output", 선택적 "image"}를 주는 loader.
        device: 실행할 디바이스.
        tgt_vocab: 타겟 어휘집 (id -> 서브워드 토큰 복원에 사용).
        max_length: greedy 생성 최대 길이.
        min_length: 이 길이 이전에는 EOS를 금지한다.
        progress: tqdm 진행바 표시 여부.

    Returns:
        corpus BLEU 점수 (0~100).
    """
    import sacrebleu  # 지연 import (test.py와 동일 라이브러리)

    was_training = model.training
    model.eval()

    hypotheses: list[str] = []
    references: list[str] = []
    iterator = tqdm(dataloader, desc="valid-bleu", leave=False) if progress else dataloader
    for batch in iterator:
        batch = move_to_device(batch, device)
        hyp_ids = greedy_decode(
            model, batch["src"],
            bos_id=tgt_vocab.bos_id, eos_id=tgt_vocab.eos_id, pad_id=tgt_vocab.pad_id,
            max_length=max_length, min_length=min_length, image=batch.get("image"),
        )
        for ids in hyp_ids:
            hypotheses.append(remove_bpe(" ".join(tgt_vocab.decode(ids))))
        # 참조: tgt_output 각 행을 detok (decode가 pad/eos/bos/unk를 자동 제거).
        for ref_row in batch["tgt_output"].tolist():
            references.append(remove_bpe(" ".join(tgt_vocab.decode(ref_row))))

    if was_training:  # 호출자의 모드를 복원한다
        model.train()

    # 표준 sacreBLEU(기본 13a 토크나이저). hyp/ref 모두 remove_bpe(decode())
    # 로 동일 공간에 있고 소문자이므로 lowercase=True(uncased)로 맞춘다.
    return sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True).score


class Evaluator:
    """반복 사용을 위해 model/device/pad_id를 미리 바인딩하는 편의 래퍼.

    Args:
        model: 평가할 모델.
        device: 실행할 디바이스.
        pad_id: 패딩 id.
    """

    def __init__(self, model: nn.Module, device: torch.device, pad_id: int) -> None:
        self.model = model
        self.device = device
        self.pad_id = pad_id

    def evaluate(self, dataloader: DataLoader, progress: bool = False) -> dict[str, float]:
        """``dataloader``에 대해 평가한다; 모듈 레벨 :func:`evaluate` 참고."""
        return evaluate(self.model, dataloader, self.device, self.pad_id, progress=progress)
