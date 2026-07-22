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
    - BLEU 지원: 이 모듈은 teacher-forced 품질만 측정한다. 생성 품질을
      보려면 inference/predict.py나 inference/beam_search.py로 디코딩하고
      trainer.metrics.corpus_bleu로 점수를 매겨라 — 두 부분은 이미
      서로 호환된다 (README의 "Evaluation" 참고).
===============================================================================
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainer.metrics import perplexity, sequence_correct, token_correct
from utils.misc import move_to_device


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
