"""
===============================================================================
 파일: trainer/trainer.py
 목적:
    전체 학습 루프: epoch, gradient 누적, 혼합 정밀도, gradient 클리핑,
    LR 스케줄링, 검증, early stopping, 체크포인팅(last / best / periodic),
    재개, TensorBoard 로깅.

 역할:
    모든 학습 관련 컴포넌트를 조율하지만 과학적 내용은 소유하지 않는다:
    모델은 models/에서, 지표는 trainer/metrics.py에서, LR 스케줄은
    trainer/scheduler.py에서, 영속화는 trainer/checkpoint.py에서, 평가는
    trainer/evaluator.py에서 온다.

 입력 / 출력:
    입력 : Config, Transformer, train/valid DataLoader, device.
    출력 : `checkpoint.save_dir` 아래에 체크포인트 + TensorBoard 로그;
           `fit()`은 관찰된 최고 검증 지표를 반환한다.

 구현 세부사항:
    - Loss: ignore_index=pad와 설정 가능한 label smoothing을 쓰는
      CrossEntropy; epoch별 평균은 패딩이 아닌 토큰 개수로 가중치가
      매겨진다.
    - Gradient 누적: 각 micro-batch의 loss를 accumulation_steps로 나눈다;
      옵티마이저는 누적 경계에서 그리고 epoch의 마지막(부분적일 수 있는)
      배치에서 스텝된다.
    - AMP: torch.amp autocast + GradScaler, CUDA에서만 활성화된다.
      클리핑 임계값이 실제(진짜) 단위가 되도록 클리핑 전에 gradient의
      스케일을 되돌린다(unscale).
    - 스케줄러는 매 OPTIMIZER 스텝마다 한 번씩 스텝된다; total_steps는
      loader 길이, 누적, epoch로부터 계산된다.
    - Early stopping은 (epoch가 아니라) 연속된 검증 횟수 중 개선이 없는
      경우를 센다.
===============================================================================
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import torch
from torch import amp, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.config import Config, OptimizationConfig
from trainer.checkpoint import CheckpointManager
from trainer.evaluator import evaluate, evaluate_bleu
from trainer.scheduler import build_scheduler
from utils.logger import get_logger
from utils.misc import AverageMeter, move_to_device
from vocab import Vocab


def build_optimizer(model: nn.Module, config: OptimizationConfig) -> torch.optim.Optimizer:
    """모델 파라미터에 대해 설정된 옵티마이저를 구성한다.

    Args:
        model: 최적화할 모델.
        config: `optimization` 설정 섹션.

    Returns:
        Adam / AdamW / SGD 옵티마이저 (SGD는 betas[0]을 momentum으로 재사용).
    """
    params = model.parameters()
    betas = tuple(config.betas)
    if config.optimizer == "adam":
        return torch.optim.Adam(
            params, lr=config.learning_rate, betas=betas, eps=config.eps,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            params, lr=config.learning_rate, betas=betas, eps=config.eps,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=config.learning_rate, momentum=betas[0],
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer '{config.optimizer}'")


class Trainer:
    """Transformer를 위한 종단간(end-to-end) 학습 오케스트레이터.

    Args:
        config: 전체 프로젝트 설정.
        model: 학습할 모델 (이미 ``device``에 올라가 있음).
        train_loader: 학습 분할에 대한 loader.
        valid_loader: 검증 분할에 대한 loader.
        device: 학습할 디바이스.
        tgt_vocab: 타겟 어휘집 (검증 중 생성 기반 valid BLEU 계산에 사용;
            None이거나 training.valid_bleu가 false면 BLEU를 건너뛴다).
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        device: torch.device,
        tgt_vocab: Optional[Vocab] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.pad_id = config.model.pad_token_id

        t, o, c = config.training, config.optimization, config.checkpoint
        self.accumulation_steps = max(1, t.accumulation_steps)
        self.logger = get_logger("trainer", log_file=Path(c.save_dir) / "train.log")

        # ------------------------------------------------ valid BLEU (생성 기반)
        # tgt_vocab이 있고 training.valid_bleu가 켜졌을 때만 매 검증마다 greedy
        # 생성으로 BLEU를 계산한다. 생성 길이는 translator와 동일 규칙.
        self.tgt_vocab = tgt_vocab
        self.valid_bleu = t.valid_bleu and tgt_vocab is not None
        self.gen_max_length = min(config.inference.max_length, config.model.max_seq_length - 1)
        if t.valid_bleu and tgt_vocab is None:
            self.logger.warning("training.valid_bleu=true 이지만 tgt_vocab이 없어 BLEU를 건너뜁니다")

        # ------------------------------------------------ loss & 최적화
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id, label_smoothing=o.label_smoothing
        )
        self.optimizer = build_optimizer(model, o)
        steps_per_epoch = math.ceil(len(train_loader) / self.accumulation_steps)
        self.scheduler = build_scheduler(
            self.optimizer, o, total_steps=steps_per_epoch * t.epochs
        )

        # ------------------------------------------------- 혼합 정밀도
        # AMP는 CUDA에서만 의미가 있고(GradScaler도 CUDA에서만 지원됨).
        self.amp_enabled = t.mixed_precision and device.type == "cuda"
        if t.mixed_precision and not self.amp_enabled:
            self.logger.warning("mixed_precision requested but no CUDA device — running FP32")
        self.scaler = amp.GradScaler("cuda", enabled=self.amp_enabled)

        # -------------------------------------------- 로깅 & 체크포인트
        self.writer = SummaryWriter(log_dir=str(Path(c.save_dir) / "tensorboard"))
        self.checkpoints = CheckpointManager(c.save_dir, metric_name=c.best_metric)

        # ------------------------------------------------------ 루프 상태
        self.global_step = 0        # 지금까지 진행된 옵티마이저 스텝 수
        self.start_epoch = 0
        self.validations_without_improvement = 0
        self.last_val_metrics: Optional[dict[str, float]] = None

        if c.resume_checkpoint:
            self._resume(c.resume_checkpoint)

    # ====================================================================== #
    #  공개 API                                                              #
    # ====================================================================== #
    def fit(self) -> dict[str, float]:
        """전체 학습 스케줄을 실행한다.

        Returns:
            관찰된 최고 검증 지표 (검증이 한 번도 실행되지 않았으면
            빈 dict, 예를 들어 epochs == 0인 경우).
        """
        t = self.config.training
        best_metrics: dict[str, float] = {}
        self.logger.info(
            "Starting training: %d epochs, %d batches/epoch, accumulation=%d, amp=%s",
            t.epochs, len(self.train_loader), self.accumulation_steps, self.amp_enabled,
        )

        for epoch in range(self.start_epoch, t.epochs):
            train_loss = self._train_epoch(epoch)
            self.writer.add_scalar("train/epoch_loss", train_loss, epoch + 1)

            improved = False
            if (epoch + 1) % t.validate_every == 0:
                metrics = self._validate(epoch)
                improved = self.checkpoints.update_best(metrics[self.checkpoints.metric_name])
                if improved:
                    self.validations_without_improvement = 0
                    best_metrics = metrics
                else:
                    self.validations_without_improvement += 1

            # 상태 저장: 롤링되는 last, 개선 시 best, 주기적인 사본.
            state = self._state_dict(epoch)
            self.checkpoints.save_last(state)
            if improved:
                self.checkpoints.save_best(state)
            if (epoch + 1) % t.save_every == 0:
                self.checkpoints.save_epoch(state, epoch + 1)

            if self._should_stop_early():
                self.logger.info(
                    "Early stopping at epoch %d (%d validations without improvement)",
                    epoch + 1, self.validations_without_improvement,
                )
                break

        # 학습 종료 마무리: 먼저 최근 epoch 체크포인트로 앙상블을 만들고
        # (이 시점엔 epoch_*.pt가 아직 존재), 그다음 epoch 파일을 정리한다.
        # 순서가 뒤바뀌면 앙상블 재료가 사라지므로 반드시 앙상블 -> 정리.
        c = self.config.checkpoint
        if c.save_ensemble:
            self.checkpoints.save_ensemble(c.ensemble_last_n)
        if c.cleanup_epoch_checkpoints:
            self.checkpoints.cleanup_epoch_checkpoints()

        self.writer.close()
        if best_metrics:
            self.logger.info("Best validation metrics: %s", _format_metrics(best_metrics))
        return best_metrics

    # ====================================================================== #
    #  학습 내부 로직                                                        #
    # ====================================================================== #
    def _train_epoch(self, epoch: int) -> float:
        """학습 데이터를 한 번 순회한다; 토큰당 평균 loss를 반환한다."""
        self.model.train()
        meter = AverageMeter()
        num_batches = len(self.train_loader)
        progress = tqdm(self.train_loader, desc=f"epoch {epoch + 1}", leave=False)

        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(progress):
            loss, num_tokens = self._train_step(batch)
            meter.update(loss, weight=num_tokens)

            # 누적 경계에서 그리고 마지막 배치에서 스텝하여, 끝에 남는
            # 부분적인 누적 구간에서도 가중치가 업데이트되게 한다.
            is_boundary = (batch_index + 1) % self.accumulation_steps == 0
            if is_boundary or (batch_index + 1) == num_batches:
                self._optimizer_step()
                if self.global_step % self.config.training.log_every == 0:
                    self._log_train_step(loss)

            progress.set_postfix(
                loss=f"{meter.average:.4f}", lr=f"{self._current_lr():.2e}"
            )

        self.logger.info("epoch %d — train loss %.4f", epoch + 1, meter.average)
        return meter.average

    def _train_step(self, batch: dict[str, torch.Tensor]) -> tuple[float, int]:
        """하나의 micro-batch에 대한 forward + backward.

        Returns:
            평균 계산을 위한 ``(스케일 되지 않은 loss 값, 패딩이 아닌 토큰 개수)``.
        """
        batch = move_to_device(batch, self.device)
        targets = batch["tgt_output"]

        with amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            # batch.get("image"): Multimodal이면 (B, C, H, W) 이미지 텐서,
            # 텍스트-only면 None (모델이 이미지 경로를 건너뛴다).
            logits = self.model(batch["src"], batch.get("image"), batch["tgt_input"])
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        # accumulation_steps로 나눠서, micro-batch gradient들의 합이 하나의
        # 큰 배치에 대한 gradient와 같아지도록 한다.
        self.scaler.scale(loss / self.accumulation_steps).backward()

        num_tokens = int((targets != self.pad_id).sum().item())
        return loss.item(), num_tokens

    def _optimizer_step(self) -> None:
        """gradient를 클리핑, 적용, 초기화한다; 스케줄러와 스텝 카운트를 진행한다."""
        clip = self.config.optimization.gradient_clip
        if clip > 0:
            # 임계값이 (loss-scale이 아닌) 실제 단위가 되도록 gradient의
            # 스케일을 먼저 되돌려야 한다.
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_step += 1

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _log_train_step(self, loss: float) -> None:
        """스텝별 스칼라 값을 TensorBoard에 기록한다."""
        self.writer.add_scalar("train/loss", loss, self.global_step)
        self.writer.add_scalar("train/lr", self._current_lr(), self.global_step)

    # ====================================================================== #
    #  검증 & early stopping                                                 #
    # ====================================================================== #
    def _validate(self, epoch: int) -> dict[str, float]:
        """검증 분할에 대해 평가하고 결과를 로깅한다.

        teacher-forced 지표(loss/perplexity/정확도)를 먼저 계산하고, valid
        BLEU가 켜져 있으면 생성 기반 BLEU를 metrics dict에 추가한다. 이후
        아래 루프가 (bleu 포함) 모든 지표를 train.log와 TensorBoard에 자동
        기록한다.
        """
        metrics = evaluate(self.model, self.valid_loader, self.device, self.pad_id)
        if self.valid_bleu:
            assert self.tgt_vocab is not None  # valid_bleu는 tgt_vocab이 있을 때만 True
            metrics["bleu"] = evaluate_bleu(
                self.model, self.valid_loader, self.device, self.tgt_vocab,
                max_length=self.gen_max_length,
                min_length=self.config.inference.min_length,
            )
        self.last_val_metrics = metrics
        for name, value in metrics.items():
            self.writer.add_scalar(f"valid/{name}", value, epoch + 1)
        self.logger.info("epoch %d — valid %s", epoch + 1, _format_metrics(metrics))
        return metrics

    def _should_stop_early(self) -> bool:
        """patience가 소진되면 True (patience 0이면 이 기능이 비활성화됨)."""
        patience = self.config.training.early_stopping_patience
        return patience > 0 and self.validations_without_improvement >= patience

    # ====================================================================== #
    #  영속화                                                                #
    # ====================================================================== #
    def _state_dict(self, epoch: int) -> dict[str, Any]:
        """학습을 재개하거나 나중에 추론을 실행하는 데 필요한 모든 것."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,                       # 마지막으로 완료된 epoch
            "global_step": self.global_step,
            "best_value": self.checkpoints.best_value,
            "validations_without_improvement": self.validations_without_improvement,
            "config": self.config.to_dict(),      # 추론이 나중에 모델을 재구성할 수 있게 함
        }

    def _resume(self, path: str) -> None:
        """model/optimizer/scheduler/scaler와 루프 카운터를 복원한다."""
        self.logger.info("Resuming from %s", path)
        state = CheckpointManager.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]
        self.start_epoch = state["epoch"] + 1
        self.checkpoints.best_value = state.get("best_value")
        self.validations_without_improvement = state.get("validations_without_improvement", 0)
        self.logger.info(
            "Resumed at epoch %d (global step %d, best %s=%s)",
            self.start_epoch, self.global_step,
            self.checkpoints.metric_name, self.checkpoints.best_value,
        )


def _format_metrics(metrics: dict[str, float]) -> str:
    """로그 줄을 위한 'loss=1.2345 | perplexity=3.44 | ...' 형태."""
    return " | ".join(f"{name}={value:.4f}" for name, value in metrics.items())
