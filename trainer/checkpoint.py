"""
===============================================================================
 파일: trainer/checkpoint.py
 목적:
    체크포인트 영속화: 주기적 스냅샷, 롤링되는 "last", 추적되는 "best",
    그리고 재개/추론을 위한 로딩.

 역할:
    Trainer가 매 epoch마다 완전한 상태 dict를 여기에 넘긴다; 매니저는
    파일 이름과 best-model 관리를 결정한다. test.py와 추론 코드는 같은
    클래스를 통해 체크포인트를 로드한다.

 입력 / 출력:
    체크포인트는 다음을 담은 일반 dict가 들어있는 하나의 .pt 파일이다:
        model / optimizer / scheduler / scaler의 state_dict,
        epoch, global_step, best_value, patience 카운터,
        그리고 설정 dict 전체 (그래서 추론이 원본 YAML에 접근하지
        않고도 모델을 재구성할 수 있음).

 구현 세부사항:
    - "더 나음"은 지표에 따라 다르다: loss/perplexity -> 낮을수록 좋음,
      accuracy들 -> 높을수록 좋음. `_HIGHER_IS_BETTER`에 한 번만
      인코딩되어 있다.
    - 체크포인트에는 텐서와 일반 Python 컨테이너만 담기므로 여기서
      `torch.load(weights_only=True)`가 안전하다.
    - 파일: last.pt (항상), best.pt (개선 시), epoch_%03d.pt
      (`save_every` epoch마다).
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from utils.logger import get_logger

logger = get_logger(__name__)

# 지원하는 best_metric마다의 개선 방향.
_HIGHER_IS_BETTER: dict[str, bool] = {
    "loss": False,
    "perplexity": False,
    "token_accuracy": True,
    "accuracy": True,
    "bleu": True,
}


class CheckpointManager:
    """체크포인트를 저장/로드하고 최고 검증 지표를 추적한다.

    Args:
        save_dir: 모든 체크포인트 파일을 담을 디렉터리 (없으면 생성).
        metric_name: 어떤 검증 지표가 "최고"를 정의하는지
            (loss / perplexity / token_accuracy / accuracy 중 하나).
    """

    def __init__(self, save_dir: str | Path, metric_name: str = "loss") -> None:
        if metric_name not in _HIGHER_IS_BETTER:
            raise ValueError(
                f"Unsupported best_metric '{metric_name}'. "
                f"Choose from {sorted(_HIGHER_IS_BETTER)}"
            )
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.higher_is_better = _HIGHER_IS_BETTER[metric_name]
        self.best_value: Optional[float] = None

    # ------------------------------------------------------------- 비교
    def is_better(self, value: float) -> bool:
        """``value``가 지금까지 본 최고값보다 개선되었는지 여부."""
        if self.best_value is None:
            return True
        if self.higher_is_better:
            return value > self.best_value
        return value < self.best_value

    def update_best(self, value: float) -> bool:
        """개선이면 ``value``를 기록한다; 개선이었으면 True를 반환한다."""
        if self.is_better(value):
            self.best_value = value
            return True
        return False

    # --------------------------------------------------------------- 쓰기
    def _write(self, state: dict[str, Any], filename: str) -> Path:
        path = self.save_dir / filename
        torch.save(state, path)
        return path

    def save_last(self, state: dict[str, Any]) -> Path:
        """롤링되는 `last.pt`를 덮어쓴다 (재개 지점)."""
        return self._write(state, "last.pt")

    def save_best(self, state: dict[str, Any]) -> Path:
        """`best.pt`를 덮어쓴다 (`update_best`가 True를 반환했을 때만 호출)."""
        path = self._write(state, "best.pt")
        logger.info("New best %s=%.4f -> %s", self.metric_name, self.best_value, path)
        return path

    def save_epoch(self, state: dict[str, Any], epoch: int) -> Path:
        """주기적인 epoch별 스냅샷을 기록한다 (학습 종료 시 정리 대상)."""
        return self._write(state, f"epoch_{epoch:03d}.pt")

    # ----------------------------------------------- 앙상블 & 정리
    def _epoch_checkpoint_paths(self) -> list[Path]:
        """모든 epoch_*.pt를 epoch 번호 오름차순으로 정렬해 반환한다.

        Returns:
            정렬된 Path 리스트 (파일이 없으면 빈 리스트). 파일명이
            "epoch_{번호}.pt" 규칙을 따르지 않는 것은 건너뛴다.
        """
        paths = []
        for path in self.save_dir.glob("epoch_*.pt"):
            try:
                epoch = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue  # 예상치 못한 이름은 무시
            paths.append((epoch, path))
        return [path for _, path in sorted(paths)]

    def save_ensemble(self, last_n: int) -> Optional[Path]:
        """최근 ``last_n``개 epoch 체크포인트를 가중치 평균해 ensemble.pt로 저장한다.

        "Attention Is All You Need"의 checkpoint averaging 기법: 여러
        체크포인트의 model 파라미터를 원소별로 평균한 단일 모델을 만든다.
        추가 메모리/연산 없이 test.py / Translator가 일반 체크포인트처럼
        그대로 로드할 수 있다.

        Args:
            last_n: 평균에 사용할 최근 epoch 체크포인트 개수 (실제 파일이
                더 적으면 있는 만큼만 사용).

        Returns:
            저장된 ensemble.pt 경로, 또는 평균할 epoch 체크포인트가 하나도
            없으면 None.
        """
        paths = self._epoch_checkpoint_paths()
        if last_n > 0:
            paths = paths[-last_n:]
        if not paths:
            logger.warning("No epoch checkpoints found to ensemble in %s", self.save_dir)
            return None

        # 첫 체크포인트로 누적 버퍼를 만들고(부동소수만 float로 승격),
        # 나머지를 더한 뒤 개수로 나눈다. 부동소수가 아닌 텐서(정수 버퍼
        # 등)는 평균이 의미 없으므로 마지막 체크포인트 값을 그대로 쓴다.
        num_ckpts = len(paths)
        averaged: dict[str, torch.Tensor] = {}
        original_dtypes: dict[str, torch.dtype] = {}
        for index, path in enumerate(paths):
            model_state = self.load(path, map_location="cpu")["model"]
            for key, tensor in model_state.items():
                if index == 0:
                    original_dtypes[key] = tensor.dtype
                    averaged[key] = tensor.float() if tensor.is_floating_point() else tensor.clone()
                elif tensor.is_floating_point():
                    averaged[key] += tensor.float()
                else:
                    averaged[key] = tensor.clone()  # 최신 값 유지

        for key, tensor in averaged.items():
            if tensor.is_floating_point():
                averaged[key] = (tensor / num_ckpts).to(original_dtypes[key])

        # 설정/epoch 메타데이터는 가장 최근(마지막) 체크포인트에서 가져온다.
        newest = self.load(paths[-1], map_location="cpu")
        ensemble_epochs = [int(p.stem.split("_")[1]) for p in paths]
        ensemble_state = {
            "model": averaged,
            "config": newest.get("config"),
            "epoch": newest.get("epoch"),
            "ensemble_epochs": ensemble_epochs,
        }
        path = self._write(ensemble_state, "ensemble.pt")
        logger.info("Saved ensemble of %d checkpoints (epochs %s) -> %s",
                    num_ckpts, ensemble_epochs, path)
        return path

    def cleanup_epoch_checkpoints(self) -> int:
        """모든 epoch_*.pt를 삭제한다 (last/best/ensemble은 이름이 달라 보존).

        Returns:
            삭제한 파일 개수.
        """
        paths = self._epoch_checkpoint_paths()
        for path in paths:
            path.unlink()
        if paths:
            logger.info("Removed %d epoch checkpoint(s) from %s", len(paths), self.save_dir)
        return len(paths)

    # --------------------------------------------------------------- 읽기
    @property
    def last_path(self) -> Path:
        return self.save_dir / "last.pt"

    @property
    def best_path(self) -> Path:
        return self.save_dir / "best.pt"

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        """디스크에서 체크포인트 dict를 로드한다.

        Args:
            path: 체크포인트 파일.
            map_location: 저장된 텐서를 어디로 매핑할지 (예: "cpu", "cuda").

        Returns:
            체크포인트 상태 dict.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, map_location=map_location, weights_only=True)
