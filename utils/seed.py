"""
===============================================================================
 파일: utils/seed.py
 목적:
    한 번의 호출로 재현성 확보: 프로젝트가 다루는 모든 난수 생성기
    (Python, NumPy, PyTorch CPU와 CUDA)를 시드로 고정한다.

 역할:
    `training.seed`와 함께 train.py / test.py 맨 처음에 호출된다.

 입력 / 출력:
    입력 : 정수 시드 (+ 선택적 결정성 플래그).
    출력 : 없음 (전역 RNG 상태가 변경됨).

 구현 세부사항:
    - `deterministic=True`이면 추가로 cuDNN을 결정적 모드로 강제하고
      autotuner를 비활성화한다. 이렇게 하면 동일한 하드웨어/소프트웨어
      환경에서 실행마다 비트 단위로 동일한 결과가 나오지만 실제 속도
      비용이 있으므로 선택 사항으로 남겨둔다.
    - DataLoader 워커들은 torch의 base seed로부터 자동으로 재시드되므로,
      우리의 사용 사례에는 별도의 worker_init_fn이 필요하지 않다.
===============================================================================
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """프로젝트가 사용하는 모든 RNG를 시드로 고정한다.

    Args:
        seed: Python, NumPy, PyTorch(CPU + 모든 GPU)에 적용할 시드.
        deterministic: True면 결정적 cuDNN 커널도 강제한다 (더 느리지만
            동일한 하드웨어에서 비트 단위로 재현 가능).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 해시 랜덤화는 Python set/dict의 순회 순서에 영향을 준다.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
