"""
===============================================================================
 파일: utils/logger.py
 목적:
    프로젝트의 모든 모듈을 위한 일관되고 타임스탬프가 찍힌 콘솔(및
    선택적 파일) 로깅.

 역할:
    print() 대신 어디서나 사용하는 단일 진입점 `get_logger(name)`.

 입력 / 출력:
    입력 : 로거 이름(보통 __name__)과 선택적 로그 파일 경로.
    출력 : 설정이 완료된 `logging.Logger`.

 구현 세부사항:
    - 핸들러는 로거 이름당 한 번만 붙는다; 같은 이름으로 반복 호출해도
      출력이 중복되지 않고 같은 로거를 반환한다.
    - `propagate = False`로 root 로거를 통한 이중 출력을 방지한다.
===============================================================================
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file: Optional[str | Path] = None, level: int = logging.INFO) -> logging.Logger:
    """표준 포맷을 가진 로거를 생성(또는 조회)한다.

    Args:
        name: 로거 이름, 관례적으로 호출하는 모듈의 ``__name__``.
        log_file: 지정하면 이 파일에도 기록을 추가한다 (디렉터리가
            없으면 생성됨). 체크포인트 옆에 학습 로그를 남겨두고
            싶을 때 유용하다.
        level: 출력할 최소 레벨.

    Returns:
        바로 사용 가능한 :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # 로거당 콘솔 핸들러를 한 번만 붙인다.
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        already_attached = any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")) == log_path.resolve()
            for h in logger.handlers
        )
        if not already_attached:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
