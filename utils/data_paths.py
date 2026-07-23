"""
===============================================================================
 파일: utils/data_paths.py
 목적:
    BPE 전처리 산출물의 파일 경로 규칙을 한 곳에서 관리한다.

 역할:
    preprocess.py(쓰기), dataset.py / inference/translator.py / test.py
    (읽기)가 모두 같은 함수를 사용하므로, 파일 이름 규칙이 프로젝트
    전체에서 절대 어긋나지 않는다. 경로는 전부 config(dataset.raw_dir,
    dataset.bin_dir 등)로부터 유도되며 하드코딩되지 않는다.

 입력 / 출력:
    원본:   raw_dir/{split}.{lang}                (예: data/raw/train.en)
    산출물: bin_dir/{src}-{tgt}/ 아래에 쌍별로 분리:
        codes.bpe                BPE merge 규칙 (해당 쌍의 train으로 공동 학습)
        {split}.bpe.{lang}       BPE가 적용된 텍스트
        {split}.ids.{lang}       토큰 id로 변환된 텍스트 (공백 구분 정수)
        vocab.{lang}             어휘집 (한 줄에 토큰 하나, 특수 토큰 선두)

 구현 세부사항:
    - pair_dir()가 언어쌍 하위 디렉터리(en-de, en-fr ...)를 만들며, 나머지
      경로 함수들은 그 디렉터리를 첫 인자로 받는다.
    - 원본 파일 이름 변형(valid vs val, test_2016_flickr vs test2016 ...)의
      자동 감지는 preprocess.find_raw_file()이 담당한다.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path


def pair_name(src_lang: str, tgt_lang: str) -> str:
    """언어쌍의 표준 이름 (예: ("en","de") -> "en-de")."""
    return f"{src_lang}-{tgt_lang}"


def pair_dir(bin_dir: str | Path, src_lang: str, tgt_lang: str) -> Path:
    """언어쌍별 산출물 디렉터리 경로 (예: data/data-bin/en-de)."""
    return Path(bin_dir) / pair_name(src_lang, tgt_lang)


def codes_path(pair_directory: str | Path, filename: str = "codes.bpe") -> Path:
    """해당 쌍의 BPE merge 규칙 파일 경로."""
    return Path(pair_directory) / filename


def bpe_path(pair_directory: str | Path, split: str, lang: str) -> Path:
    """BPE가 적용된 텍스트 파일 경로 (예: en-de/train.bpe.en)."""
    return Path(pair_directory) / f"{split}.bpe.{lang}"


def ids_path(pair_directory: str | Path, split: str, lang: str) -> Path:
    """토큰 id로 변환된 파일 경로 (예: en-de/train.ids.en)."""
    return Path(pair_directory) / f"{split}.ids.{lang}"


def vocab_path(pair_directory: str | Path, lang: str) -> Path:
    """언어별 어휘집 파일 경로 (예: en-de/vocab.en)."""
    return Path(pair_directory) / f"vocab.{lang}"


def default_checkpoint_path(
    save_dir: str | Path, src_lang: str, tgt_lang: str, filename: str = "ensemble.pt"
) -> Path:
    """활성 쌍의 기본 체크포인트 경로를 해석한다.

    train.py의 저장 규칙과 동일: save_dir가 이미 쌍 이름으로 끝나지
    않으면 쌍 하위 디렉터리를 붙인다 -> {save_dir}/{src}-{tgt}/{filename}.
    test.py와 translate.py가 --checkpoint 미지정 시 사용한다. 기본값은
    ensemble.pt(최근 여러 epoch 가중치 평균 — 단일 체크포인트보다 일반적으로
    더 안정적)이며, filename="best.pt"로 넘기면 단일 best 체크포인트를 쓸 수 있다.
    """
    save_dir = Path(save_dir)
    if save_dir.name != pair_name(src_lang, tgt_lang):
        save_dir = save_dir / pair_name(src_lang, tgt_lang)
    return save_dir / filename
