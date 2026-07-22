"""
===============================================================================
 파일: preprocess.py
 목적:
    Multi30k 원본 텍스트를 학습 가능한 형태로 변환하는 전처리 진입점.
    설정된 모든 언어쌍(기본: en-de, en-fr)을 한 번에 전처리한다.

    Raw Text (data/raw/)
        ↓  (토큰화/소문자화)
    subword-nmt BPE 학습  (쌍별: train.src + train.tgt 공동)
        ↓
    BPE 적용  (6개 split × 2개 언어)
        ↓
    언어별 Vocabulary 생성
        ↓
    토큰 id 변환  ->  data/data-bin/{src}-{tgt}/

 역할:
    `python preprocess.py` 한 번으로 언어쌍마다 아래 산출물을
    dataset.bin_dir/{pair}/ 에 만든다 (경로 규칙은 utils/data_paths.py):
        codes.bpe                          해당 쌍의 BPE merge 규칙
        {split}.bpe.{lang}                 BPE 적용 텍스트
        vocab.{src_lang}, vocab.{tgt_lang} 언어별 어휘집
        {split}.ids.{lang}                 공백 구분 정수 id (Dataset 입력)
    split = train, val, test2016, test2017, testcoco, test2018

 입력 / 출력:
    입력 : config/default.yaml (dataset.raw_dir / bin_dir / lang_pairs /
           lowercase, bpe.num_merges / vocab_size / min_freq) + 원본 파일.
    출력 : 위 산출물 파일들 + 콘솔 통계.

 구현 세부사항:
    - 원본 파일 이름은 배포본마다 다르므로 자동 감지한다:
        val      -> val | valid | dev
        test2016 -> test2016 | test_2016_flickr | test_2016
        testcoco -> testcoco | test_2017_mscoco | test_coco   등
      (정확한 이름 우선, 없으면 접두사 glob으로 대체.)
    - BPE는 쌍별로 독립 학습한다: en-de 코드는 train.en+train.de로,
      en-fr 코드는 train.en+train.fr로. 따라서 같은 en 문장이라도 쌍에
      따라 분절이 다를 수 있으며, 어휘집도 쌍별 디렉터리에 저장된다.
    - 특수 토큰 <pad>=0, <unk>=1, <s>=2, </s>=3 은 vocab.py가 보장한다.
    - 이미 존재하는 산출물은 재사용하며 --force 로 강제 재생성한다.
===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from config.config import Config
from utils.data_paths import bpe_path, codes_path, ids_path, pair_dir, vocab_path
from utils.logger import get_logger
from utils.text import simple_tokenize
from vocab import Vocab

logger = get_logger("preprocess")

# 전처리 대상 split (정규화된 이름; 산출물 파일명에 그대로 사용됨).
SPLITS: tuple[str, ...] = ("train", "val", "test2016", "test2017", "testcoco", "test2018")

# 정규화된 split 이름 -> 원본 파일 이름 후보 (우선순위 순).
SPLIT_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train",),
    "val": ("val", "valid", "dev"),
    "test2016": ("test2016", "test_2016_flickr", "test_2016"),
    "test2017": ("test2017", "test_2017_flickr", "test_2017"),
    "testcoco": ("testcoco", "test_2017_mscoco", "test_coco"),
    "test2018": ("test2018", "test_2018_flickr", "test_2018"),
}


def find_raw_file(raw_dir: str | Path, split: str, lang: str) -> Path:
    """분할(split)과 언어에 해당하는 원본 코퍼스 파일을 자동 감지한다.

    Args:
        raw_dir: 원본 코퍼스 디렉터리 (dataset.raw_dir).
        split: SPLITS의 정규화된 split 이름.
        lang: 언어 코드 (파일 확장자, 예: "en").

    Returns:
        존재하는 원본 파일의 Path.

    Raises:
        KeyError: 알 수 없는 split 이름일 때.
        FileNotFoundError: 어떤 후보 이름으로도 파일을 찾지 못했을 때
            (시도한 패턴을 에러 메시지에 나열).
    """
    raw_dir = Path(raw_dir)
    tried: list[str] = []
    for alias in SPLIT_ALIASES[split]:
        # 1) 정확한 이름 우선 (예: val.en)
        exact = raw_dir / f"{alias}.{lang}"
        tried.append(exact.name)
        if exact.exists():
            return exact
        # 2) 접두사 glob 대체 (예: test_2016_flickr.en 등 변형 이름).
        #    전처리 산출물(*.bpe.*, *.ids.*)은 원본이 아니므로 제외한다
        #    (raw_dir가 분리되어 있어도 안전장치로 유지).
        matches = sorted(
            p
            for p in raw_dir.glob(f"{alias}*.{lang}")
            if ".bpe." not in p.name and ".ids." not in p.name
        )
        tried.append(f"{alias}*.{lang}")
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No raw file for split '{split}' / lang '{lang}' in {raw_dir} "
        f"(tried: {', '.join(tried)})"
    )


def read_tokenized_lines(path: Path, lowercase: bool) -> list[str]:
    """원본 파일을 읽어 토큰화된(공백 구분) 문장 리스트를 반환한다.

    Args:
        path: 원본 텍스트 파일 (한 줄에 한 문장).
        lowercase: 소문자화 여부 (config: dataset.lowercase).

    Returns:
        토큰들이 공백으로 이어진 문장 리스트 (빈 줄은 그대로 보존해
        소스/타겟 줄 정렬이 어긋나지 않게 한다).
    """
    with open(path, "r", encoding="utf-8") as fh:
        return [" ".join(simple_tokenize(line, lowercase)) for line in fh]


def learn_bpe_codes(
    config: Config, src_lang: str, tgt_lang: str, codes_file: Path, force: bool
) -> None:
    """한 언어쌍의 train 소스+타겟을 합쳐 공동 BPE merge 규칙을 학습한다.

    Args:
        config: 전체 설정 (bpe.num_merges, dataset.raw_dir/lowercase 사용).
        src_lang: 쌍의 소스 언어 코드.
        tgt_lang: 쌍의 타겟 언어 코드.
        codes_file: 학습된 규칙을 저장할 경로 (쌍별 디렉터리 안).
        force: True면 기존 파일이 있어도 다시 학습한다.
    """
    if codes_file.exists() and not force:
        logger.info("[%s-%s] BPE codes already exist: %s (use --force to relearn)",
                    src_lang, tgt_lang, codes_file)
        return

    from subword_nmt.learn_bpe import learn_bpe  # 지연 임포트 (미설치 시 명확한 에러)

    d = config.dataset
    src_file = find_raw_file(d.raw_dir, "train", src_lang)
    tgt_file = find_raw_file(d.raw_dir, "train", tgt_lang)
    logger.info(
        "[%s-%s] Learning joint BPE (%d merges) from %s + %s ...",
        src_lang, tgt_lang, config.bpe.num_merges, src_file.name, tgt_file.name,
    )
    # learn_bpe는 라인 이터러블을 받으므로, 토큰화된 두 코퍼스를 이어 붙인
    # 리스트를 그대로 전달한다.
    joint_lines = read_tokenized_lines(src_file, d.lowercase) + read_tokenized_lines(
        tgt_file, d.lowercase
    )
    codes_file.parent.mkdir(parents=True, exist_ok=True)
    with open(codes_file, "w", encoding="utf-8") as out:
        learn_bpe(joint_lines, out, num_symbols=config.bpe.num_merges, min_frequency=2)
    logger.info("[%s-%s] Saved BPE codes -> %s", src_lang, tgt_lang, codes_file)


def apply_bpe_to_corpus(
    config: Config, src_lang: str, tgt_lang: str, out_dir: Path, codes_file: Path, force: bool
) -> None:
    """한 쌍의 모든 split × 언어 조합에 BPE를 적용해 {split}.bpe.{lang}을 만든다.

    Args:
        config: 전체 설정.
        src_lang / tgt_lang: 쌍의 언어 코드.
        out_dir: 쌍별 산출물 디렉터리 (bin_dir/{pair}).
        codes_file: 이 쌍의 BPE 규칙 파일.
        force: True면 기존 산출물이 있어도 다시 적용한다.
    """
    from subword_nmt.apply_bpe import BPE  # 지연 임포트

    d = config.dataset
    with open(codes_file, "r", encoding="utf-8") as fh:
        bpe = BPE(fh)

    for split in SPLITS:
        for lang in (src_lang, tgt_lang):
            out_path = bpe_path(out_dir, split, lang)
            if out_path.exists() and not force:
                logger.info("[%s-%s] BPE output already exists: %s",
                            src_lang, tgt_lang, out_path.name)
                continue
            raw_path = find_raw_file(d.raw_dir, split, lang)
            lines = read_tokenized_lines(raw_path, d.lowercase)
            with open(out_path, "w", encoding="utf-8") as out:
                for line in tqdm(lines, desc=f"apply_bpe {split}.{lang}", leave=False):
                    out.write(bpe.process_line(line) + "\n")
            logger.info("[%s-%s] %s (%d lines) -> %s",
                        src_lang, tgt_lang, raw_path.name, len(lines), out_path.name)


def build_vocabularies(
    config: Config, src_lang: str, tgt_lang: str, out_dir: Path, force: bool
) -> dict[str, Vocab]:
    """한 쌍의 언어별 어휘집을 구축/로드해 vocab.{lang}으로 저장한다.

    Args:
        config: 전체 설정 (bpe.vocab_size / min_freq 사용).
        src_lang / tgt_lang: 쌍의 언어 코드.
        out_dir: 쌍별 산출물 디렉터리.
        force: True면 기존 어휘집이 있어도 다시 구축한다.

    Returns:
        {언어 코드: Vocab} 매핑.
    """
    vocabs: dict[str, Vocab] = {}
    for lang in (src_lang, tgt_lang):
        path = vocab_path(out_dir, lang)
        if path.exists() and not force:
            vocabs[lang] = Vocab.load(path)
            logger.info("[%s-%s] Loaded existing vocab.%s (%d tokens)",
                        src_lang, tgt_lang, lang, len(vocabs[lang]))
            continue
        vocab = Vocab.build(
            files=[bpe_path(out_dir, "train", lang)],
            max_size=config.bpe.vocab_size,
            min_freq=config.bpe.min_freq,
        )
        vocab.save(path)
        vocabs[lang] = vocab
        logger.info("[%s-%s] Built vocab.%s (%d tokens) -> %s",
                    src_lang, tgt_lang, lang, len(vocab), path)
    return vocabs


def convert_to_ids(
    src_lang: str, tgt_lang: str, out_dir: Path, vocabs: dict[str, Vocab], force: bool
) -> None:
    """한 쌍의 BPE 텍스트를 토큰 id 파일({split}.ids.{lang})로 변환한다.

    한 줄 = 한 문장 = 공백 구분 정수. BOS/EOS는 여기서 붙이지 않으며
    Dataset(dataset.py)이 타겟 쪽에만 추가한다.

    Args:
        src_lang / tgt_lang: 쌍의 언어 코드.
        out_dir: 쌍별 산출물 디렉터리.
        vocabs: :func:`build_vocabularies`가 반환한 언어별 어휘집.
        force: True면 기존 산출물이 있어도 다시 변환한다.
    """
    for split in SPLITS:
        for lang in (src_lang, tgt_lang):
            out_path = ids_path(out_dir, split, lang)
            if out_path.exists() and not force:
                logger.info("[%s-%s] ids output already exists: %s",
                            src_lang, tgt_lang, out_path.name)
                continue
            vocab = vocabs[lang]
            in_path = bpe_path(out_dir, split, lang)
            with open(in_path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            with open(out_path, "w", encoding="utf-8") as out:
                for line in tqdm(lines, desc=f"to_ids {split}.{lang}", leave=False):
                    token_ids = vocab.encode(line.split())
                    out.write(" ".join(str(i) for i in token_ids) + "\n")
            logger.info("[%s-%s] %s -> %s (%d lines)",
                        src_lang, tgt_lang, in_path.name, out_path.name, len(lines))


def preprocess_pair(config: Config, src_lang: str, tgt_lang: str, force: bool) -> dict[str, Vocab]:
    """한 언어쌍의 전처리 전체: BPE 학습 -> 적용 -> 어휘집 -> id 변환.

    Args:
        config: 전체 설정.
        src_lang / tgt_lang: 쌍의 언어 코드.
        force: 산출물 강제 재생성 여부.

    Returns:
        해당 쌍의 {언어 코드: Vocab} 매핑.
    """
    out_dir = pair_dir(config.dataset.bin_dir, src_lang, tgt_lang)
    out_dir.mkdir(parents=True, exist_ok=True)
    codes_file = codes_path(out_dir, config.bpe.codes_filename)

    # 1) BPE merge 규칙 학습 (train.src + train.tgt 공동)
    learn_bpe_codes(config, src_lang, tgt_lang, codes_file, force)
    # 2) 모든 split/언어에 BPE 적용
    apply_bpe_to_corpus(config, src_lang, tgt_lang, out_dir, codes_file, force)
    # 3) 언어별 어휘집 구축
    vocabs = build_vocabularies(config, src_lang, tgt_lang, out_dir, force)
    # 4) 토큰 id 변환
    convert_to_ids(src_lang, tgt_lang, out_dir, vocabs, force)
    return vocabs


def main() -> None:
    """설정된 모든 언어쌍에 대해 전처리 파이프라인을 실행한다."""
    parser = argparse.ArgumentParser(description="Multi30k BPE preprocessing (multi-pair)")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config path")
    parser.add_argument(
        "--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
        help="config overrides (repeatable), e.g. --set bpe.num_merges=8000",
    )
    parser.add_argument("--force", action="store_true", help="regenerate all artifacts")
    args = parser.parse_args()

    config = Config.from_yaml(args.config, overrides=args.set)
    raw_dir = Path(config.dataset.raw_dir)
    if not raw_dir.exists():
        logger.error("raw_dir does not exist: %s", raw_dir)
        sys.exit(1)

    summary: list[str] = []
    try:
        for pair in config.dataset.lang_pairs:
            src_lang, tgt_lang = pair.split("-")
            logger.info("=== Preprocessing pair %s-%s ===", src_lang, tgt_lang)
            vocabs = preprocess_pair(config, src_lang, tgt_lang, args.force)
            summary.append(
                f"{pair}: vocab.{src_lang}={len(vocabs[src_lang])}, "
                f"vocab.{tgt_lang}={len(vocabs[tgt_lang])}"
            )
    except FileNotFoundError as error:
        logger.error("%s", error)
        sys.exit(1)
    except ImportError as error:
        logger.error("Missing dependency: %s — run `pip install -r requirements.txt`", error)
        sys.exit(1)

    logger.info("Preprocessing complete — %s | artifacts in %s",
                " | ".join(summary), config.dataset.bin_dir)


if __name__ == "__main__":
    main()
