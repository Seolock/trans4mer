"""
===============================================================================
 파일: inference/translator.py
 목적:
    BPE 파이프라인용 종단간 번역기: 원본 문장 -> 토큰화 -> BPE ->
    id 인코딩 -> Transformer 디코딩(greedy / 빔 서치) -> id 디코딩 ->
    BPE 제거 -> 번역 문장.

 역할:
    test.py(테스트 세트 번역 + BLEU)와 translate.py(단일 문장 CLI)가
    공유하는 단일 진입점. 모델은 체크포인트에 내장된 설정으로부터
    재구성되므로 아키텍처가 가중치와 절대 어긋나지 않는다.

 입력 / 출력:
    translate(text: str) -> str            (문장 하나 -> 번역 하나)
    translate_batch(list[str]) -> list[str]

 구현 세부사항:
    - 입력 전처리(simple_tokenize + lowercase + BPE)는 preprocess.py와
      정확히 같은 함수/설정을 사용한다 (utils/text.py, codes.bpe).
    - 빔 서치는 기존 inference/beam_search.py의 BeamSearchDecoder를
      그대로 재사용한다 (id 기반이라 어휘집 종류와 무관).
    - greedy는 argmax 루프로 직접 구현 (inference.min_length 존중,
      완료된 행은 패딩을 이어 붙여 배치 형태 유지).
    - 생성 길이는 fairseq 방식으로 소스 길이에 비례해 정한다:
      int(max_len_a*src_len + max_len_b), 단 inference.max_length와
      model.max_seq_length-1을 하드 캡으로 상한한다.
===============================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from config.config import Config
from inference.beam_search import BeamSearchDecoder
from inference.decode import greedy_decode
from models.transformer import Transformer
from trainer.checkpoint import CheckpointManager
from utils.data_paths import codes_path, pair_dir, vocab_path
from utils.image import build_image_transform, load_image_tensor
from utils.misc import get_device
from utils.text import remove_bpe, simple_tokenize
from vocab import Vocab


class Translator:
    """학습된 체크포인트로 원본 문장을 번역하는 종단간 래퍼.

    Args:
        model: 학습된 Transformer (이미 ``device``에 있고 eval 모드).
        src_vocab: 소스 언어 어휘집 (vocab.{src_lang}).
        tgt_vocab: 타겟 언어 어휘집 (vocab.{tgt_lang}).
        bpe: 학습 때와 동일한 subword_nmt.apply_bpe.BPE 인스턴스.
        config: 전체 설정 (inference 섹션이 디코딩 전략을 제어).
        device: 디코딩할 디바이스.
    """

    def __init__(
        self,
        model: Transformer,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        bpe: "object",
        config: Config,
        device: torch.device,
    ) -> None:
        self.model = model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.bpe = bpe
        self.config = config
        self.device = device
        # positional encoding이 지원하는 길이를 절대 넘지 않는 하드 캡.
        self.max_len_cap = min(config.inference.max_length, config.model.max_seq_length - 1)
        # fairseq 스타일 소스 비례 생성 길이 계수 (배치별로 실제 길이를 계산).
        self.max_len_a = config.inference.max_len_a
        self.max_len_b = config.inference.max_len_b
        # Multimodal: 이미지 전처리 transform을 한 번만 만들어 재사용한다.
        self.use_image = config.multimodal.use_image
        self.image_transform = (
            build_image_transform(config.multimodal.image_size) if self.use_image else None
        )

    # ------------------------------------------------------------- 팩토리
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config_override: Optional[Config] = None,
        device: Optional[torch.device] = None,
    ) -> "Translator":
        """체크포인트 + 전처리 산출물로부터 번역기를 조립한다.

        Args:
            checkpoint_path: Trainer가 저장한 .pt 파일.
            config_override: 체크포인트의 것 대신 이 설정의
                inference/dataset 섹션을 사용 (모델 섹션은 항상
                체크포인트에서 온다 — 가중치와 일치해야 하므로).
            device: 대상 디바이스 (생략하면 자동 감지).

        Returns:
            바로 사용 가능한 :class:`Translator`.

        Raises:
            FileNotFoundError: 체크포인트/어휘집/BPE 코드가 없을 때.
        """
        from subword_nmt.apply_bpe import BPE  # 지연 임포트

        state = CheckpointManager.load(checkpoint_path, map_location="cpu")
        config = Config.from_dict(state["config"])
        if config_override is not None:
            # 디코딩과 데이터 옵션은 조정될 수 있지만 아키텍처는 안 된다.
            config.inference = config_override.inference
            config.dataset = config_override.dataset
        device = device or get_device(config.training.device)

        # ------------------------- 모델 재구성 + 가중치 로드
        model = Transformer(config).to(device)
        model.load_state_dict(state["model"])

        # ------------------------- 어휘집 + BPE 코드 로드 (쌍별 디렉터리)
        d = config.dataset
        base_dir = pair_dir(d.bin_dir, d.src_lang, d.tgt_lang)
        src_vocab = Vocab.load(vocab_path(base_dir, d.src_lang))
        tgt_vocab = Vocab.load(vocab_path(base_dir, d.tgt_lang))
        codes_file = codes_path(base_dir, config.bpe.codes_filename)
        if not codes_file.exists():
            raise FileNotFoundError(
                f"BPE codes not found: {codes_file} — run `python preprocess.py` first."
            )
        with open(codes_file, "r", encoding="utf-8") as fh:
            bpe = BPE(fh)

        return cls(model, src_vocab, tgt_vocab, bpe, config, device)

    # -------------------------------------------------------- 입력 인코딩
    def _encode_batch(self, texts: list[str]) -> Tensor:
        """원본 문장 배치를 패딩된 소스 id 텐서로 변환한다.

        토큰화 -> BPE -> id 인코딩 -> truncation -> 오른쪽 패딩.

        Args:
            texts: 원본 입력 문장들.

        Returns:
            ``(batch, max_src_len)`` int64 텐서 (device 위에 있음).
        """
        lowercase = self.config.dataset.lowercase
        encoded: list[list[int]] = []
        for text in texts:
            tokenized = " ".join(simple_tokenize(text, lowercase))
            bpe_tokens = self.bpe.process_line(tokenized).split()
            ids = self.src_vocab.encode(bpe_tokens)[: self.config.model.max_seq_length]
            encoded.append(ids or [self.src_vocab.unk_id])  # 빈 입력 방어

        max_len = max(len(ids) for ids in encoded)
        src = torch.full(
            (len(texts), max_len), self.src_vocab.pad_id, dtype=torch.long, device=self.device
        )
        for row, ids in enumerate(encoded):
            src[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
        return src

    def _encode_images(self, images: list[str]) -> Tensor:
        """이미지 경로 배치를 모델 입력 텐서로 로드/전처리한다.

        Args:
            images: 이미지 파일 경로들 (texts와 인덱스 정렬).

        Returns:
            ``(batch, C, H, W)`` float 텐서 (device 위에 있음).
        """
        tensors = [load_image_tensor(path, transform=self.image_transform) for path in images]
        return torch.stack(tensors, dim=0).to(self.device)

    # -------------------------------------------------------------- 디코딩
    def _greedy_search(
        self, src: Tensor, max_length: int, image: Tensor | None = None
    ) -> list[list[int]]:
        """argmax greedy 디코딩 (재사용 가능한 inference.decode.greedy_decode 위임).

        Args:
            src: ``(batch, src_len)`` 오른쪽 패딩된 소스 id.
            max_length: 이 배치의 최대 생성 길이 (소스 비례로 계산됨).
            image: ``(batch, C, H, W)`` 이미지 텐서 (Multimodal), 또는 None.

        Returns:
            예제마다 생성된 타겟 id 리스트 (BOS/EOS 제외).
        """
        return greedy_decode(
            self.model,
            src,
            bos_id=self.tgt_vocab.bos_id,
            eos_id=self.tgt_vocab.eos_id,
            pad_id=self.tgt_vocab.pad_id,
            max_length=max_length,
            min_length=self.config.inference.min_length,
            image=image,
        )

    # ------------------------------------------------------------ 공개 API
    def translate_batch(
        self, texts: list[str], images: Optional[list[str]] = None
    ) -> list[str]:
        """원본 문장 배치를 번역한다 (Multimodal이면 이미지와 함께).

        Args:
            texts: 소스 언어 원본 문장들.
            images: 각 문장에 대응되는 이미지 파일 경로들 (use_image일 때).
                None이면 이미지 없이(텍스트-only 경로로) 번역한다.

        Returns:
            타겟 언어 번역 문장들 (BPE 마커가 제거된 일반 텍스트).
        """
        inf = self.config.inference
        src = self._encode_batch(texts)
        # fairseq 스타일 소스 비례 최대 생성 길이: 배치 내 최대 소스 길이(패딩
        # 제외)를 기준으로 int(a*src_len + b)를 계산하고 하드 캡으로 상한.
        src_len = int((src != self.src_vocab.pad_id).sum(dim=1).max())
        gen_len = min(int(self.max_len_a * src_len + self.max_len_b), self.max_len_cap)
        # use_image이고 이미지 경로가 주어졌을 때만 이미지 텐서를 만든다.
        image = self._encode_images(images) if (self.use_image and images is not None) else None

        if inf.beam_size > 1:
            decoder = BeamSearchDecoder(
                self.model,
                bos_id=self.tgt_vocab.bos_id,
                eos_id=self.tgt_vocab.eos_id,
                pad_id=self.tgt_vocab.pad_id,
                beam_size=inf.beam_size,
                max_length=gen_len,
                min_length=inf.min_length,
                length_penalty=inf.length_penalty,
            )
            outputs = decoder.search(src, image=image)
        else:
            outputs = self._greedy_search(src, max_length=gen_len, image=image)

        # id -> BPE 토큰 -> 문자열 -> BPE 마커 제거.
        return [remove_bpe(" ".join(self.tgt_vocab.decode(ids))) for ids in outputs]

    def translate(self, text: str, image: Optional[str] = None) -> str:
        """문장 하나를 (선택적으로 이미지와 함께) 번역한다."""
        images = [image] if image is not None else None
        return self.translate_batch([text], images=images)[0]
