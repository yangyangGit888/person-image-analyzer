"""SAM 2.1 本地分割 Provider。

依赖为可选项：未安装 torch/sam2 或缺少权重时，仅编辑功能返回
SEGMENTATION_UNAVAILABLE，不影响原有分析接口。
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from app.ai.segmentation.base import (
    BaseSegmentationProvider,
    SegmentationMask,
    SegmentationPrompt,
)
from app.core.config import Settings
from app.core.exceptions import SegmentationError, SegmentationUnavailableError

logger = logging.getLogger(__name__)


class SAM2SegmentationProvider(BaseSegmentationProvider):
    provider_name = "sam2"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.device = "cpu"
        self._predictor: Any | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._load_error: str | None = None

    async def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        if self._load_error:
            raise SegmentationUnavailableError(detail=self._load_error)
        async with self._load_lock:
            if self._predictor is not None:
                return
            try:
                await asyncio.to_thread(self._load_sync)
            except SegmentationUnavailableError as exc:
                self._load_error = exc.detail
                raise
            except Exception as exc:
                self._load_error = "SAM2 模型加载失败，请检查权重和 PyTorch 环境"
                raise SegmentationUnavailableError(detail=self._load_error) from exc

    def _load_sync(self) -> None:
        checkpoint = Path(self._settings.segmentation_checkpoint).expanduser()
        if not checkpoint.is_file():
            raise SegmentationUnavailableError(
                detail=(
                    "未找到 SAM2 权重文件，请下载 sam2.1_hiera_small.pt 并设置 "
                    "SEGMENTATION_CHECKPOINT"
                )
            )

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise SegmentationUnavailableError(
                detail=(
                    "未安装 SAM2/PyTorch 依赖。请安装 requirements-ml.txt，"
                    "并按 README 安装 facebookresearch/sam2"
                )
            ) from exc

        requested = self._settings.segmentation_device
        if requested == "auto":
            if torch.cuda.is_available():
                requested = "cuda"
            elif torch.backends.mps.is_available():
                requested = "mps"
            else:
                requested = "cpu"

        torch.set_grad_enabled(False)
        try:
            predictor = self._build_predictor(
                build_sam2,
                SAM2ImagePredictor,
                checkpoint,
                requested,
            )
        except Exception as exc:
            if requested != "mps":
                raise
            logger.warning("SAM2 MPS 初始化失败，回退 CPU: %s", type(exc).__name__)
            predictor = self._build_predictor(
                build_sam2,
                SAM2ImagePredictor,
                checkpoint,
                "cpu",
            )
            requested = "cpu"

        self._predictor = predictor
        self.device = requested
        logger.info("SAM2 已加载: device=%s", requested)

    def _build_predictor(
        self,
        build_sam2: Any,
        predictor_cls: Any,
        checkpoint: Path,
        device: str,
    ) -> Any:
        model = build_sam2(
            self._settings.segmentation_model_cfg,
            str(checkpoint),
            device=device,
        )
        model.eval()
        return predictor_cls(model)

    async def segment(
        self,
        image: np.ndarray,
        prompts: list[SegmentationPrompt],
    ) -> list[SegmentationMask]:
        if not prompts:
            return []
        await self._ensure_loaded()
        async with self._inference_lock:
            return await asyncio.to_thread(self._segment_sync, image, prompts)

    def _segment_sync(
        self,
        image: np.ndarray,
        prompts: list[SegmentationPrompt],
    ) -> list[SegmentationMask]:
        assert self._predictor is not None
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        self._predictor.set_image(image)
        results: list[SegmentationMask] = []

        for prompt in prompts:
            kwargs: dict[str, Any] = {"multimask_output": True}
            if prompt.box is not None:
                kwargs["box"] = np.asarray(prompt.box, dtype=np.float32)
            points = list(prompt.positive_points) + list(prompt.negative_points)
            if points:
                kwargs["point_coords"] = np.asarray(points, dtype=np.float32)
                kwargs["point_labels"] = np.asarray(
                    [1] * len(prompt.positive_points)
                    + [0] * len(prompt.negative_points),
                    dtype=np.int32,
                )
            if prompt.box is None and not points:
                continue
            try:
                masks, scores, _ = self._predictor.predict(**kwargs)
            except Exception as exc:
                raise SegmentationError(
                    detail=f"分割提示 {prompt.prompt_id} 执行失败",
                ) from exc

            best_index = 0
            best_score = -1.0
            for index, (mask, score) in enumerate(zip(masks, scores, strict=True)):
                area = int(np.count_nonzero(mask))
                if area == 0:
                    continue
                numeric_score = float(score)
                if numeric_score > best_score:
                    best_index = index
                    best_score = numeric_score

            binary = np.asarray(masks[best_index]) > 0
            if not binary.any():
                raise SegmentationError(
                    detail=f"分割提示 {prompt.prompt_id} 未生成有效 Mask",
                )
            results.append(
                SegmentationMask(
                    prompt_id=prompt.prompt_id,
                    mask=binary,
                    score=max(0.0, min(1.0, best_score)),
                    method="sam2",
                )
            )
        return results

    async def aclose(self) -> None:
        self._predictor = None
