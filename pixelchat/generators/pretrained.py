"""SD-Turbo backed generator. Loads once, lazily, and reuses the pipeline."""

import logging
import random
import time

import numpy as np
import torch

from pixelchat import config
from pixelchat.device import get_device_and_dtype
from pixelchat.generators.base import GenerationResult, ImageGenerator

logger = logging.getLogger(__name__)

MAX_SEED = 2**32 - 1


def _is_black_or_nan(image_array: np.ndarray) -> bool:
    """True if the array is all-NaN or effectively solid black."""
    if np.isnan(image_array).any():
        return True
    return bool(np.all(image_array < 2))


class SDTurboGenerator(ImageGenerator):
    """Generates images with stabilityai/sd-turbo via diffusers."""

    name = "sd-turbo"

    def __init__(self) -> None:
        self._pipe = None
        self._device, self._dtype = get_device_and_dtype()

    def _load_pipeline(self):
        from diffusers import AutoPipelineForText2Image

        logger.debug("Loading %s on %s (%s)", config.MODEL_ID, self._device, self._dtype)
        start = time.perf_counter()

        kwargs = {"torch_dtype": self._dtype}
        if self._dtype == torch.float16:
            kwargs["variant"] = "fp16"

        pipe = AutoPipelineForText2Image.from_pretrained(config.MODEL_ID, **kwargs)
        pipe = pipe.to(self._device)
        pipe.enable_attention_slicing()

        elapsed = time.perf_counter() - start
        logger.debug("Pipeline loaded in %.2fs", elapsed)
        return pipe

    def _ensure_loaded(self):
        if self._pipe is None:
            self._pipe = self._load_pipeline()
        return self._pipe

    def _run(self, pipe, prompt: str, seed: int, steps: int):
        generator = torch.Generator("cpu").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=config.GUIDANCE_SCALE,
            height=config.IMAGE_SIZE,
            width=config.IMAGE_SIZE,
            generator=generator,
        )
        return result.images[0]

    def generate(self, prompt: str, seed: int | None = None, steps: int | None = None) -> GenerationResult:
        pipe = self._ensure_loaded()

        if seed is None:
            seed = random.randint(0, MAX_SEED)
        steps = steps or config.DEFAULT_STEPS
        steps = max(config.MIN_STEPS, min(config.MAX_STEPS, steps))

        start = time.perf_counter()
        image = self._run(pipe, prompt, seed, steps)

        if _is_black_or_nan(np.asarray(image, dtype=np.float32)):
            logger.debug("Black/NaN image detected, upcasting VAE to float32 and retrying once")
            pipe.vae.to(torch.float32)
            image = self._run(pipe, prompt, seed, steps)

        elapsed = time.perf_counter() - start
        logger.debug("Generated image for %r in %.2fs (seed=%d, steps=%d)", prompt, elapsed, seed, steps)

        return GenerationResult(image=image, prompt=prompt, seed=seed, seconds=elapsed)
