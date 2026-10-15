"""Picks the best available torch device and matching dtype."""

import logging

import torch

logger = logging.getLogger(__name__)


def get_device_and_dtype() -> tuple[str, torch.dtype]:
    """Return (device, dtype) for this machine.

    Prefers Apple's MPS backend (fp16), falls back to CPU (fp32) when MPS
    isn't available.
    """
    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    logger.debug("Selected device=%s dtype=%s", device, dtype)
    return device, dtype


if __name__ == "__main__":
    device, dtype = get_device_and_dtype()
    print(device)
