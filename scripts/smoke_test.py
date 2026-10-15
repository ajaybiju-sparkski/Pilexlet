"""Generate one image and save it to outputs/. First run downloads SD-Turbo (~2.5GB)."""

import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pixelchat import config
from pixelchat.generators import get_generator

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    generator = get_generator(config.GENERATOR)
    result = generator.generate("a red apple on a wooden table")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(config.OUTPUT_DIR, f"{timestamp}_{result.seed}.png")
    result.image.save(out_path)

    print(f"Saved {out_path}")
    print(f"Prompt: {result.prompt}")
    print(f"Seed: {result.seed}")
    print(f"Time: {result.seconds:.2f}s")


if __name__ == "__main__":
    main()
