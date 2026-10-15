"""The interface every generator implements. The app only talks to this."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass
class GenerationResult:
    """The result of a single image generation."""

    image: Image.Image
    prompt: str
    seed: int
    seconds: float


class ImageGenerator(ABC):
    """Common interface for all image backends (pretrained or our own)."""

    name: str

    @abstractmethod
    def generate(self, prompt: str, seed: int | None = None, steps: int | None = None) -> GenerationResult:
        """Generate one image for the given prompt."""
        raise NotImplementedError
