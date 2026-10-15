import pytest
from PIL import Image

from pixelchat.generators import get_generator
from pixelchat.generators.base import GenerationResult, ImageGenerator


class FakeGenerator(ImageGenerator):
    """Returns a solid-color image. Never touches the network or a real model."""

    name = "fake"

    def generate(self, prompt: str, seed: int | None = None, steps: int | None = None) -> GenerationResult:
        image = Image.new("RGB", (64, 64), color=(120, 60, 200))
        return GenerationResult(image=image, prompt=prompt, seed=seed or 0, seconds=0.01)


def test_fake_generator_result_shape():
    generator = FakeGenerator()
    result = generator.generate("a red apple", seed=42)

    assert isinstance(result, GenerationResult)
    assert isinstance(result.image, Image.Image)
    assert result.prompt == "a red apple"
    assert result.seed == 42
    assert result.seconds >= 0


def test_fake_generator_default_seed():
    generator = FakeGenerator()
    result = generator.generate("a cat")
    assert result.seed == 0


def test_get_generator_unknown_name_raises():
    with pytest.raises(ValueError):
        get_generator("not-a-real-generator")


def test_get_generator_pixelchat_placeholder_raises_not_implemented():
    generator = get_generator("pixelchat")
    with pytest.raises(NotImplementedError):
        generator.generate("anything")
