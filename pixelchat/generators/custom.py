"""Our own model, trained from scratch. Not ready yet (Phases 3-4)."""

from pixelchat.generators.base import GenerationResult, ImageGenerator


class PixelChatGenerator(ImageGenerator):
    """Placeholder for our own diffusion model."""

    name = "pixelchat"

    def generate(self, prompt: str, seed: int | None = None, steps: int | None = None) -> GenerationResult:
        raise NotImplementedError("Our own model isn't trained yet: see Phases 3-4")
