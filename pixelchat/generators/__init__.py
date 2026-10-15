"""Factory for ImageGenerator implementations."""

from pixelchat.generators.base import ImageGenerator


def get_generator(name: str) -> ImageGenerator:
    """Return an ImageGenerator instance for the given name.

    "sd-turbo" -> SDTurboGenerator (pretrained, used now)
    "pixelchat" -> PixelChatGenerator (our own model, not trained yet)
    """
    if name == "sd-turbo":
        from pixelchat.generators.pretrained import SDTurboGenerator

        return SDTurboGenerator()
    if name == "pixelchat":
        from pixelchat.generators.custom import PixelChatGenerator

        return PixelChatGenerator()
    raise ValueError(f"Unknown generator: {name!r}")
