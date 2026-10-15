"""All tunable settings live here. No magic numbers scattered around."""

# Which generator backend to use. Switching models later is just this value:
# "sd-turbo" (now) -> "pixelchat" (our own model, Phase 3+).
GENERATOR = "sd-turbo"

# SD-Turbo settings
MODEL_ID = "stabilityai/sd-turbo"
DEFAULT_STEPS = 2
MAX_STEPS = 4
MIN_STEPS = 1
IMAGE_SIZE = 512
GUIDANCE_SCALE = 0.0  # SD-Turbo is trained without classifier-free guidance

# Output
OUTPUT_DIR = "outputs"

# Server
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860
