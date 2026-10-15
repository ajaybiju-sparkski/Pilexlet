"""Turns chat messages into prompts: decides new-prompt vs. edit-of-previous."""

EDIT_PREFIXES = ("make it", "now", "add", "change", "but", "more", "less", "bigger", "smaller")

# A lone color or style word, with no other context, is treated as an edit.
LONE_EDIT_WORDS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "gold", "silver",
    "cartoon", "realistic", "sketch", "watercolor", "oil painting", "anime",
    "cyberpunk", "vintage", "minimalist", "3d", "pixel art",
}

HELP_TEXT = (
    "Type a prompt to generate an image (e.g. \"a cat on a sofa\").\n"
    "Follow-ups like \"make it blue\" or \"now at night\" edit the previous prompt.\n"
    "/new - start a new conversation\n"
    "/help - show this message"
)


def is_edit(message: str, has_previous_prompt: bool) -> bool:
    """Decide whether `message` edits the previous prompt or starts a new one."""
    if not has_previous_prompt:
        return False

    text = message.strip().lower()
    if not text:
        return False

    if text in LONE_EDIT_WORDS:
        return True

    return text.startswith(EDIT_PREFIXES)


def combine_prompts(previous_prompt: str, edit_message: str) -> str:
    """Combine the previous prompt with an edit message, e.g.
    "a cat on a sofa" + "make it orange" -> "a cat on a sofa, orange".
    """
    text = edit_message.strip()
    for prefix in EDIT_PREFIXES:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
            break

    text = text.lstrip(",").strip()
    if not text:
        text = edit_message.strip()

    return f"{previous_prompt}, {text}"


class ChatSession:
    """Tracks the last prompt and routes messages to new-prompt/edit/command handling."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def reset(self) -> None:
        self.last_prompt = None

    def handle(self, message: str) -> dict:
        """Process one user message.

        Returns a dict: {"type": "new" | "edit" | "reset" | "help" | "empty", "prompt": str | None}
        """
        text = message.strip()

        if not text:
            return {"type": "empty", "prompt": None}

        if text == "/new":
            self.reset()
            return {"type": "reset", "prompt": None}

        if text == "/help":
            return {"type": "help", "prompt": None}

        if is_edit(text, has_previous_prompt=self.last_prompt is not None):
            prompt = combine_prompts(self.last_prompt, text)
            self.last_prompt = prompt
            return {"type": "edit", "prompt": prompt}

        self.last_prompt = text
        return {"type": "new", "prompt": text}
