"""PixelChat: a local chatbot that turns text into images. Entry point."""

import logging
import os
from datetime import datetime

import gradio as gr

from pixelchat import config
from pixelchat.chat import HELP_TEXT, ChatSession
from pixelchat.generators import get_generator

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

generator = get_generator(config.GENERATOR)

THEME_CSS_PATH = os.path.join(os.path.dirname(__file__), "pixelchat", "theme.css")

EXAMPLE_PROMPTS = [
    ("🌆 Neon alley", "a rain-soaked cyberpunk alley at night, neon reflections"),
    ("🍳 Rustic breakfast", "a rustic breakfast plate with eggs and avocado, soft morning light"),
    ("🐉 Fantasy creature", "a majestic dragon perched on a mountain peak at dawn"),
]


def _save_image(image, seed: int):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(config.OUTPUT_DIR, f"{timestamp}_{seed}.png")
    image.save(path)
    return path


def _friendly_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text or "mps" in text and "alloc" in text:
        return "Out of memory: try fewer steps."
    if generator._pipe is None:
        return "Model is still loading, hang on a second and try again."
    return "Something went wrong generating that image. Try again, maybe with fewer steps."


def _parse_seed(seed_box: str) -> int | None:
    """Empty/blank -> random seed. Otherwise must be a whole number."""
    text = (seed_box or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _image_message(path: str, prompt: str, seed: int, seconds: float) -> gr.ChatMessage:
    caption = f"{prompt} · seed {seed} · {seconds:.1f}s"
    return gr.ChatMessage(role="assistant", content=[{"path": path}, caption])


def load_model():
    """Triggered on page load. Loads the pipeline once (shared across sessions)."""
    try:
        if hasattr(generator, "_ensure_loaded"):
            generator._ensure_loaded()
        status = "✅ Model ready."
    except Exception:
        logger.exception("Failed to load model")
        status = "⚠️ Model failed to load. Check the server logs."
    return (
        gr.Markdown(status),
        gr.Textbox(interactive=True, placeholder="Ask me to create something…"),
        gr.Button(interactive=True),
        gr.Button(interactive=True),
    )


def _chat_outputs(history: list, session: ChatSession):
    return gr.Chatbot(value=history, visible=True), session, gr.Textbox(value=""), gr.Column(visible=False), gr.Row(visible=False)


def respond(message: str, history: list, session: ChatSession, steps: int, seed_box):
    history = history or []

    action = session.handle(message)

    if action["type"] == "empty":
        still_empty = len(history) == 0
        return (
            gr.Chatbot(value=history, visible=not still_empty),
            session,
            gr.Textbox(value=""),
            gr.Column(visible=still_empty),
            gr.Row(visible=still_empty),
        )

    history.append(gr.ChatMessage(role="user", content=message))

    if action["type"] == "help":
        history.append(gr.ChatMessage(role="assistant", content=HELP_TEXT))
        return _chat_outputs(history, session)

    if action["type"] == "reset":
        history.append(gr.ChatMessage(role="assistant", content="Conversation cleared. What should we make?"))
        return _chat_outputs(history, session)

    prompt = action["prompt"]
    seed = _parse_seed(seed_box)

    try:
        result = generator.generate(prompt, seed=seed, steps=int(steps))
        path = _save_image(result.image, result.seed)
        history.append(_image_message(path, result.prompt, result.seed, result.seconds))
    except Exception as exc:
        logger.exception("Generation failed")
        history.append(gr.ChatMessage(role="assistant", content=_friendly_error(exc)))

    return _chat_outputs(history, session)


def regenerate(history: list, session: ChatSession, steps: int):
    history = history or []

    if not session.last_prompt:
        history.append(gr.ChatMessage(role="assistant", content="Nothing to regenerate yet — send a prompt first."))
        return history, session

    prompt = session.last_prompt
    try:
        result = generator.generate(prompt, seed=None, steps=int(steps))
        path = _save_image(result.image, result.seed)
        history.append(_image_message(path, result.prompt, result.seed, result.seconds))
    except Exception as exc:
        logger.exception("Regeneration failed")
        history.append(gr.ChatMessage(role="assistant", content=_friendly_error(exc)))

    return history, session


def clear_chat():
    return gr.Chatbot(value=[], visible=False), ChatSession(), gr.Column(visible=True), gr.Row(visible=True)


with gr.Blocks(title="PixelChat") as demo:
    gr.HTML('<div id="pc-header"><h1>PixelChat</h1><p>Made by Ajay · Powered by SD-Turbo</p></div>')
    status = gr.Markdown(
        "⏳ Loading model… (first run downloads ~2.5GB from Hugging Face)",
        elem_id="pc-status",
    )

    session_state = gr.State(ChatSession())

    with gr.Column(elem_id="hero") as hero_col:
        gr.HTML('<div class="hero-orb"></div>')
        gr.HTML('<span class="hero-greeting">Hello</span><p class="hero-subtitle">What should we create today?</p>')

    chatbot = gr.Chatbot(elem_id="chatbot", height=480, show_label=False, visible=False)

    with gr.Row(elem_id="composer-row"):
        msg = gr.Textbox(
            placeholder="Model is loading…",
            interactive=False,
            scale=9,
            container=False,
            show_label=False,
        )
        send_btn = gr.Button("➤", elem_id="send-btn", interactive=False, scale=1, min_width=42)

    with gr.Row(elem_id="examples-row") as examples_row:
        example_btns = [
            gr.Button(f"{label}\n{prompt}", elem_classes=["example-card"]) for label, prompt in EXAMPLE_PROMPTS
        ]

    with gr.Accordion("Settings", open=False, elem_id="options-panel"):
        with gr.Row():
            steps_slider = gr.Slider(
                minimum=config.MIN_STEPS,
                maximum=config.MAX_STEPS,
                value=config.DEFAULT_STEPS,
                step=1,
                label="Steps",
            )
            seed_box = gr.Textbox(label="Seed (optional)", placeholder="random", value="")

        with gr.Row():
            regenerate_btn = gr.Button("🔁 Regenerate", interactive=False)
            clear_btn = gr.Button("🗑️ Clear")

    inputs = [msg, chatbot, session_state, steps_slider, seed_box]
    outputs = [chatbot, session_state, msg, hero_col, examples_row]

    msg.submit(respond, inputs, outputs)
    send_btn.click(respond, inputs, outputs)
    regenerate_btn.click(regenerate, [chatbot, session_state, steps_slider], [chatbot, session_state])
    clear_btn.click(clear_chat, None, [chatbot, session_state, hero_col, examples_row])

    for btn, (_, prompt) in zip(example_btns, EXAMPLE_PROMPTS):
        btn.click(lambda p=prompt: gr.Textbox(value=p), None, msg).then(respond, inputs, outputs)

    demo.load(load_model, None, [status, msg, send_btn, regenerate_btn])


if __name__ == "__main__":
    demo.queue().launch(
        server_name=config.SERVER_HOST,
        server_port=config.SERVER_PORT,
        css_paths=[THEME_CSS_PATH],
    )
