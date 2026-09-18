from __future__ import annotations

import html as html_lib
import json
from importlib.metadata import version, PackageNotFoundError
import logging
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import bleach
import markdown
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from latex2mathml import converter as latex_converter
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound
from generation_controls import GenerationOptions, RepetitionGuard, explicit_parameters, sampling_kwargs
from image_inputs import decode_message_images
from model_paths import resolve_model_path

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bonsai-chat")

# BONSAI_MODEL_PATH, then the installer's choice, then ~/models/<model folder>.
MODEL_PATH, MODEL_PATH_SOURCE = resolve_model_path()

app = Flask(__name__)
# Attached images travel as base64 inside the JSON body.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


class BonsaiRuntime:
    """Loads the Prism runtime once and serializes MLX generation requests."""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.state = "idle"
        self.error: str | None = None

        self.model: Any = None
        self.processor: Any = None
        self.config: Any = None
        self.chat_cfg: Any = None

        self._stream_generate = None
        self._apply_chat_template = None
        self._get_message_json = None
        self._get_chat_template = None

        self._state_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._load_thread: threading.Thread | None = None
        self.loaded_at: float | None = None
        self.sampling_supported: set[str] = set()
        self.mlx_vlm_version: str | None = None
        self._active_request_id: str | None = None
        self._stop_event: threading.Event | None = None

    def start_loading(self) -> None:
        with self._state_lock:
            if self.state in {"loading", "ready"}:
                return
            self.state = "loading"
            self.error = None
            self._load_thread = threading.Thread(
                target=self._load,
                name="bonsai-model-loader",
                daemon=True,
            )
            self._load_thread.start()

    def _load(self) -> None:
        started = time.perf_counter()
        try:
            if not self.model_path.is_dir():
                raise FileNotFoundError(
                    f"Model directory does not exist: {self.model_path}. "
                    "Run ./install.sh, or set BONSAI_MODEL_PATH."
                )

            runtime_dir = self.model_path / "runtime"
            if not runtime_dir.is_dir():
                raise FileNotFoundError(
                    f"Bundled Prism runtime not found: {runtime_dir}"
                )

            # The Prism pack requires its bundled loader. Do not replace this
            # with mlx_lm.load() or generic mlx_vlm.load().
            runtime_str = str(runtime_dir)
            if runtime_str not in sys.path:
                sys.path.insert(0, runtime_str)

            from vision_artifact import chat_config, load_vl_model
            from mlx_vlm import stream_generate
            from mlx_vlm.generate import generate_step

            self.sampling_supported = explicit_parameters(generate_step)
            try:
                self.mlx_vlm_version = version("mlx-vlm")
            except PackageNotFoundError:
                self.mlx_vlm_version = "unknown"
            from mlx_vlm.prompt_utils import apply_chat_template, get_chat_template, get_message_json

            log.info("Loading Bonsai model from %s", self.model_path)
            model, processor, config = load_vl_model(self.model_path)

            self.model = model
            self.processor = processor
            self.config = config
            self.chat_cfg = chat_config(config)
            self._stream_generate = stream_generate
            self._apply_chat_template = apply_chat_template
            self._get_message_json = get_message_json
            self._get_chat_template = get_chat_template

            with self._state_lock:
                self.state = "ready"
                self.loaded_at = time.time()

            log.info(
                "Model ready in %.2f seconds",
                time.perf_counter() - started,
            )
        except Exception as exc:
            log.error("Model loading failed:\n%s", traceback.format_exc())
            with self._state_lock:
                self.state = "error"
                self.error = f"{type(exc).__name__}: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "error": self.error,
            "model_path": str(self.model_path),
            "model_name": self.model_path.name,
            "vision": bool((self.config or {}).get("components", {}).get("vision")),
            "loaded_at": self.loaded_at,
            "sampling_supported": sorted(self.sampling_supported),
            "mlx_vlm_version": self.mlx_vlm_version,
            "busy": self._active_request_id is not None,
        }

    def begin_request(self, request_id: str) -> threading.Event:
        with self._state_lock:
            if self._active_request_id is not None:
                raise RuntimeError("Another response is still being generated. Stop it first.")
            self._active_request_id = request_id
            self._stop_event = threading.Event()
            return self._stop_event

    def stop_request(self, request_id: str) -> bool:
        with self._state_lock:
            if self._active_request_id != request_id or self._stop_event is None:
                return False
            self._stop_event.set()
            return True

    def finish_request(self, request_id: str) -> None:
        with self._state_lock:
            if self._active_request_id == request_id:
                self._active_request_id = None
                self._stop_event = None

    def _build_prompt(
        self, messages: list[dict[str, str]], images: list[list[Any]], enable_thinking: bool,
    ) -> Any:
        if not any(images):
            return self._apply_chat_template(
                self.processor, self.chat_cfg, messages,
                num_images=0,
                enable_thinking=enable_thinking,
            )
        # apply_chat_template() puts every image token on the last user
        # message. Format each turn separately so an image stays in the turn
        # it was sent with, in the same order as the flat image list.
        assert self._get_message_json is not None
        assert self._get_chat_template is not None
        model_type = self.chat_cfg["model_type"]
        formatted = [
            self._get_message_json(
                model_type, message["content"], message["role"],
                skip_image_token=not group, num_images=len(group),
                enable_thinking=enable_thinking,
            )
            for message, group in zip(messages, images)
        ]
        return self._get_chat_template(
            self.processor, formatted, True, enable_thinking=enable_thinking,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        options: GenerationOptions,
        request_id: str,
        stop_event: threading.Event,
        images: list[list[Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """'images' holds one list of PIL images per entry in 'messages'."""
        if self.state != "ready":
            raise RuntimeError(f"Model is not ready (state={self.state})")
        assert self._apply_chat_template is not None
        assert self._stream_generate is not None
        kwargs = sampling_kwargs(options, self.sampling_supported)
        images = images if images is not None else [[] for _ in messages]
        if len(images) != len(messages):
            raise ValueError("'images' must have one entry per message")
        flat_images = [image for group in images for image in group]
        if flat_images:
            kwargs["image"] = flat_images

        # Keep templating and generation together: the processor is shared.
        with self._generation_lock:
            started = time.perf_counter()
            last = None
            stream = None
            generated_text = ""
            guard = RepetitionGuard() if options.loop_guard else None
            reason = None
            notice = None
            try:
                if not stop_event.is_set():
                    prompt = self._build_prompt(messages, images, options.enable_thinking)
                    stream = self._stream_generate(
                        self.model, self.processor, prompt=prompt, **kwargs,
                    )
                    while not stop_event.is_set():
                        try:
                            chunk = next(stream)
                        except StopIteration:
                            break
                        last = chunk
                        # A stop cannot interrupt a native MLX kernel/prefill.
                        # Check again as soon as that work returns.
                        if stop_event.is_set():
                            break
                        if getattr(chunk, "is_draft", False):
                            continue
                        delta = getattr(chunk, "text", "") or ""
                        if delta:
                            generated_text += delta
                            yield {"type": "delta", "text": delta}
                        match = guard.feed(delta) if guard else None
                        if match is not None:
                            reason = "repetition_guard"
                            notice = (
                                "Stopped: a substantial text fragment repeated at least "
                                "three times consecutively. This response is incomplete and "
                                "will be excluded from subsequent prompts. Start a new chat "
                                "or retry the last request with different settings."
                            )
                            break
                    if reason is None and guard and guard.feed("", force=True):
                        reason = "repetition_guard"
                        notice = "Repeated text detected. This incomplete response is excluded from future prompts."
                if stop_event.is_set():
                    reason = "user_stop"
                    notice = "Stopped by you. This incomplete response is excluded from future prompts."
            finally:
                # Closing the generator releases its cache/context managers even
                # when the client disconnects or the repetition guard fires.
                if stream is not None:
                    close = getattr(stream, "close", None)
                    if close is not None:
                        close()

            tokens = getattr(last, "generation_tokens", None) if last else None
            reason = reason or (getattr(last, "finish_reason", None) if last else None)
            if not reason:
                reason = "length" if tokens is not None and tokens >= options.max_tokens else "stop"
            if reason == "length":
                notice = "Token limit reached. The response may be incomplete; it is not a finished implementation."

            def metric(name: str, digits: int) -> float | None:
                value = getattr(last, name, None) if last else None
                return round(float(value), digits) if value is not None else None

            stats = {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "generation_tokens": tokens,
                "prompt_tokens": getattr(last, "prompt_tokens", None) if last else None,
                "generation_tps": metric("generation_tps", 2),
                "prompt_tps": metric("prompt_tps", 2),
                "peak_memory_gb": metric("peak_memory", 3),
                "finish_reason": reason,
                "characters": len(generated_text),
                "images": len(flat_images),
            }
            yield {
                "type": "done", "stats": stats, "notice": notice,
                "exclude_from_context": reason in {"repetition_guard", "user_stop"},
                "settings": options.to_dict(),
            }


runtime = BonsaiRuntime(MODEL_PATH)


# Markdown is sanitized before generated MathML is inserted. That means model
# output cannot inject arbitrary HTML/JavaScript, while math generated by the
# trusted converter remains intact.
_ALLOWED_MARKDOWN_TAGS = {
    "a", "blockquote", "br", "code", "del", "em", "h1", "h2", "h3", "h4",
    "h5", "h6", "hr", "li", "ol", "p", "pre", "strong", "table", "tbody",
    "td", "th", "thead", "tr", "ul",
}
_ALLOWED_MARKDOWN_ATTRIBUTES = {
    "a": ["href", "title"],
    "code": ["class"],
}

_CODE_FORMATTER = HtmlFormatter(
    nowrap=True,
    noclasses=True,
    style="monokai",
)

_CODE_LANGUAGE_ALIASES = {
    # C and C++
    "c++": ("cpp", "C++"),
    "cpp": ("cpp", "C++"),
    "cxx": ("cpp", "C++"),
    "cc": ("cpp", "C++"),
    "hpp": ("cpp", "C++"),
    "hxx": ("cpp", "C++"),
    "c": ("c", "C"),
    "h": ("c", "C"),

    # Python
    "python": ("python", "Python"),
    "python3": ("python", "Python"),
    "py": ("python", "Python"),

    # Shells
    "bash": ("bash", "Bash"),
    "shell": ("bash", "Shell"),
    "shellscript": ("bash", "Shell"),
    "shell-script": ("bash", "Shell"),
    "sh": ("bash", "Shell"),
    "zsh": ("bash", "Zsh"),

    # Useful common fallbacks
    "json": ("json", "JSON"),
    "yaml": ("yaml", "YAML"),
    "yml": ("yaml", "YAML"),
    "toml": ("toml", "TOML"),
    "javascript": ("javascript", "JavaScript"),
    "js": ("javascript", "JavaScript"),
    "typescript": ("typescript", "TypeScript"),
    "ts": ("typescript", "TypeScript"),
    "html": ("html", "HTML"),
    "css": ("css", "CSS"),
    "sql": ("sql", "SQL"),
    "text": ("text", "Plain text"),
    "plaintext": ("text", "Plain text"),
    "txt": ("text", "Plain text"),
}



def _normalize_fence_markers(text: str) -> str:
    r"""Normalize escaped Markdown code-fence markers emitted by some models.

    Examples accepted:
        \`\`\`cpp
        \```python
        ```bash
    """
    lines: list[str] = []
    for line in text.splitlines():
        # Backslash before every backtick: \`\`\`cpp
        line = re.sub(r"^(\s*)(?:\\`){3}", r"\1```", line)
        # One backslash before a normal fence: \```cpp
        line = re.sub(r"^(\s*)\\(`{3,}|~{3,})", r"\1\2", line)
        lines.append(line)
    return "\n".join(lines)


def _canonical_code_language(info: str) -> tuple[str, str]:
    raw = (info or "").strip()

    # Markdown fence info strings can contain attributes after the language.
    # Only the first token is used for highlighting.
    language = raw.split()[0].strip("{}").lower() if raw else ""
    language = language.removeprefix("language-")

    if language in _CODE_LANGUAGE_ALIASES:
        return _CODE_LANGUAGE_ALIASES[language]

    if not language:
        return "text", "Plain text"

    # Pygments supports many additional languages. Keep them working without
    # making the app depend on a fixed allow-list.
    try:
        lexer = get_lexer_by_name(language)
        display = lexer.name or language
        return language, display
    except ClassNotFound:
        return "text", raw or "Plain text"


def _extract_fenced_code(text: str) -> tuple[str, list[dict[str, str]]]:
    """Extract fenced code blocks before Markdown/LaTeX processing.

    Supports backtick and tilde fences. An unterminated final fence is also
    treated as code until end-of-response, which is useful for truncated local
    model generations.
    """
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    blocks: list[dict[str, str]] = []
    index = 0

    opener = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\r\n]*)\r?\n?$")

    while index < len(lines):
        match = opener.match(lines[index])
        if not match:
            output.append(lines[index])
            index += 1
            continue

        fence = match.group(1)
        info = match.group(2).strip()
        fence_char = fence[0]
        fence_len = len(fence)

        closing = re.compile(
            rf"^[ \t]*{re.escape(fence_char)}{{{fence_len},}}[ \t]*\r?\n?$"
        )

        code_lines: list[str] = []
        index += 1
        while index < len(lines) and not closing.match(lines[index]):
            code_lines.append(lines[index])
            index += 1

        if index < len(lines) and closing.match(lines[index]):
            index += 1

        token = f"BNSCODEBLOCK{len(blocks):05d}XQZ"
        lexer_name, display_name = _canonical_code_language(info)

        blocks.append(
            {
                "token": token,
                "code": "".join(code_lines),
                "lexer": lexer_name,
                "language": display_name,
            }
        )
        output.append(f"\n\n{token}\n\n")

    return "".join(output), blocks


def _render_code_block(item: dict[str, str]) -> str:
    source = item["code"]
    lexer_name = item["lexer"]

    try:
        lexer = (
            TextLexer()
            if lexer_name == "text"
            else get_lexer_by_name(lexer_name)
        )
    except ClassNotFound:
        lexer = TextLexer()

    highlighted = highlight(source, lexer, _CODE_FORMATTER)
    language = html_lib.escape(item["language"] or "Plain text")

    return (
        '<div class="code-listing">'
        '<div class="code-toolbar">'
        f'<span class="code-language">{language}</span>'
        '<button type="button" class="copy-code" '
        'title="Copy this code block">⧉ Copy</button>'
        '</div>'
        f'<pre><code>{highlighted}</code></pre>'
        '</div>'
    )

def _normalize_model_markdown(text: str) -> str:
    r"""Fix common over-escaped Markdown at the beginning of output lines.

    Some local models emit things such as ``\### Heading`` or ``\- item``.
    Those escapes are useful inside Markdown source, but they prevent the
    browser UI from displaying the intended heading/list/table.
    """
    lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^(\s*)\\(#{1,6}\s+)", r"\1\2", line)
        line = re.sub(r"^(\s*)\\([-+*]\s+)", r"\1\2", line)
        line = re.sub(r"^(\s*)\\(>\s*)", r"\1\2", line)

        stripped = line.strip()
        if stripped in {r"\---", r"\***", r"\___"}:
            prefix = line[: len(line) - len(line.lstrip())]
            line = prefix + stripped[1:]
        elif stripped.startswith(r"\|") and stripped.endswith(r"\|"):
            # Only unescape pipes on lines that clearly look like Markdown
            # table rows. This avoids touching LaTeX norm notation elsewhere.
            line = line.replace(r"\|", "|")

        lines.append(line)
    return "\n".join(lines)


def _protect_code(text: str) -> tuple[str, list[str]]:
    """Temporarily remove fenced and inline code so math parsing ignores it."""
    code_parts: list[str] = []
    pattern = re.compile(r"```[\s\S]*?```|`[^`\n]*`")

    def repl(match: re.Match[str]) -> str:
        token = f"BNSCODETOKEN{len(code_parts):05d}XQZ"
        code_parts.append(match.group(0))
        return token

    return pattern.sub(repl, text), code_parts


def _restore_code(text: str, code_parts: list[str]) -> str:
    for index, source in enumerate(code_parts):
        text = text.replace(f"BNSCODETOKEN{index:05d}XQZ", source)
    return text


def _extract_math(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Replace common TeX delimiters with stable Markdown-safe tokens."""
    items: list[dict[str, Any]] = []

    def store(latex: str, display: bool, original: str) -> str:
        token = f"BNSMATH{'BLOCK' if display else 'INLINE'}{len(items):05d}XQZ"
        items.append(
            {
                "token": token,
                "latex": latex.strip(),
                "display": display,
                "original": original,
            }
        )
        return f"\n\n{token}\n\n" if display else token

    # Display math first.
    block_dollar = re.compile(r"(?<!\\)\$\$([\s\S]+?)(?<!\\)\$\$")
    text = block_dollar.sub(
        lambda m: store(m.group(1), True, m.group(0)),
        text,
    )

    block_bracket = re.compile(r"\\\[([\s\S]+?)\\\]")
    text = block_bracket.sub(
        lambda m: store(m.group(1), True, m.group(0)),
        text,
    )

    # Inline \( ... \)
    inline_paren = re.compile(r"\\\(([^\n]+?)\\\)")
    text = inline_paren.sub(
        lambda m: store(m.group(1), False, m.group(0)),
        text,
    )

    # Inline $...$. Require non-whitespace next to both delimiters so ordinary
    # currency such as "$1.10 ... $1.00" is not accidentally treated as math.
    inline_dollar = re.compile(
        r"(?<!\\)\$(?!\$)(?=\S)([^\n$]*?\S)(?<!\\)\$(?!\$)"
    )
    text = inline_dollar.sub(
        lambda m: store(m.group(1), False, m.group(0)),
        text,
    )

    return text, items


def _mathml(item: dict[str, Any]) -> str:
    try:
        converted = latex_converter.convert(item["latex"])
        if item["display"]:
            if "<math " in converted:
                converted = converted.replace("<math ", '<math display="block" ', 1)
            else:
                converted = converted.replace("<math>", '<math display="block">', 1)
            return f'<div class="math-block">{converted}</div>'
        return f'<span class="math-inline">{converted}</span>'
    except Exception:
        log.warning("Could not render LaTeX: %r", item["latex"], exc_info=True)
        fallback = html_lib.escape(item["original"])
        cls = "math-block-fallback" if item["display"] else "math-inline-fallback"
        return f'<code class="{cls}">{fallback}</code>'


def render_assistant_html(text: str) -> str:
    """Render Markdown, fenced source listings, and LaTeX safely."""
    normalized = _normalize_fence_markers(text)
    protected, code_blocks = _extract_fenced_code(normalized)

    # Inline code must also be hidden from the LaTeX parser.
    protected, inline_code_parts = _protect_code(protected)
    protected = _normalize_model_markdown(protected)
    protected, math_items = _extract_math(protected)
    protected = _restore_code(protected, inline_code_parts)

    rendered = markdown.markdown(
        protected,
        extensions=[
            "tables",
            "sane_lists",
            "nl2br",
        ],
        output_format="html5",
    )

    rendered = bleach.clean(
        rendered,
        tags=_ALLOWED_MARKDOWN_TAGS,
        attributes=_ALLOWED_MARKDOWN_ATTRIBUTES,
        protocols={"http", "https", "mailto"},
        strip=True,
    )

    # Insert trusted, server-generated syntax-highlighted listings after
    # sanitizing the model-generated Markdown.
    for item in code_blocks:
        token = item["token"]
        code_html = _render_code_block(item)
        rendered = re.sub(
            rf"<p>\s*{re.escape(token)}\s*</p>",
            lambda _match: code_html,
            rendered,
        )
        rendered = rendered.replace(token, code_html)

    # Insert MathML only after sanitizing Markdown. The MathML comes from the
    # local trusted converter, not from raw model-generated HTML.
    for item in math_items:
        token = item["token"]
        math_html = _mathml(item)
        if item["display"]:
            rendered = re.sub(
                rf"<p>\s*{re.escape(token)}\s*</p>",
                lambda _match: math_html,
                rendered,
            )
        rendered = rendered.replace(token, math_html)

    return rendered


def _clean_messages(
    raw_messages: Any, system_prompt: str = "",
) -> tuple[list[dict[str, str]], list[list[Any]]]:
    """Return the prompt messages and, aligned with them, each message's images."""
    if not isinstance(raw_messages, list):
        raise ValueError("'messages' must be an array")

    messages: list[dict[str, str]] = []
    images: list[list[Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
        images.append([])

    for item in raw_messages[-100:]:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "assistant" and item.get("exclude_from_context") is True:
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant", "system"}:
            continue
        if not isinstance(content, str):
            continue
        attached = []
        if role == "user":
            attached = decode_message_images(item.get("images"), sum(map(len, images)))
        if not content.strip() and not attached:
            continue
        messages.append({"role": role, "content": content})
        images.append(attached)

    if not messages or messages[-1]["role"] != "user":
        raise ValueError("The final non-empty message must be from the user")

    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars > 500_000:
        raise ValueError("Conversation is too large; start a new chat or trim older messages")

    return messages, images


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Request is too large. Remove some images or start a new chat."}), 413


@app.get("/")
def index():
    return render_template("index.html", model_name=MODEL_PATH.name)


@app.get("/api/status")
def api_status():
    return jsonify(runtime.status())




@app.post("/api/render")
def api_render():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    text = data.get("text", "")
    if not isinstance(text, str):
        return jsonify({"error": "'text' must be a string"}), 400
    if len(text) > 500_000:
        return jsonify({"error": "Response is too large to render"}), 413

    try:
        return jsonify({"html": render_assistant_html(text)})
    except Exception as exc:
        log.error("Response rendering failed:\n%s", traceback.format_exc())
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.post("/api/reload")
def api_reload():
    if runtime.state == "loading":
        return jsonify(runtime.status()), 409
    runtime.start_loading()
    return jsonify(runtime.status()), 202


@app.post("/api/stop")
def api_stop():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("request_id"), str):
        return jsonify({"error": "'request_id' must be a string"}), 400
    return jsonify({"stop_requested": runtime.stop_request(data["request_id"])})


@app.post("/api/chat")
def api_chat():
    if runtime.state != "ready":
        return jsonify(runtime.status()), 503
    data = request.get_json(silent=True)
    try:
        options = GenerationOptions.from_request(data)
        system_prompt = data.get("system_prompt", "")
        if not isinstance(system_prompt, str):
            raise ValueError("'system_prompt' must be a string")
        messages, images = _clean_messages(data.get("messages"), system_prompt)
        request_id = data.get("request_id")
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", request_id):
            raise ValueError("A unique 'request_id' of 8-80 letters, numbers, '-' or '_' is required")
        sampling_kwargs(options, runtime.sampling_supported)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        stop_event = runtime.begin_request(request_id)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409

    @stream_with_context
    def event_stream():
        generator = runtime.generate(
            messages, options=options, request_id=request_id, stop_event=stop_event,
            images=images,
        )
        try:
            for event in generator:
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except GeneratorExit:
            log.info("Client disconnected from generation stream")
            return
        except Exception as exc:
            log.error("Generation failed:\n%s", traceback.format_exc())
            yield json.dumps(
                {"type": "error", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ) + "\n"
        finally:
            try:
                generator.close()
            finally:
                runtime.finish_request(request_id)

    response = Response(
        event_stream(), mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
    response.call_on_close(lambda: runtime.finish_request(request_id))
    return response


if __name__ == "__main__":
    log.info("Model path (%s): %s", MODEL_PATH_SOURCE, MODEL_PATH)
    runtime.start_loading()

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))

    # use_reloader=False is important: Flask's reloader would load the 27B model twice.
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
