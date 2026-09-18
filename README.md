# Bonsai Chat: repetition controls

Bonsai Chat is a local Flask chat UI for
`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, using the model's bundled
`vision_artifact.load_vl_model` loader and `mlx_vlm.stream_generate`.
The app does not replace that loader with a generic model loader.

## Install, download, and run

Stop an old Flask process with Ctrl+C. Extract this project and replace the app
files, including the new `generation_controls.py`. Do not overwrite an existing
model folder.

Create and activate a virtual environment **before** installing this project's
requirements. Run the commands from the repository root:

```bash
cd bonsai-chat
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The activation command above is for macOS/Linux. In PowerShell on Windows, use
`.venv\Scripts\Activate.ps1` instead. Activate this environment again with
`source .venv/bin/activate` whenever opening a new macOS/Linux terminal.

### Download the Bonsai model with Hugging Face

The model is hosted on Hugging Face at
`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`. The `hf` command is provided by the
`huggingface_hub` package, so install it in the active virtual environment:

```bash
python -m pip install --upgrade huggingface_hub
hf --help
```

Choose a directory for the complete model pack, then download it there. The
model is about 8.6 GB on disk, so ensure the destination has adequate free
space. This example keeps models outside the app repository:

```bash
MODEL_DIR="$HOME/projects/ternary-bonsai-2/bonsai2-27b-mlx"
mkdir -p "$(dirname "$MODEL_DIR")"
hf download prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --local-dir "$MODEL_DIR"
```

`hf download` downloads the entire Hugging Face model repository, including the
bundled `runtime/` loader required by this app. If Hugging Face prompts for
access, create or sign in to a Hugging Face account, create a read token, and
run `hf auth login` before retrying the download.

Install the model pack's runtime dependencies into the same active virtual
environment, then start the app while pointing it at the directory chosen
above:

```bash
python -m pip install -r "$MODEL_DIR/runtime/requirements.txt"
BONSAI_MODEL_PATH="$MODEL_DIR" python app.py
```

The project's `requirements.txt` contains the app's UI/formatting dependencies.
The model pack's `runtime/requirements.txt` supplies its MLX runtime
dependencies; do not replace it with a generic model loader or blindly upgrade
the MLX stack.

By default, the app looks for the model directory at:

```text
$HOME/projects/ternary-bonsai-2/bonsai2-27b-mlx
```

Set `BONSAI_MODEL_PATH` if you keep the model somewhere else. If the model is
already installed at that default location on your macOS/Linux machine,
activation, app dependency installation, and startup are enough:

```bash
cd bonsai-chat
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

An alternative model path or port can be supplied explicitly:

```bash
BONSAI_MODEL_PATH="/path/to/bonsai2-27b-mlx" PORT=5050 python app.py
```

Open `http://127.0.0.1:5000` and refresh the page. The server starts immediately;
the model is loaded once in a loader thread. Wait for **Ready**.

No additional Python dependency was introduced for repetition controls. The
requirements file contains the existing UI/formatting dependencies only, not an
upgrade of your MLX, MLX-VLM, or Transformers installation.

Chat history still uses `bonsai-chat-v1` in browser localStorage. Keep the same
browser and origin (including hostname and port) to access existing chats.
Sampling settings use a new v2 key and initially select **Bonsai Instruct**; the
old optional system prompt is retained. Old settings are not deleted.

## Start here when output loops

First read the tokenizer correction below. Then start a **new chat**, select
**Bonsai Instruct** in the gear menu, and keep **Stop repeated text automatically**
enabled. Retry the original request without including the old repeated output.

Profiles:

| Setting | Bonsai Instruct | Bonsai Thinking | Code: mild penalty (trial) |
|---|---:|---:|---:|
| Temperature | 0.7 | 1.0 | 0.7 |
| Top-p | 0.80 | 0.95 | 0.80 |
| Top-k | 20 | 20 | 20 |
| Min-p | 0.0 | 0.0 | 0.0 |
| Presence penalty | 1.5 | 0.0 | 0.0 |
| Repetition penalty | 1.00 | 1.00 | 1.05 |
| Frequency penalty | 0.0 | 0.0 | 0.0 |
| Enable thinking in template | Off | On | Off |

The first two profiles use Prism's published sampling values. The **Code**
profile is an experimental app choice, not a manufacturer recommendation or a
quality guarantee. It uses mild repetition penalty instead of presence penalty.
For a controlled test, try 1.05, then 1.10, without simultaneously increasing
all penalties. Necessary code tokens and identifiers repeat too.

The default output cap is 2,048 new tokens. Increase it to 4,096 for longer code
when needed; this increases the output budget, not the model's knowledge or
correctness. The UI marks `length` termination as possibly incomplete.

The 256-token **Penalty window** is an app choice applied to enabled repetition,
presence, and frequency penalties. It is NOT the full chat context window.
Prism's published sampling recommendations do not specify this window.

Temperature zero selects greedy decoding in MLX-VLM. It is useful for some
reproducible comparisons, but it is not a general repetition fix.

## Automatic loop guard

`generation_controls.RepetitionGuard` looks for at least three consecutive exact
copies of a substantial text fragment, using a bounded recent-text window.
A repeated unit must be 80-2,048 characters, contain enough alphanumeric content,
and not simply be repetitions of a short-period sequence such as braces or
`std::`. Checks occur after approximately 32 additional characters.

The guard stops generation, closes the stream, displays a persistent warning,
and sets `finish_reason` to `repetition_guard`. It does **not** silently delete
text, repair source code, regenerate automatically, or claim a complete answer.
Guard-stopped replies remain visible/copyable, but the UI excludes them from
subsequent model prompts. Replies stopped manually or interrupted by an error
are also excluded. Older replies generated before this release are not
retroactively classified; use **New chat** for those.

This is a heuristic. It can miss near-repetitions and very short/long repeated
units. Deliberately repeated long code/data blocks can trigger it, so the guard
can be disabled. It is not a substitute for evaluating the model's answers.

## Stop and generation status

The Stop button sends `/api/stop` with the active request ID. The server checks a
cancellation event between generation steps. It cannot interrupt an MLX native
kernel or prompt-prefill call already in progress, so cancellation may take
until that work returns. A client disconnect also closes the generator once
control returns to the application.

The app rejects a second simultaneous generation with HTTP 409 rather than
queuing multiple requests behind the same model. The processor and model remain
serialized. The Flask reloader stays disabled to avoid loading the model twice.

## Runtime compatibility

At load time the app inspects the installed `mlx_vlm.generate.generate_step`
signature. `/api/status` and the settings panel report the MLX-VLM version and
available sampling controls. A generic `**kwargs` is not treated as proof that a
control works. Unsupported non-neutral values fail explicitly instead of being
silently ignored.

Some older builds do not expose presence/frequency penalties. If your runtime
reports that, select **Code** (presence/frequency zero) if repetition penalty is
available, or disable unsupported penalties. Do not blindly upgrade the whole
MLX stack just to dismiss an error; keep a runtime compatible with your model
pack. `generation_controls.py` has no MLX-specific imports.

## Important correction: tokenizer regex warning

A previous README recommended adding `fix_mistral_regex=True` to the bundled
loader. That recommendation is withdrawn for this model. Prism's own Bonsai 2
MLX demo explains that the Mistral warning does not apply here and that the pack
uses the base model's tokenizer.

If you added that flag solely because of the earlier advice, remove the added
argument from `runtime/vision_artifact.py`, preserving the rest of the loader.
For the original call shown in that advice, restore:

```python
tokenizer = AutoTokenizer.from_pretrained(str(directory))
```

Restart Flask so the tokenizer is reloaded. Do not change the model type, edit
`tokenizer.json`, or otherwise substitute the tokenizer just to suppress a
warning. This app does not modify files in your model directory. The earlier
flag has not been established as the cause of your particular repetition loop;
restoring the intended loader is a controlled troubleshooting step.

## Markdown, math, and source listings

Existing response-level Copy, per-code-block Copy, server-side Markdown, LaTeX /
MathML, and Pygments highlighting are retained. C++, Python, and shell aliases
include `cpp`, `c++`, `python`, `py`, `bash`, `sh`, `shell`, and `zsh`.

Raw text streams immediately and is formatted after completion. A stopped or
truncated fenced code block can still be displayed as code; that does not mean
the implementation is complete or valid. Response Copy preserves the raw model
response. Code Copy copies that listing's source only. Generated listings are
not executed by this app.

## Tests and limitations

Run the dependency-free logic tests with:

```bash
python -m unittest discover -s tests -v
```

There are 28 tests covering parameter validation, unsupported controls,
repetition detection, the reported ODE45 comment loop, normal repeated code
tokens, stream cleanup, cancellation, finish reasons, and request ownership.
The runtime tests execute the actual `BonsaiRuntime` class extracted from the
app source with a fake token stream. They do not import or run MLX/Flask.

Python compilation and JavaScript syntax checks also passed. Additional
Chromium checks used in-memory mocked API responses to test profiles, outgoing
parameters, loop notices, context exclusion, stop requests, storage writes, and
mobile layout. These are not live Flask or model-inference tests. This build
has not been tested with your model weights on Apple Silicon.

## Local use and privacy

By default the server binds only to `127.0.0.1`. This is a single-user local app,
not an authenticated service. Model inference and rendering run locally; chat
history is stored unencrypted in browser storage. The app does not serve a CDN
script for formatting. Model/runtime libraries may access their usual download
services if your local model files are incomplete.

Do not expose the Flask development server directly to the public internet.

## Primary references

Sampling recommendations:
https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit

Prism's MLX demo, including the tokenizer warning note:
https://github.com/PrismML-Eng/Bonsai-demo/blob/main/scripts/mlx_generate_bonsai2.py

MLX-VLM's native generation controls:
https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/generate/ar.py

Sampling / penalty definitions:
https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py

These references were consulted for this update; installed package versions may
differ, which is why the app checks the actual installed function signature.
