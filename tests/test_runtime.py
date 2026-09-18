"""Exercise the actual runtime class with a fake stream, without Flask/MLX.

Extract the class AST to avoid importing hardware and formatting dependencies.
These tests are NOT an end-to-end inference test on Apple Silicon.
"""
from __future__ import annotations

import ast
import logging
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation_controls import GenerationOptions, RepetitionGuard, sampling_kwargs
from test_controls import ODE_COMMENT

source = ast.parse((ROOT / "app.py").read_text())
cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "BonsaiRuntime")
namespace = dict(
    Path=Path, Any=Any, Iterator=Iterator, threading=threading, time=time,
    GenerationOptions=GenerationOptions, RepetitionGuard=RepetitionGuard,
    sampling_kwargs=sampling_kwargs, log=logging.getLogger("test"),
)
exec(compile(ast.Module(body=[cls], type_ignores=[]), "app.py", "exec"), namespace)
BonsaiRuntime = namespace["BonsaiRuntime"]


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = BonsaiRuntime(Path("/not-a-real-model"))
        self.runtime.state = "ready"
        self.runtime.sampling_supported = set(GenerationOptions().to_dict())
        self.prompt_kwargs = None
        self.stream_kwargs = None
        self.closed = False
        self.produced = 0
        self.source_text = "A coherent answer."
        self.finish_reason = "stop"

        def prompt(*args, **kwargs):
            self.prompt_kwargs = kwargs
            return "prompt"

        def stream(*args, **kwargs):
            self.stream_kwargs = kwargs
            try:
                for index, character in enumerate(self.source_text):
                    self.produced += 1
                    yield SimpleNamespace(text=character, generation_tokens=index + 1,
                                          prompt_tokens=10, finish_reason=None)
                yield SimpleNamespace(text="", generation_tokens=self.produced,
                                      prompt_tokens=10, finish_reason=self.finish_reason)
            finally:
                self.closed = True

        self.runtime._apply_chat_template = prompt
        self.runtime._stream_generate = stream

    def generate(self, **kwargs):
        options = GenerationOptions(**kwargs)
        event = self.runtime.begin_request("test-request-123")
        return self.runtime.generate(
            [{"role": "user", "content": "Hello"}], options=options,
            request_id="test-request-123", stop_event=event,
        )

    def test_native_controls_and_thinking_template(self):
        events = list(self.generate(enable_thinking=True, repetition_penalty=1.05))
        self.assertTrue(self.prompt_kwargs["enable_thinking"])
        self.assertEqual(self.stream_kwargs["repetition_penalty"], 1.05)
        self.assertEqual(self.stream_kwargs["top_k"], 20)
        self.assertEqual(events[-1]["stats"]["finish_reason"], "stop")
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "delta"), self.source_text)
        self.assertTrue(self.closed)

    def test_guard_closes_stream_and_marks_context_exclusion(self):
        self.source_text = "```cpp\n" + ODE_COMMENT * 20
        events = list(self.generate())
        self.assertEqual(events[-1]["stats"]["finish_reason"], "repetition_guard")
        self.assertTrue(events[-1]["exclude_from_context"])
        self.assertTrue(self.closed)
        self.assertLess(self.produced, len(self.source_text))

    def test_disabling_guard_preserves_output(self):
        self.source_text = ODE_COMMENT * 4
        events = list(self.generate(loop_guard=False))
        self.assertEqual(self.produced, len(self.source_text))
        self.assertEqual(events[-1]["stats"]["finish_reason"], "stop")

    def test_stop_before_first_token(self):
        generator = self.generate()
        self.assertTrue(self.runtime.stop_request("test-request-123"))
        events = list(generator)
        self.assertEqual(self.produced, 0)
        self.assertEqual(events[-1]["stats"]["finish_reason"], "user_stop")

    def test_stop_during_generation(self):
        generator = self.generate()
        self.assertEqual(next(generator)["type"], "delta")
        self.runtime.stop_request("test-request-123")
        rest = list(generator)
        self.assertEqual(rest[-1]["stats"]["finish_reason"], "user_stop")
        self.assertTrue(self.closed)

    def test_disconnect_closes_generator(self):
        generator = self.generate()
        next(generator)
        generator.close()
        self.assertTrue(self.closed)
        self.assertTrue(self.runtime._generation_lock.acquire(blocking=False))
        self.runtime._generation_lock.release()

    def test_busy_and_cancel_ownership(self):
        self.runtime.begin_request("one-request")
        with self.assertRaises(RuntimeError):
            self.runtime.begin_request("second-request")
        self.assertFalse(self.runtime.stop_request("someone-else"))
        self.runtime.finish_request("someone-else")
        self.assertTrue(self.runtime.status()["busy"])
        self.runtime.finish_request("one-request")
        self.assertFalse(self.runtime.status()["busy"])
        self.runtime.begin_request("second-request")
        # A late cleanup of an older stream cannot release the new request.
        self.runtime.finish_request("one-request")
        self.assertTrue(self.runtime.status()["busy"])

    def test_token_limit_is_visible(self):
        self.finish_reason = "length"
        events = list(self.generate())
        self.assertEqual(events[-1]["stats"]["finish_reason"], "length")
        self.assertIn("Token limit", events[-1]["notice"])


if __name__ == "__main__":
    unittest.main()
