from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation_controls import GenerationOptions, RepetitionGuard, explicit_parameters, sampling_kwargs

ODE_COMMENT = (
    "// Compile-time feature detection for std::expected.\n"
    "// P1999R1 was adopted as C++23. Compile with -std=c++23.\n\n"
)


class ControlsTests(unittest.TestCase):
    def test_instruct_defaults(self):
        options = GenerationOptions.from_request({})
        self.assertEqual((options.temperature, options.top_p, options.top_k), (0.7, 0.8, 20))
        self.assertEqual((options.presence_penalty, options.repetition_penalty), (1.5, 1.0))
        self.assertFalse(options.enable_thinking)
        self.assertTrue(options.loop_guard)

    def test_non_object_rejected(self):
        for value in (None, [], "", 3, False):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GenerationOptions.from_request(value)

    def test_invalid_numbers_rejected(self):
        for key, value in (("temperature", math.nan), ("top_p", math.inf),
                           ("max_tokens", 0), ("max_tokens", 9000), ("top_k", 20.5),
                           ("presence_penalty", True), ("repetition_penalty", "1.05"),
                           ("repetition_context_size", 0), ("temperature", 10**10000)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                GenerationOptions.from_request({key: value})

    def test_boolean_flags_require_booleans(self):
        for key in ("enable_thinking", "loop_guard"):
            for value in ("false", 0, None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    GenerationOptions.from_request({key: value})

    def test_kwargs_are_not_capabilities(self):
        def step(*, temperature=0, repetition_penalty=None, **kwargs):
            pass
        self.assertEqual(explicit_parameters(step), {"temperature", "repetition_penalty"})

    def test_native_sampling_forwarded(self):
        options = GenerationOptions(repetition_penalty=1.05)
        actual = sampling_kwargs(options, set(options.to_dict()))
        self.assertEqual(actual["repetition_penalty"], 1.05)
        self.assertEqual(actual["presence_penalty"], 1.5)
        self.assertEqual(actual["repetition_context_size"], 256)
        self.assertNotIn("loop_guard", actual)
        self.assertNotIn("enable_thinking", actual)

    def test_non_neutral_unsupported_setting_is_not_ignored(self):
        options = GenerationOptions()
        supported = set(options.to_dict()) - {"presence_penalty"}
        with self.assertRaisesRegex(ValueError, "presence_penalty"):
            sampling_kwargs(options, supported)

    def test_older_runtime_with_penalties_disabled(self):
        options = GenerationOptions(presence_penalty=0, repetition_penalty=1, top_k=0)
        supported = {"max_tokens", "temperature", "top_p"}
        self.assertEqual(set(sampling_kwargs(options, supported)), supported)

    def test_required_penalty_window_not_silently_ignored(self):
        options = GenerationOptions(repetition_penalty=1.05)
        supported = set(options.to_dict()) - {"repetition_context_size"}
        with self.assertRaisesRegex(ValueError, "repetition_context_size"):
            sampling_kwargs(options, supported)


class GuardTests(unittest.TestCase):
    def test_reported_comment_loop(self):
        guard = RepetitionGuard()
        found = None
        count = 0
        for character in "```cpp\n" + ODE_COMMENT * 20:
            count += 1
            found = guard.feed(character)
            if found:
                break
        self.assertIsNotNone(found)
        self.assertEqual(found.repetitions, 3)
        self.assertEqual(found.unit_characters, len(ODE_COMMENT))
        self.assertLessEqual(count, 7 + len(ODE_COMMENT) * 3 + 32)

    def test_two_copies_do_not_stop(self):
        self.assertIsNone(RepetitionGuard().feed(ODE_COMMENT * 2, force=True))

    def test_three_copies_stop(self):
        self.assertIsNotNone(RepetitionGuard().feed(ODE_COMMENT * 3, force=True))

    def test_arbitrary_stream_chunk_boundaries(self):
        for chunk_size in (1, 7, 29, 80, 350):
            with self.subTest(chunk_size=chunk_size):
                source = "Introduction.\n" + ODE_COMMENT * 8
                guard = RepetitionGuard()
                for i in range(0, len(source), chunk_size):
                    if guard.feed(source[i:i+chunk_size]):
                        break
                self.assertIsNotNone(guard.match)

    def test_common_code_tokens_are_not_a_loop(self):
        for unit in ("}\n", "    ", "std::", "std::span<double> ", "return 0;\n"):
            with self.subTest(unit=unit):
                self.assertIsNone(RepetitionGuard().feed(unit * 1000, force=True))

    def test_cpp_with_different_identifiers_does_not_stop(self):
        text = "\n".join(f"const double value_{i} = state[{i}] * step;" for i in range(100))
        self.assertIsNone(RepetitionGuard().feed(text, force=True))

    def test_unicode(self):
        block = "Dette er en lengre forklaring om regulatorens måleverdier og pådrag. " * 1
        block += "Neste steg bruker både æ, ø og å, uten å endre innholdet.\n"
        self.assertIsNotNone(RepetitionGuard().feed(block * 3, force=True))

    def test_short_or_empty_output(self):
        for text in ("", "ok", "x = 1\n", "Hello " * 20):
            with self.subTest(text=text):
                self.assertIsNone(RepetitionGuard().feed(text, force=True))

    def test_different_blocks_do_not_trigger(self):
        text = "".join(ODE_COMMENT.replace("P1999R1", f"Entry-{i:03}") for i in range(20))
        self.assertIsNone(RepetitionGuard().feed(text, force=True))

    def test_detection_is_latched(self):
        guard = RepetitionGuard()
        match = guard.feed(ODE_COMMENT * 3, force=True)
        self.assertIs(guard.feed("another piece of text"), match)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            RepetitionGuard(min_unit=100, max_unit=20)


if __name__ == "__main__":
    unittest.main()
