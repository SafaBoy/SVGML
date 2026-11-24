"""Tests for honest reporting of weak SVG output and the worker VRAM budget.

Run from the ml-worker directory:

    .venv/bin/python -m unittest discover -s tests -t .
"""

import unittest

import app

CRUISE_PROMPT = (
    "A Cruise ship on the seas. Bottom half of the hull needs to be "
    "navy blue and top half white."
)

BAD_SVG = (
    '<svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">'
    "<!-- Bottom half is navy blue -->"
    '<rect width="512" height="256" fill="#6366F1"/>'
    '<rect y="256" width="512" height="256" fill="#FFFFFF"/>'
    "</svg>"
)

GOOD_SVG = (
    '<svg viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">'
    "<!-- hull: navy blue below, white above -->"
    '<rect width="512" height="256" fill="#FFFFFF"/>'
    '<rect y="256" width="512" height="256" fill="#000080"/>'
    '<path d="M0 256 L512 256"/>'
    '<path d="M64 256 L64 448"/>'
    '<path d="M448 256 L448 448"/>'
    '<circle cx="256" cy="128" r="32" fill="#FFFFFF"/>'
    "</svg>"
)


class PromptColourTests(unittest.TestCase):
    def test_explicit_colours_are_extracted_from_the_prompt(self):
        terms = app.prompt_color_terms(CRUISE_PROMPT)
        self.assertIn("navy blue", terms)
        self.assertIn("white", terms)

    def test_instruction_tells_model_prompt_colours_beat_the_palette(self):
        instruction = app._svg_instruction(
            CRUISE_PROMPT, "vibrant-gradient", ["#6366F1", "#EC4899"]
        ).lower()
        self.assertIn("navy blue", instruction)
        self.assertIn("white", instruction)
        self.assertIn("take precedence", instruction)


class SvgQualityTests(unittest.TestCase):
    def test_warns_when_requested_colour_is_missing(self):
        warning = app.check_svg_quality(BAD_SVG, CRUISE_PROMPT)
        self.assertTrue(warning)
        self.assertIn("navy blue", warning.lower())

    def test_accepts_a_shape_rich_svg_with_the_requested_colours(self):
        self.assertEqual(app.check_svg_quality(GOOD_SVG, CRUISE_PROMPT), "")


class WorkerHealthTests(unittest.TestCase):
    def test_reports_vram_budget_as_an_integer(self):
        vram = app.health()["vram_total_mb"]
        self.assertIsInstance(vram, int)
        self.assertGreaterEqual(vram, 0)


if __name__ == "__main__":
    unittest.main()
