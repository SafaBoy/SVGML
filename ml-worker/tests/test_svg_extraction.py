"""Unit tests for the SVG extraction helpers used by direct HF text-to-SVG.

Run from the ml-worker directory:

    .venv/bin/python -m unittest discover -s tests -t .
"""

import unittest
import xml.etree.ElementTree as ET

import app


def parses(svg: str) -> bool:
    try:
        ET.fromstring(svg)
        return True
    except ET.ParseError:
        return False


class ExtractSvgDocumentTests(unittest.TestCase):
    def test_plain_svg_passes_through(self):
        source = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
            '<rect width="10" height="10"/></svg>'
        )
        svg, reason = app.extract_svg_document(source)
        self.assertEqual(reason, "")
        self.assertEqual(svg, source)
        self.assertTrue(parses(svg))

    def test_strips_markdown_fence_and_prose(self):
        source = 'Here you go:\n```xml\n<svg viewBox="0 0 4 4"><circle r="2"/></svg>\n```\nEnjoy!'
        svg, reason = app.extract_svg_document(source)
        self.assertEqual(reason, "")
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertNotIn("```", svg)

    def test_drops_thinking_block(self):
        source = '<think>plan: draw a circle</svg> oops</think><svg viewBox="0 0 4 4"><circle r="2"/></svg>'
        svg, reason = app.extract_svg_document(source)
        self.assertEqual(reason, "")
        self.assertNotIn("plan:", svg)
        self.assertTrue(parses(svg))

    def test_repairs_truncated_document(self):
        source = '<svg viewBox="0 0 8 8"><g><rect width="8" height="8"'
        svg, reason = app.extract_svg_document(source)
        self.assertEqual(reason, "")
        self.assertIn("</g></svg>", svg)
        self.assertTrue(parses(svg))

    def test_injects_missing_xmlns(self):
        svg, reason = app.extract_svg_document('<svg viewBox="0 0 4 4"><circle r="2"/></svg>')
        self.assertEqual(reason, "")
        self.assertIn('xmlns="http://www.w3.org/2000/svg"', svg)

    def test_reports_missing_svg_tag(self):
        svg, reason = app.extract_svg_document("sorry, I cannot draw that")
        self.assertEqual(svg, "")
        self.assertIn("no <svg> tag", reason)

    def test_reports_empty_output(self):
        svg, reason = app.extract_svg_document("   ")
        self.assertEqual(svg, "")
        self.assertIn("no text", reason)

    def test_reports_malformed_xml(self):
        svg, reason = app.extract_svg_document("<svg viewBox='0 0 4 4'><circle r=></svg>")
        self.assertEqual(svg, "")
        self.assertIn("not well-formed", reason)


class CloseUnclosedTagsTests(unittest.TestCase):
    def test_nested_stack_order(self):
        self.assertEqual(
            app._close_unclosed_tags("<svg><g><rect/>"),
            "<svg><g><rect/></g></svg>",
        )

    def test_ignores_void_and_balanced_tags(self):
        fragment = "<svg><br/><path d='M0 0'/></svg>"
        self.assertEqual(app._close_unclosed_tags(fragment), fragment)


class ClassifyHfModelTests(unittest.TestCase):
    def test_omnisvg_detected_without_loading_config(self):
        self.assertEqual(app.classify_hf_model("OmniSVG/OmniSVG1.1_4B"), "omnisvg")

    def test_unknown_model_raises_runtime_error(self):
        original = app.hf_model_config

        def boom(model_id):
            raise OSError("no config.json")

        app.hf_model_config = boom
        try:
            with self.assertRaises(RuntimeError):
                app.classify_hf_model("some/lora-adapter-repo")
        finally:
            app.hf_model_config = original


class ShortReasonTests(unittest.TestCase):
    def test_flattens_whitespace_and_truncates(self):
        self.assertEqual(app.short_reason("a\n b\tc"), "a b c")
        self.assertTrue(app.short_reason("x" * 400, limit=20).endswith("…"))
        self.assertEqual(len(app.short_reason("x" * 400, limit=20)), 20)


if __name__ == "__main__":
    unittest.main()
