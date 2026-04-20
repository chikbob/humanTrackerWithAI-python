import unittest

from ui.monitoring import (
    _build_live_window_url,
    _max_rendered_sources,
    _resolve_default_selection,
    _resolve_standalone_binding,
)


def _bindings():
    return [
        {"source_id": 10, "kind": "production", "label": "Камера 1 [rtsp]"},
        {"source_id": "browser-live", "kind": "browser_camera", "label": "Браузерная камера"},
        {"source_id": "local-macbook", "kind": "local_camera", "label": "Локальная камера MacBook"},
    ]


class MonitoringHelperTests(unittest.TestCase):
    def test_resolve_default_selection_by_source_id(self):
        self.assertEqual(_resolve_default_selection(_bindings(), "", "10"), ["Камера 1 [rtsp]"])

    def test_resolve_default_selection_prefers_browser_camera(self):
        self.assertEqual(_resolve_default_selection(_bindings(), "browser_camera", ""), ["Браузерная камера"])

    def test_resolve_standalone_binding_by_kind(self):
        binding = _resolve_standalone_binding(_bindings(), preferred_source="", preferred_source_id="", preferred_source_kind="local_camera")
        self.assertEqual(binding["label"], "Локальная камера MacBook")

    def test_live_window_url_contains_stable_source_parameters(self):
        url = _build_live_window_url({"kind": "production", "source_id": 7})
        self.assertIn("source_id=7", url)
        self.assertIn("source_kind=production", url)

    def test_max_rendered_sources(self):
        self.assertEqual(_max_rendered_sources("single"), 1)
        self.assertEqual(_max_rendered_sources("2x2 grid"), 4)
        self.assertEqual(_max_rendered_sources("list"), 6)


if __name__ == "__main__":
    unittest.main()
