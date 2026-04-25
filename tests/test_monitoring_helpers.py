import unittest

try:
    from ui.monitoring import (
        _build_source_bindings,
        _build_live_window_url,
        _max_rendered_sources,
        _resolve_default_selection,
        _resolve_standalone_binding,
    )
except ModuleNotFoundError:
    _build_source_bindings = None
    _build_live_window_url = None
    _max_rendered_sources = None
    _resolve_default_selection = None
    _resolve_standalone_binding = None


def _bindings():
    return [
        {"source_id": 10, "kind": "production", "name": "Камера 1", "label": "Камера 1 [rtsp]"},
        {"source_id": "browser-live", "kind": "browser_camera", "name": "Браузер оператора", "label": "Браузер оператора [browser_camera]"},
        {"source_id": "local-macbook", "kind": "local_camera", "name": "Камера MacBook", "label": "Камера MacBook [local_camera]"},
    ]


@unittest.skipIf(_resolve_default_selection is None, "monitoring runtime dependencies are not installed")
class MonitoringHelperTests(unittest.TestCase):
    def test_resolve_default_selection_by_source_id(self):
        self.assertEqual(_resolve_default_selection(_bindings(), "", "10"), ["Камера 1 [rtsp]"])

    def test_resolve_default_selection_prefers_browser_camera(self):
        self.assertEqual(_resolve_default_selection(_bindings(), "browser_camera", ""), ["Браузер оператора [browser_camera]"])

    def test_resolve_default_selection_prefers_macbook_camera_without_explicit_hint(self):
        self.assertEqual(_resolve_default_selection(_bindings(), "", ""), ["Камера MacBook [local_camera]"])

    def test_resolve_standalone_binding_by_kind(self):
        binding = _resolve_standalone_binding(_bindings(), preferred_source="", preferred_source_id="", preferred_source_kind="local_camera")
        self.assertEqual(binding["label"], "Локальная камера MacBook")

    def test_build_source_bindings_maps_browser_camera_and_prioritizes_operator_browser(self):
        bindings = _build_source_bindings(
            [
                {"id": 2, "name": "Камера 1", "source_type": "rtsp"},
                {"id": 1, "name": "Браузер оператора", "source_type": "browser_camera"},
            ],
            {},
        )

        self.assertEqual(bindings[0]["name"], "Камера MacBook")
        self.assertEqual(bindings[0]["kind"], "local_camera")
        self.assertEqual(bindings[1]["name"], "Браузер оператора")
        self.assertEqual(bindings[1]["kind"], "browser_camera")
        self.assertEqual(bindings[2]["kind"], "production")

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
