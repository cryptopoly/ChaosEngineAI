import unittest
from unittest.mock import patch, MagicMock

from backend_service.plugins import (
    PluginManifest,
    PluginRegistry,
    PluginType,
)


class PluginManifestTests(unittest.TestCase):
    def test_dataclass_defaults(self):
        m = PluginManifest(
            id="test-plugin",
            name="Test Plugin",
            plugin_type=PluginType.TOOL,
        )
        self.assertEqual(m.id, "test-plugin")
        self.assertEqual(m.name, "Test Plugin")
        self.assertEqual(m.plugin_type, PluginType.TOOL)
        self.assertEqual(m.version, "0.1.0")
        self.assertEqual(m.author, "")
        self.assertEqual(m.description, "")
        self.assertTrue(m.enabled)
        self.assertFalse(m.builtin)

    def test_dataclass_custom_values(self):
        m = PluginManifest(
            id="custom",
            name="Custom",
            plugin_type=PluginType.CACHE_STRATEGY,
            version="1.2.3",
            author="Alice",
            description="A custom plugin.",
            builtin=True,
            enabled=False,
        )
        self.assertEqual(m.version, "1.2.3")
        self.assertEqual(m.author, "Alice")
        self.assertTrue(m.builtin)
        self.assertFalse(m.enabled)

    def test_plugin_type_enum_values(self):
        self.assertEqual(PluginType.CACHE_STRATEGY.value, "cache_strategy")
        self.assertEqual(PluginType.INFERENCE_ENGINE.value, "inference_engine")
        self.assertEqual(PluginType.TOOL.value, "tool")
        self.assertEqual(PluginType.MODEL_SOURCE.value, "model_source")
        self.assertEqual(PluginType.POST_PROCESSOR.value, "post_processor")


class PluginRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = PluginRegistry()

    def test_register_and_get(self):
        manifest = PluginManifest(id="p1", name="Plugin 1", plugin_type=PluginType.TOOL)
        instance = MagicMock()
        self.registry.register(manifest, instance)

        result = self.registry.get("p1")
        self.assertIsNotNone(result)
        m, inst = result
        self.assertEqual(m.id, "p1")
        self.assertIs(inst, instance)

    def test_get_nonexistent_returns_none(self):
        self.assertIsNone(self.registry.get("nonexistent"))

    def test_register_without_instance(self):
        manifest = PluginManifest(id="p2", name="Plugin 2", plugin_type=PluginType.MODEL_SOURCE)
        self.registry.register(manifest)
        result = self.registry.get("p2")
        self.assertIsNotNone(result)
        m, inst = result
        self.assertEqual(m.name, "Plugin 2")
        self.assertIsNone(inst)

    def test_list_all(self):
        for i in range(3):
            m = PluginManifest(id=f"p{i}", name=f"Plugin {i}", plugin_type=PluginType.TOOL)
            self.registry.register(m)
        all_plugins = self.registry.list_all()
        self.assertEqual(len(all_plugins), 3)
        ids = {p.id for p in all_plugins}
        self.assertEqual(ids, {"p0", "p1", "p2"})

    def test_list_by_type(self):
        self.registry.register(
            PluginManifest(id="tool1", name="Tool 1", plugin_type=PluginType.TOOL)
        )
        self.registry.register(
            PluginManifest(id="cache1", name="Cache 1", plugin_type=PluginType.CACHE_STRATEGY)
        )
        self.registry.register(
            PluginManifest(id="tool2", name="Tool 2", plugin_type=PluginType.TOOL)
        )

        tools = self.registry.list_by_type(PluginType.TOOL)
        self.assertEqual(len(tools), 2)

        caches = self.registry.list_by_type(PluginType.CACHE_STRATEGY)
        self.assertEqual(len(caches), 1)

        engines = self.registry.list_by_type(PluginType.INFERENCE_ENGINE)
        self.assertEqual(len(engines), 0)

    def test_enable_plugin(self):
        m = PluginManifest(id="p1", name="P1", plugin_type=PluginType.TOOL, enabled=False)
        self.registry.register(m)
        self.assertFalse(self.registry.get("p1")[0].enabled)

        result = self.registry.enable("p1")
        self.assertTrue(result)
        self.assertTrue(self.registry.get("p1")[0].enabled)

    def test_enable_nonexistent_returns_false(self):
        self.assertFalse(self.registry.enable("nonexistent"))

    def test_disable_plugin(self):
        m = PluginManifest(id="p1", name="P1", plugin_type=PluginType.TOOL, enabled=True)
        self.registry.register(m)
        self.assertTrue(self.registry.get("p1")[0].enabled)

        result = self.registry.disable("p1")
        self.assertTrue(result)
        self.assertFalse(self.registry.get("p1")[0].enabled)

    def test_disable_nonexistent_returns_false(self):
        self.assertFalse(self.registry.disable("nonexistent"))

    def test_register_builtins_registers_cache_strategies_and_tools(self):
        reg = PluginRegistry()
        reg.register_builtins()

        all_plugins = reg.list_all()
        self.assertGreater(len(all_plugins), 0)

        # Should have cache strategy plugins
        cache_plugins = reg.list_by_type(PluginType.CACHE_STRATEGY)
        self.assertGreater(len(cache_plugins), 0)
        cache_ids = {m.id for m, _ in cache_plugins}
        self.assertTrue(any("cache." in cid for cid in cache_ids))

        # Should have tool plugins
        tool_plugins = reg.list_by_type(PluginType.TOOL)
        self.assertGreater(len(tool_plugins), 0)
        tool_ids = {m.id for m, _ in tool_plugins}
        self.assertIn("tool.calculator", tool_ids)
        self.assertIn("tool.web_search", tool_ids)

    def test_register_builtins_marks_builtin_flag(self):
        reg = PluginRegistry()
        reg.register_builtins()
        for m in reg.list_all():
            self.assertTrue(m.builtin)


if __name__ == "__main__":
    unittest.main()


class PluginDirectoryDiscoveryTests(unittest.TestCase):
    """What a folder dropped into the plugins directory can and cannot do (review C).

    The registry reads manifests only. Nothing calls ``discover_from_directory`` in the app today, and even
    when it is called an ``entry_point`` is never imported: a dropped folder cannot run code.
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.folder = self.root / "sample"
        self.folder.mkdir()
        (self.folder / "plugin.py").write_text(
            "from pathlib import Path\n"
            "Path(__file__).with_name('IMPORTED.marker').write_text('imported')\n"
            "def register():\n"
            "    Path(__file__).with_name('REGISTERED.marker').write_text('registered')\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _manifest(self, **overrides):
        import json

        data = {"id": "review.sample", "name": "Review Sample", "type": "tool", "entry_point": "plugin:register"}
        data.update(overrides)
        (self.folder / "manifest.json").write_text(json.dumps(data), encoding="utf-8")

    def test_manifest_is_listed_but_its_code_is_never_imported(self):
        self._manifest()
        registry = PluginRegistry()
        registry.discover_from_directory(self.root)
        manifest, instance = registry.get("review.sample")
        self.assertEqual(manifest.entry_point, "plugin:register")
        self.assertFalse(manifest.builtin)
        self.assertIsNone(instance)
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ["manifest.json", "plugin.py"])

    def test_bad_manifests_are_skipped(self):
        registry = PluginRegistry()
        (self.folder / "manifest.json").write_text("{not json", encoding="utf-8")
        registry.discover_from_directory(self.root)
        self._manifest(type="not_a_type")
        registry.discover_from_directory(self.root)
        (self.folder / "manifest.json").write_text('{"name": "no id", "type": "tool"}', encoding="utf-8")
        registry.discover_from_directory(self.root)
        self.assertEqual(registry.list_all(), [])

    def test_missing_directory_is_not_an_error(self):
        registry = PluginRegistry()
        registry.discover_from_directory(self.root / "nope")
        self.assertEqual(registry.list_all(), [])

    def test_the_app_registry_does_not_scan_any_directory(self):
        # The Plugins page tells people to put folders in ~/.chaosengine/plugins/. Nothing reads that
        # folder: if a caller is ever added, it must come with a trust model, and this test must change.
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        hits = subprocess.run(
            ["git", "grep", "-l", "discover_from_directory", "--", "backend_service", "src-tauri", "scripts"],
            cwd=root, capture_output=True, text=True, check=False,
        ).stdout.split()
        self.assertEqual(hits, ["backend_service/plugins/__init__.py"])
