import importlib.util
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("builder", ROOT / "tools/build_overlay.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)
METADATA = json.loads((ROOT / "mod.json").read_bytes())


class BuilderTest(unittest.TestCase):
    def test_wrong_source_creates_no_output(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            sdk, output = root / "sdk", root / "out"
            source = sdk / METADATA["files"]["state_address"]["path"]
            source.parent.mkdir(parents=True)
            source.write_bytes(b"wrong source")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                BUILDER.build(sdk, output)
            self.assertFalse(output.exists())
            self.assertEqual(source.read_bytes(), b"wrong source")

    def test_output_must_be_outside_sdk_and_fresh(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            sdk, output = root / "sdk", root / "existing"
            sdk.mkdir()
            output.mkdir()
            with self.assertRaises(ValueError):
                BUILDER.build(sdk, sdk / "outputs")
            with self.assertRaises(FileExistsError):
                BUILDER.build(sdk, output)

    def test_contract_changes_two_pins_only(self):
        files = METADATA["files"]
        names = ("state_address", "serving_warmup")
        original = json.dumps({name: files[name]["source_sha256"] for name in names}).encode()
        result = BUILDER.edit_contract(original, files)
        self.assertEqual(json.loads(result), {name: files[name]["overlay_sha256"] for name in names})
        with self.assertRaises(ValueError):
            BUILDER.edit_contract(original + original, files)

    def test_kernel_and_startup_gate(self):
        compile(BUILDER.KERNEL, "experimental_kernel", "exec")
        self.assertIn(METADATA["live_marker"].encode(), BUILDER.KERNEL)
        self.assertIn(METADATA["startup_variant"].encode(), BUILDER.COMPILE_BEFORE_SEAL)
        self.assertIn(b"record=False", BUILDER.COMPILE_BEFORE_SEAL)
        for record in METADATA["files"].values():
            self.assertRegex(record["source_sha256"], re.compile(r"^[0-9a-f]{64}$"))
            self.assertRegex(record["overlay_sha256"], re.compile(r"^[0-9a-f]{64}$"))
            self.assertNotEqual(record["source_sha256"], record["overlay_sha256"])
