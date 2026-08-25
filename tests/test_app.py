import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as app_module


class ResolveModelPathTests(unittest.TestCase):
    def test_prefers_existing_local_model_file(self):
        resolved = app_module.resolve_model_path()
        resolved_path = Path(resolved)

        self.assertTrue(resolved_path.exists(), f"Expected a model file at {resolved_path}")
        self.assertIn(resolved_path, [
            app_module.BASE_DIR / "yolov8x.pt",
            app_module.BASE_DIR / "yolov8n.pt",
            app_module.PARENT_DIR / "yolov8x.pt",
            app_module.PARENT_DIR / "yolov8n.pt",
        ])


if __name__ == "__main__":
    unittest.main()
