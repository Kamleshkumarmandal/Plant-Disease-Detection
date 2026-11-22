import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitDemoSmokeTests(unittest.TestCase):
    def test_home_page_runs_without_model_weights(self):
        app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=45)
        errors = [element.message for element in app.exception]
        self.assertEqual(errors, [], "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
