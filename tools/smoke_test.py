"""Headless smoke test: runs streamlit/app.py in Streamlit's AppTest
harness and fails if the script raises. Run: python tools/smoke_test.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "streamlit" / "app.py"
at = AppTest.from_file(str(APP), default_timeout=180)
at.run()
for e in at.exception:
    print("EXCEPTION:", str(e.value)[:500])
print("title:", [t.value for t in at.title])
print("headers:", [h.value for h in at.header][:4])
assert not at.exception, "app raised an exception"
print("SMOKE OK")
