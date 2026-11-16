"""Make the `powerfc` package importable when a script is run directly.

Every script starts with `import _bootstrap`. That adds the project's `src`
folder to Python's import path, so `from powerfc import ...` works even if the
package was never installed.

Why not rely on `pip install -e .` alone? On a Mac whose Documents folder is
synced with iCloud, files inside `.venv` get a "hidden" flag, and recent
Python versions silently ignore the hidden file an editable install depends
on. This two-line shim sidesteps that and costs nothing elsewhere.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
