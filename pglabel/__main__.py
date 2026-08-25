"""``python -m pglabel`` — the same launcher as ``run_app.py``, without the checkout."""

import sys

from .desktop import main

if __name__ == "__main__":
    sys.exit(main())
