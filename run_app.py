#!/usr/bin/env python3
"""Launch PG-Label from a source checkout.

    python run_app.py                                   # start screen, bundled demo prefilled
    python run_app.py --images ./my/images --classes cell,rbc,wbc
    python run_app.py --train-python /path/to/env/bin/python    # enable the Train button
    python run_app.py --where                           # print every resolved path and exit

Identical to ``python -m pglabel``; this file exists so the repository can be run without
installing anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pglabel.desktop import main   # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
