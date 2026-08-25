"""PG-Label — the annotation application of Pseudo-Guard Studio.

A local web app: the server is Python's own ``http.server``, the UI is one page served from
``pglabel/static``. Nothing here imports torch — training runs as a subprocess under a separate
interpreter — so the annotator itself installs and runs anywhere Pillow does.

Module map, in dependency order (each layer only imports the ones above it):

    paths          every filesystem root, in a checkout or a packaged install
    labelio        YOLO label files on disk (pure functions)
    geometry       box overlap, NMS, containment-aware de-duplication (pure functions)
    state          the session: open dataset, ownership, job and gate state
    methods        acceptance methodologies + the decisions derived from the human seed
    backend        the app's view of "is there an AI, and what does it propose?"
    candidates     the prediction cache and the two Automate-Label paths
    noise_rule     the negative-crop rule as the review screen exposes it
    training       running the trainer subprocess, and the human review that interrupts it
    dataset_setup  the folder picker, dataset presets, opening a dataset, seeding
    export         COCO / YOLO export
    api            the HTTP route tables and request handler
    cli            command line, start-up configuration, serve loop
    desktop        the double-click entry point (logging, single instance, browser)
"""

from .paths import APP_NAME, APP_VERSION

__all__ = ["APP_NAME", "APP_VERSION"]
__version__ = APP_VERSION
