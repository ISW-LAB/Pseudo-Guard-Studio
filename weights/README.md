# Pretrained detector weights

Drop the ultralytics base weights here to train **offline**:

    yolov8n.pt   yolov8s.pt   yolo11n.pt   yolo26n.pt   rtdetr-l.pt

`tools/train_and_predict.py` looks in this folder (and the repository root) before training and
copies the matching file into its work directory, so ultralytics loads it locally instead of
downloading. If the file is not here, ultralytics downloads it on first use — which needs
network access, and fails behind a proxy that blocks it.

The files themselves are not committed: they are 6-100 MB binaries that every user can fetch
from the upstream project, and `.gitignore` excludes `*.pt` for that reason.
