# data/ — a small working dataset

A runnable example, so every command in the top-level README works the moment you clone the
repository. Ten images from the African-Wildlife benchmark, of which **four are already
labeled** and six are not — which is the point: the labeled four are the *seed* the count prior
is estimated from, and the six give Automate Label something to actually do.

```
data/
  images/       10 .jpg
  labels/        4 .txt  (YOLO format, one row per box)
  classes.txt    buffalo, elephant, rhino, zebra
```

## Use it

```bash
python run_app.py --images ./data/images --labels ./data/labels \
                  --classes buffalo,elephant,rhino,zebra
```

The four labeled images show green (yours); the other six are unlabeled. Press **Automate
Label** on one of them, or **Auto-label ALL**, and the acceptance policy fits a per-image box
count from your four.

With a training environment attached you can run the whole loop on this folder:

```bash
python run_app.py --images ./data/images --labels ./data/labels \
                  --classes buffalo,elephant,rhino,zebra \
                  --train-python .venv-train/bin/python
```

Training on ten images takes well under a minute on a CPU. It will not produce a good detector —
that is not what it is for. It proves the pipeline runs end to end on your machine.

## Bring your own data

The layout is all the app needs:

```
<your-folder>/
  images/     .jpg .jpeg .png .bmp
  labels/     <same stem>.txt  — optional; created as you label
```

A label row is `class_id cx cy w h`, normalised to `[0, 1]`, one row per box — the YOLO format
every detection framework reads. You can also start with **no** labels at all and draw the first
few in the app.

## The difference from `demo/`

`demo/` is the sample dataset baked into the installer, so a packaged app has something to open
on first run; it is read-only inside the bundle. `data/` is a normal working folder in your
checkout — write to it, replace it, delete it.
