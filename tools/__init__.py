"""Training entry points, run by the TRAINING interpreter (torch + ultralytics).

These are never imported by the application: the app hands them to a separate interpreter as
file paths, which is what keeps the annotator itself free of the heavy ML stack.

    train_and_predict.py   the Train button — detector + validator + candidate overlay
    gen_noise_crops.py     fabricate the validator's crops for the human review screen
    train_validator.py     retrain only the validator (cheap, no detector run)
    precompute_overlays.py freeze a candidate overlay so sessions need no GPU
    common.py              helpers shared by the above
"""
