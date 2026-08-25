# Windows: building, installing, and verifying

Two ways to ship this app on Windows. They produce the same experience for the user and differ
only in where they can be built and how the code is packaged.

| | `build.py` (PyInstaller) | `make_portable_zip.py` |
|---|---|---|
| Build machine | **Windows only** | any OS, including Linux |
| What ships | one frozen `PG-Label.exe` + `_internal\` | Python runtime + readable `.py` sources |
| Size | ~60–90 MB | ~20 MB + sources |
| Installer | Inno Setup `PG-Label-Setup-<version>.exe` | `Install PG-Label.exe` (per-user copy) |
| Code is inspectable | no (frozen bytecode) | yes |

Neither bundles PyTorch. That is deliberate: freezing it would add ~2.5 GB and still not give
the user a working CUDA stack, because the right wheel depends on their driver. Training runs
in a separate interpreter — see [The training pack](#the-training-pack).

---

## A. Frozen executable (on Windows)

Requires Python 3.9+ (64-bit) from python.org or the Microsoft Store. Everything else goes into
a throwaway venv under `packaging\build\`; your Python is left untouched.

```bat
packaging\build.bat
packaging\build.bat --installer      REM also compile the Inno Setup installer
packaging\build.bat --no-training    REM label-only build (no Train button)
packaging\build.bat --windowed       REM no console window
packaging\build.bat --clean          REM remove previous artefacts first
```

Output:

```
packaging\dist\PG-Label\PG-Label.exe            the app — ship the WHOLE folder
packaging\dist\PG-Label-Setup-1.0.0.exe          the installer (with --installer)
```

For the installer you also need Inno Setup:

```bat
winget install -e --id JRSoftware.InnoSetup
```

The installer is **per user**: it writes to `%LOCALAPPDATA%\Programs\PG-Label`, needs no
administrator rights, and triggers no UAC prompt — which matters on managed lab machines where
annotators are not administrators.

## B. Portable release (from any OS)

```bash
python packaging/make_portable_zip.py                 # -> pseudo-guard-studio.zip
python packaging/make_portable_zip.py --no-demo       # smaller
python packaging/make_portable_zip.py --keep-tree     # leave the staged tree to inspect
```

It downloads Python's official Windows embeddable runtime and the matching Pillow wheel (cached
under `packaging/build/cache`), copies the app sources next to them, and writes three `.exe`
launchers — the same distlib stub pip uses for console scripts. Each `.exe` has a `.cmd` twin
for machines whose endpoint protection blocks unsigned launchers.

The user unzips it and runs `Install PG-Label.exe`, or just `PG-Label.exe` to try it in place.

---

## Verifying a build

```bash
python packaging/verify_build.py                                    # source tree, any OS
python packaging/verify_build.py --dist packaging/dist/PG-Label     # a built folder
```

It checks the things that otherwise fail only on a user's machine:

- every file the spec bundles as data actually exists;
- the app imports with **no** torch present;
- no module under `pglabel/` imports a heavy ML package;
- every local module the trainer reaches resolves inside the bundle;
- the UI's stylesheet and script are linked and no inline block was left behind;
- the version in the installer matches `pglabel/paths.py`.

Exit code 0 means all checks passed, so it can gate a release.

## What is already guarded

These are the Windows-specific failures the code handles deliberately. Knowing they are covered
is what lets the manual checklist below stay short.

| Windows behaviour | Where it would bite | How it is handled |
|:--|:--|:--|
| Joining an ABSOLUTE path discards the base (`images / "C:\Windows\x"`) | `/api/file/…` could serve any readable file | every URL-supplied name is resolved and required to stay inside the dataset (`pglabel/api.py: safe_image`) |
| A redirected stdout falls back to the ANSI code page; `—` `⏸` `✅` are not in cp949/cp1252/cp437 | `PG-Label.exe > log.txt` dies with `UnicodeEncodeError` | every entry point calls `console.enable()` first — UTF-8 with `errors="replace"` |
| Read-only files refuse to be deleted | `build.py --clean`, the uninstaller, stale staged datasets | `fsutil.remove_tree` clears the bit and retries |
| Renaming onto an existing file fails | log rotation at 2 MB | `os.replace`, not `Path.rename` |
| An open file cannot be deleted | the uninstaller runs from the folder it removes | the removal reports failure and a detached `.cmd` finishes the job |
| `cscript` reads a `.vbs` as ANSI unless it finds a UTF-16 BOM | shortcut creation under `C:\Users\<non-ASCII name>` | the fallback `.vbs` is written as UTF-16 |
| `.cmd` files are read in the OEM code page | the uninstaller's cleanup script | written through `_write_cmd`, which uses `mbcs` |
| No `killpg`; a child holds the GPU | Stop must kill `conda run` *and* the trainer | `taskkill /T /F` on a new process group |
| Symlinks need developer mode or admin | staging a dataset for training | `symlink_to` with a copy fallback |
| A frozen bundle's library path leaks into children | the trainer loads the app's Python/SSL/JPEG libraries | `<VAR>_ORIG` is restored and `_MEIPASS*` stripped before spawning |

The training interpreter can only import `pseudoguard`, `pgcount` and `tools` — the packages the
spec ships beside the executable. `verify_build.py` fails the build if anything under `tools/`
reaches outside that set, because such an import works from a checkout and breaks in every
packaged install.

---

## Manual checklist on the target machine

Automated checks cannot cover these. Run through them once per release:

1. `PG-Label.exe --where` prints resolved paths, and `research_root` is **not** empty
   (empty means the build shipped without training support).
2. Double-click `PG-Label.exe`: a console window appears and a browser opens on
   `http://127.0.0.1:8000`.
3. The start screen shows the bundled demo dataset; press **Start**.
4. Draw a box, press **Save**, restart the app — the box is still there, and the image is
   still marked as yours (green).
5. **Automate Label** on an unlabeled image produces boxes (needs a trained model or an
   overlay).
6. **Export COCO** writes `instances_export.json` into the labels folder.
7. Close the console window: the app stops, and no `python.exe` is left running.
8. With the training pack installed, press **Train**: the log streams, training pauses at the
   crop review, **Confirm** resumes it, and **Stop** kills the whole process tree.
9. Uninstall: the app is gone and `%LOCALAPPDATA%\PG-Label` — your labels — is still there.

---

## The training pack

The Train and Run-cycle buttons need PyTorch. Install it once:

```bat
REM from the Start menu: "Install training pack", or:
packaging\install_training_pack.bat
packaging\install_training_pack.bat --cuda cpu       REM no NVIDIA GPU (~250 MB instead of ~2.5 GB)
packaging\install_training_pack.bat --python "C:\Python311\python.exe"
```

It creates one folder (`%LOCALAPPDATA%\PG-Label\gpu-env`), installs torch + ultralytics into
it, and registers it with the app. It needs a normal Python 3.9–3.12 (64-bit) on the PC:

```bat
winget install -e --id Python.Python.3.11
```

Already have a conda environment with torch? Skip all of that and point the app at it:

```bat
PG-Label.exe --set-train-python "C:\Users\me\miniconda3\envs\mytorch\python.exe"
```

Training falls back to the CPU when no GPU is present — slower, but it works.

---

## Where your files live

| What | Where |
|---|---|
| Labels (workspace) | `%LOCALAPPDATA%\PG-Label\workspace` |
| Settings | `%LOCALAPPDATA%\PG-Label\settings.json` |
| Log | `%LOCALAPPDATA%\PG-Label\logs\pglabel.log` |
| Training pack | `%LOCALAPPDATA%\PG-Label\gpu-env` |
| The app itself | `%LOCALAPPDATA%\Programs\PG-Label` |

Uninstalling removes the app but **not** `%LOCALAPPDATA%\PG-Label` — losing annotation work to
an uninstall would be unforgivable. Delete that folder by hand when you actually mean to.

## Troubleshooting

**"Windows protected your PC"** — SmartScreen warns about every unsigned program. *More info →
Run anyway*.

**An `.exe` will not start** — run the `.cmd` file of the same name; it does exactly the same
thing without the launcher stub.

**The Train button is missing** — the build shipped label-only, or no training interpreter is
registered. Check with `PG-Label.exe --where`: `research_root` and `train_python` must both be
non-empty.

**Training fails immediately** — open the log
(`%LOCALAPPDATA%\PG-Label\logs\pglabel.log`). A missing interpreter reports
`cannot start the trainer`; fix it with `--set-train-python`.

**Port 8000 is taken** — the app scans upward for a free port automatically. To pin one:
`PG-Label.exe --port 8123`.
