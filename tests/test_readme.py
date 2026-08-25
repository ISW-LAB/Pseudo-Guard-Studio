"""The README is documentation users act on — its links and figures have to resolve.

A broken image or a link to a moved file is invisible until someone reads the repository page,
which is exactly when it costs the most. These checks are cheap and catch both.
"""

from __future__ import annotations

import re
import sys
import unittest

from support import ROOT                                            # noqa: E402


def _documented_commands():
    """Full shell commands from every doc, with backslash continuations joined back up."""
    out = []
    for doc in ("README.md", "docs/WINDOWS.md", "data/README.md", "weights/README.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        for block in re.findall(r"```(?:bash|bat|sh)?\n(.*?)```", text, re.S):
            buffer = ""
            for raw in block.splitlines():
                line = raw.rstrip()
                if not line.strip() or line.strip().startswith(("#", "REM")):
                    continue
                if line.endswith("\\"):
                    buffer += line[:-1].strip() + " "
                    continue
                out.append((buffer + line.strip()).strip())
                buffer = ""
    return [c for c in out if re.match(r"^(python|py -3)\b", c)]


class TestReadme(unittest.TestCase):
    def setUp(self):
        self.text = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_every_image_resolves(self):
        missing = [src for src in re.findall(r'src="([^"]+)"', self.text)
                   if not src.startswith("http") and not (ROOT / src).exists()]
        self.assertEqual(missing, [], "README references images that are not in the repository")

    def test_every_local_link_resolves(self):
        missing = []
        for target in re.findall(r"\]\(([^)#][^)]*)\)", self.text):
            if target.startswith(("http", "#", "<")):
                continue
            if not (ROOT / target).exists():
                missing.append(target)
        self.assertEqual(missing, [], "README links to paths that do not exist")

    def test_no_figure_is_shipped_unused(self):
        used = {src.split("/")[-1] for src in re.findall(r'src="(docs/figures/[^"]+)"', self.text)}
        present = {p.name for p in (ROOT / "docs" / "figures").iterdir() if p.is_file()}
        self.assertEqual(sorted(present - used), [],
                         "these figures are committed but never shown")

    def test_the_documented_entry_points_exist(self):
        for command in ("run_app.py", "tools/train_and_predict.py", "tools/train_validator.py",
                        "tools/gen_noise_crops.py", "tools/precompute_overlays.py",
                        "packaging/build.py", "packaging/make_portable_zip.py",
                        "packaging/verify_build.py"):
            self.assertIn(command, self.text, f"README no longer documents {command}")
            self.assertTrue((ROOT / command).exists(), command)

    def test_every_documented_command_names_a_real_script(self):
        import shlex
        missing = []
        for cmd in _documented_commands():
            for token in shlex.split(cmd.replace("^", "")):
                if token.endswith((".py", ".bat")) and not (ROOT / token).exists():
                    missing.append(cmd)
        self.assertEqual(missing, [], "the docs invoke scripts that are not in this repository")

    def test_every_documented_flag_is_actually_accepted(self):
        """A flag the docs show and the program rejects is worse than an undocumented one.

        ``--where`` used to be handled before argparse, so it worked but never appeared in
        ``--help`` — a user following the README could not discover it from the program itself.
        """
        import re
        import shlex
        import subprocess
        wrong = []
        for cmd in _documented_commands():
            parts = shlex.split(cmd.replace("^", ""))
            script = next((p for p in parts if p.endswith(".py")), None)
            if script is None or not (ROOT / script).exists():
                continue
            used = {p.split("=")[0] for p in parts if p.startswith("--")}
            if not used:
                continue
            out = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                                 capture_output=True, text=True, timeout=120)
            accepted = set(re.findall(r"(--[a-z0-9][a-z0-9-]*)", out.stdout))
            for flag in sorted(used - accepted):
                wrong.append(f"{script} {flag}")
        self.assertEqual(wrong, [], "documented flags that --help does not list")

    def test_the_quoted_test_count_matches_reality(self):
        # The badge says a number; make it a number that has to stay true.
        claimed = re.search(r"tests-(\d+)%20passing", self.text)
        self.assertIsNotNone(claimed, "the test badge is missing")
        import unittest as ut
        loader = ut.TestLoader()
        suite = loader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT / "tests"))
        actual = suite.countTestCases()
        self.assertEqual(int(claimed.group(1)), actual,
                         f"README badge says {claimed.group(1)} tests, the suite has {actual}")


if __name__ == "__main__":
    unittest.main()
