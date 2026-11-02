"""Make a Drama 3 launcher for your ComfyUI.

Drama 3 needs ComfyUI started with --disable-pinned-memory on PCs with
32 GB of RAM or less (otherwise loading can fail with
"cuMemAllocHost ... out of memory"). This script makes a second launcher
that adds that flag, so your normal launcher stays as it is.

How to run it (from anywhere):
    python make_drama3_launcher.py
    python make_drama3_launcher.py "C:\\path\\to\\ComfyUI"

What it does:
  1. Finds your ComfyUI folder (the one with main.py and custom_nodes).
  2. If it finds your normal launcher (.bat on Windows, .sh on Mac/Linux),
     it copies it and adds --disable-pinned-memory to the line that runs main.py.
  3. If not, it writes a new launcher that uses the right Python
     (python_embeded, a venv, or the Python running this script).
  4. Saves it next to your normal launcher. It never changes your normal one.

Only uses Python's standard library.
"""

import os
import stat
import sys
from pathlib import Path

FLAG = "--disable-pinned-memory"
IS_WINDOWS = os.name == "nt"
OUT_NAME = "Start ComfyUI Drama3.bat" if IS_WINDOWS else "start_comfyui_drama3.sh"
PREFERRED = ["run_nvidia_gpu.bat", "run_nvidia_gpu_fast.bat", "run_cpu.bat"]


def is_comfy(p: Path) -> bool:
    return (p / "main.py").is_file() and (p / "custom_nodes").is_dir()


def find_comfy() -> Path | None:
    candidates = []
    if len(sys.argv) > 1:
        candidates.append(Path(sys.argv[1]).expanduser())
    cwd = Path.cwd()
    candidates += [cwd, cwd / "ComfyUI", cwd.parent]
    here = Path(__file__).resolve()
    candidates += list(here.parents)  # works when run from custom_nodes/<pack>/tools
    for c in candidates:
        if is_comfy(c):
            return c.resolve()
        if is_comfy(c / "ComfyUI"):
            return (c / "ComfyUI").resolve()
    return None


def find_launcher(folders: list[Path]) -> Path | None:
    ext = ".bat" if IS_WINDOWS else ".sh"
    found = []
    for folder in folders:
        for f in sorted(folder.glob(f"*{ext}")):
            if f.name == OUT_NAME:
                continue
            try:
                text = f.read_text(errors="ignore")
            except OSError:
                continue
            if "main.py" in text:
                found.append(f)
    for name in PREFERRED:
        for f in found:
            if f.name.lower() == name:
                return f
    return found[0] if found else None


def add_flag(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "main.py" in line and FLAG not in line and not line.lstrip().lower().startswith(("rem", "::", "#")):
            lines[i] = line.rstrip() + " " + FLAG
    return "\n".join(lines) + "\n"


def pick_python(comfy: Path, base: Path) -> str:
    options = [
        base / "python_embeded" / "python.exe",
        comfy / "venv" / "Scripts" / "python.exe",
        comfy / ".venv" / "Scripts" / "python.exe",
        comfy / "venv" / "bin" / "python",
        comfy / ".venv" / "bin" / "python",
    ]
    for p in options:
        if p.is_file():
            return str(p)
    return sys.executable


def new_launcher(comfy: Path, base: Path) -> str:
    py = pick_python(comfy, base)
    main = comfy / "main.py"
    if IS_WINDOWS:
        extra = " --windows-standalone-build" if "python_embeded" in py else ""
        return (
            "@echo off\n"
            "rem Drama 3 launcher: your normal start-up plus " + FLAG + "\n"
            f'cd /d "{base}"\n'
            f'"{py}" -s "{main}"{extra} {FLAG}\n'
            "pause\n"
        )
    return (
        "#!/usr/bin/env bash\n"
        "# Drama 3 launcher: your normal start-up plus " + FLAG + "\n"
        f'cd "{comfy}"\n'
        f'"{py}" "{main}" {FLAG} "$@"\n'
    )


def main() -> int:
    comfy = find_comfy()
    if comfy is None:
        print("Could not find your ComfyUI folder (the one with main.py and custom_nodes).")
        print('Run it again with the path, e.g.:  python make_drama3_launcher.py "C:\\path\\to\\ComfyUI"')
        return 1
    base = comfy.parent
    print(f"ComfyUI folder: {comfy}")

    launcher = find_launcher([base, comfy])
    if launcher:
        print(f"Copying your launcher: {launcher.name}")
        text = add_flag(launcher.read_text(errors="ignore"))
        out_dir = launcher.parent
    else:
        print("No launcher found, writing a new one.")
        text = new_launcher(comfy, base)
        out_dir = base if IS_WINDOWS else comfy

    out = out_dir / OUT_NAME
    if out.exists():
        answer = input(f"{out.name} already exists. Replace it? [y/N] ").strip().lower()
        if answer != "y":
            print("Nothing changed.")
            return 0

    out.write_text(text, newline="\r\n" if IS_WINDOWS else "\n")
    if not IS_WINDOWS:
        out.chmod(out.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"\nMade: {out}")
    print("Use it for Drama 3. Keep your normal launcher for image and video work.")
    print("Close ComfyUI fully before switching between them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
