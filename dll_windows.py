"""Register MSYS2 MinGW DLL directories on Windows before importing libmars.

Call ``ensure_mingw_dll_dirs(repo_root)`` from any entry point that imports ``libmars``
(training, tests, scripts), not only the FastAPI app.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_mingw_dll_dirs(repo_root: Path) -> None:
    """MinGW-built libmars.pyd needs libgcc/libstdc++/libwinpthread/libgomp from MSYS2.

    Python 3.8+ on Windows uses :func:`os.add_dll_directory` for dependent DLLs;
    PATH alone is often insufficient.
    """
    if sys.platform != "win32":
        return
    candidates: list[Path] = []
    _env = (os.environ.get("MSYS2_UCRT64_BIN") or "").strip()
    if _env:
        candidates.append(Path(_env))
    candidates.extend(
        [
            Path(r"C:\msys64\ucrt64\bin"),
            Path(r"C:\msys64\mingw64\bin"),
            Path(r"C:\tools\msys64\ucrt64\bin"),
            repo_root / "build",
            repo_root / "build" / "Release",
        ]
    )
    seen: set[Path] = set()
    for raw in candidates:
        try:
            p = raw.resolve()
        except OSError:
            continue
        if not p.is_dir() or p in seen:
            continue
        seen.add(p)
        try:
            os.add_dll_directory(str(p))
        except (OSError, AttributeError, FileNotFoundError):
            pass
        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
