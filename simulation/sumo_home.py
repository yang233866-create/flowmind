"""Locate SUMO from the eclipse-sumo pip package and set SUMO_HOME."""
from __future__ import annotations

import os
from pathlib import Path


def ensure_sumo_home() -> Path:
    if "SUMO_HOME" in os.environ and Path(os.environ["SUMO_HOME"]).exists():
        home = Path(os.environ["SUMO_HOME"])
    else:
        import sumo

        home = Path(sumo.__file__).parent
        os.environ["SUMO_HOME"] = str(home)
    bin_dir = str(home / "bin")
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    return home


def sumo_binary(gui: bool = False) -> str:
    home = ensure_sumo_home()
    name = "sumo-gui.exe" if gui else "sumo.exe"
    return str(home / "bin" / name)


def use_libsumo() -> bool:
    """Training can opt into libsumo (5-10x faster) via FLOWMIND_LIBSUMO=1."""
    if os.environ.get("FLOWMIND_LIBSUMO") != "1":
        return False
    try:
        import libsumo  # noqa: F401

        return True
    except ImportError:
        return False
