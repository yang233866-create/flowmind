"""Build net.net.xml for this template via netconvert. Run from anywhere."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from simulation.sumo_home import ensure_sumo_home  # noqa: E402


def build() -> Path:
    home = ensure_sumo_home()
    exe = "netconvert.exe" if os.name == "nt" else "netconvert"
    cmd = [
        str(home / "bin" / exe),
        "--node-files", str(HERE / "nodes.nod.xml"),
        "--edge-files", str(HERE / "edges.edg.xml"),
        "--connection-files", str(HERE / "connections.con.xml"),
        "--tllogic-files", str(HERE / "tls.tll.xml"),
        "--output-file", str(HERE / "net.net.xml"),
        "--no-turnarounds",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout.strip():
        print(res.stdout)
    if res.stderr.strip():
        print(res.stderr, file=sys.stderr)
    res.check_returncode()
    return HERE / "net.net.xml"


if __name__ == "__main__":
    print(f"built: {build()}")
