import os
from pathlib import Path


def ensure_dir(path: str) -> str:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory)


def savefig(fig, path: str, tight: bool = True):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.clf()

