"""Launcher for Spiking-WM (Brain-Cog-Lab) that monkeypatches argparse.Namespace
to support config[act] / config[norm] subscripting used by the dmc_proprio
MLP path (official code only exercises dmc_vision). External repo is untouched.
"""
import argparse
import os
import sys

argparse.Namespace.__getitem__ = lambda self, key: getattr(self, key)

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/home/lx/Spiking-WM")

import runpy  # noqa: E402

if __name__ == "__main__":
    runpy.run_path("/home/lx/Spiking-WM/dreamer.py", run_name="__main__")
