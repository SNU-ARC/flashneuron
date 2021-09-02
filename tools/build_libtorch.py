import argparse
from os.path import dirname, abspath
import sys

# By appending pytorch_root to sys.path, this module can import other torch
# modules even when run as a standalone script. i.e., it's okay either you
# do `python build_libtorch.py` or `python -m tools.build_libtorch`.
pytorch_root = dirname(dirname(abspath(__file__)))
sys.path.append(pytorch_root)

from tools.build_pytorch_libs import build_caffe2
from tools.setup_helpers.cmake import CMake

if __name__ == '__main__':
    # Placeholder for future interface. For now just gives a nice -h.
    parser = argparse.ArgumentParser(description='Build libtorch')
    options = parser.parse_args()

    build_caffe2(version=None, cmake_python_library=None, build_python=False,
                 rerun_cmake=True, cmake_only=False, cmake=CMake())
