#!/bin/bash
#using: pyhon3.X -m venv legacyVenv

#setup before executing package.sh:

#sudo apt install python 3.X python3.X-dev python3.X-venv
#python3.X -m venv legacyVenv
#source legacyVenv/bin/activate
#sudo apt install binutils
#pip install -r requirements.txt

set -e

# Define the root directory where the script and build should be located
ROOT_DIR="$HOME/xrd_data_collect"

if [ ! -d "$ROOT_DIR" ]; then
    mkdir -p "$ROOT_DIR"
fi


DIST_DIR="$ROOT_DIR/dist"
BUILD_DIR="$ROOT_DIR/build"
script_dir="$(cd "$(dirname "$0")" && pwd)"

pyinstaller "prod_run.spec" --distpath "$DIST_DIR" --workpath "$BUILD_DIR"
echo "-> Built executable under $ROOT_DIR/dist"
echo "done"
