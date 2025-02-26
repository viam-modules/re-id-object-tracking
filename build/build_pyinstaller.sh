#!/bin/bash

# Execute PyInstaller with the provided paths
python -m PyInstaller --workpath "$WORKPATH" --distpath "$DISTPATH" main.spec
