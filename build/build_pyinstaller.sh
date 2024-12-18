#!/bin/bash

# Ensure WORKPATH and DISTPATH are provided, otherwise exit with an error
if [[ -z "$WORKPATH" || -z "$DISTPATH" ]]; then
    echo "Error: WORKPATH and DISTPATH must be provided."
    exit 1
fi

# Execute PyInstaller with the provided paths
python -m PyInstaller --workpath "$WORKPATH" --distpath "$DISTPATH" main.spec
