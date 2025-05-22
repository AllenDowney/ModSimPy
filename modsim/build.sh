#!/bin/sh

# Exit on any error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CANONICAL_FILE="$SCRIPT_DIR/modsim.py"

# Function to copy if destination exists
copy_if_exists() {
    dest="$1"
    if [ -f "$dest" ]; then
        echo "Copying to $dest"
        cp "$CANONICAL_FILE" "$dest"
    fi
}

# Copy to root directory
copy_if_exists "$SCRIPT_DIR/../modsim.py"

# Copy to chapters directory
copy_if_exists "$SCRIPT_DIR/../chapters/modsim.py"

# Copy to examples directory
copy_if_exists "$SCRIPT_DIR/../examples/modsim.py"

# Copy to ModSimPySolutions directory and its subdirectories
copy_if_exists "$SCRIPT_DIR/../ModSimPySolutions/modsim.py"
copy_if_exists "$SCRIPT_DIR/../ModSimPySolutions/examples/modsim.py"
copy_if_exists "$SCRIPT_DIR/../ModSimPySolutions/chapters/modsim.py"

echo "Done copying modsim.py to all locations" 