#!/bin/bash
# Build the MkDocs documentation for ModSimPy

set -e

# Get the canonical modsim.py
cp ModSimPySolutions/chapters/modsim.py modsim/
cp ModSimPySolutions/chapters/modsim.py .

# Clean previous build
rm -rf site

# Build HTML docs
mkdocs build
