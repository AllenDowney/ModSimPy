#!/bin/bash
# Build the MkDocs documentation for ModSimPy

set -e

# Clean previous build
rm -rf site

# Build HTML docs
mkdocs build
