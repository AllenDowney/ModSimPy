#!/bin/bash
# Build the Jupyter Book for ModSimPy

set -e

# pip install jupyter-book
# pip install ghp-import

# Build the Jupyter Book version

# copy the chapter notebooks
cp ../ModSimPySolutions/chapters/chap*.ipynb .
cp ../ModSimPySolutions/examples/queue.ipynb .

# add tags to hide the solutions
python prep_notebooks.py

# remove the old build
rm -rf _build

# build the HTML version
jb build .

# Create the api/ directory if it doesn't exist
mkdir -p _build/html/api

# Copy the MkDocs docs to the Jupyter Book's HTML output
cp -r ../site/* _build/html/api

echo "Jupyter Book built and MkDocs docs copied to _build/html/api/"

# push it to GitHub
ghp-import -n -p -f _build/html

echo "Pushed to GitHub"