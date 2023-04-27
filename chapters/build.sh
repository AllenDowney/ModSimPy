# copy the notebooks and remove the solutions
cp ../ModSimPySolutions/chapters/chap*.ipynb .
python remove_soln.py

# run nbmake
rm modsim.py chap*.py
pytest --nbmake chap*.ipynb

# add and commit
git add chap*.ipynb chap*.py
git commit -m "Updating chapters"

# build the zip file
cd ../..; zip -r ModSimPyNotebooks.zip \
    ModSimPy/chapters/chap*.ipynb \
    ModSimPy/examples/*.ipynb

# add and commit it
mv ModSimPyNotebooks.zip ModSimPy
cd ModSimPy

git add ModSimPyNotebooks.zip
git commit -m "Updating the notebook zip file"

git push
