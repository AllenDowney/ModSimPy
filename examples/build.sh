# copy the notebooks and remove the solutions
cp ../ModSimPySolutions/examples/*.ipynb .
python remove_soln.py

# add and commit
git add *.ipynb
git commit -m "Updating examples"
