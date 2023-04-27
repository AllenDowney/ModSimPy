# copy the notebooks and remove the solutions
cp ../ModSimPySolutions/examples/*.ipynb .
python remove_soln.py

# run nbmake
rm modsim.py chap*.py
pytest --nbmake \
bungee1.ipynb \
bungee2.ipynb \
glucose.ipynb \
header.ipynb \
insulin.ipynb \
plague.ipynb \
queue.ipynb \
spiderman.ipynb \
throwingaxe.ipynb \
trees.ipynb \
wall.ipynb \


# add and commit
git add *.ipynb
git commit -m "Updating examples"
git push
