# copy the notebooks and remove the solutions
cp ../ModSimPySolutions/chapters/chap*.ipynb .
python remove_soln.py

# run nbmake
rm modsim.py chap*.py
pytest --nbmake chap*.ipynb

