PROJECT_NAME = ModSimPy
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python


create_environment:
	conda create -y --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt


lint:
	flake8 pacs
	black --check --config pyproject.toml pacs


format:
	black --config pyproject.toml pacs


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


tests:
	cd chapters; pytest --nbmake chap*.ipynb
	# many of the examples don't pass tests without the solutions
	# cd examples; pytest --nbmake *.ipynb
