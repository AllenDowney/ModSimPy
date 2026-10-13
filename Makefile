PROJECT_NAME = ModSimPy
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

.PHONY: all create_environment create_environment_dev update_environment update_environment_dev clean tests build

all: build

build:
	@echo "Copying modsim.py"
	cd modsim && sh build.sh
	@echo "Building solutions..."
	cd ModSimPySolutions/chapters && sh build.sh
	cd ModSimPySolutions && sh build.sh
	@echo "Building notebooks..."
	cd examples && sh build.sh
	cd chapters && sh build.sh
	@echo "Pushing ModSimPy (one push, one CI run)..."
	git push
	@echo "Building API documentation..."
	sh build.sh
	@echo "Building Jupyter Book..."
	cd jb && sh build.sh
	@echo "Build complete!"

create_environment:
	mamba env create -f environment.yml
	@echo ">>> mamba env created. Activate with:\nconda activate $(PROJECT_NAME)"

create_environment_dev:
	mamba env create -f environment-dev.yml
	@echo ">>> mamba dev env created. Activate with:\nconda activate $(PROJECT_NAME)"

update_environment:
	mamba env update -f environment.yml --prune

update_environment_dev:
	mamba env update -f environment-dev.yml --prune


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


tests:
	cd modsim; python -m pytest test_modsim.py -v
	cd chapters; pytest --nbmake chap*.ipynb
	# many of the examples don't pass tests without the solutions
	# cd examples; pytest --nbmake *.ipynb
