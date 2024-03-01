# TODO: add optional targets for e.g. setting up and installation, running tests, cleanup etc.
# examples: init (install requirements), make venv, install custom packages, run tests

# define the name of the virtual environment directory
VENV := venv

# default target, when make executed without arguments
all: venv

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt

# venv is a shortcut target
venv: $(VENV)/bin/activate

test: venv
	cd tests && ../$(VENV)/bin/python3 -m pytest

clean:
	find . -type f -name '*.pyc' -delete
	rm -rf __pycache__

.PHONY: all venv run clean test