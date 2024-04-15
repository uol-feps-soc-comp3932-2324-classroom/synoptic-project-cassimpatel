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
	cd tests && ../$(VENV)/bin/python3 -m pytest -vv

# this will only work from a Mac, and if the shell script is present, pointing to host with the repo instantiated
remote_test: run_tests_remote.sh
	caffeinate -disu ./run_tests_remote.sh

clean:
	find . -type f -name '*.pyc' -delete
	rm -rf __pycache__

list:
	@grep '^[^#[:space:]].*:' Makefile

.PHONY: all venv run clean test remote_test list