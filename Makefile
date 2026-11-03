# Author: wanghaobo

.PHONY: check

PYTHON ?= python3

check:
	$(PYTHON) -m unittest discover -s tests -v
