.PHONY: all codegen custom_function multi_function vjp composed_function help

PYTHON ?= python
REPEATS ?= 5

all: codegen custom_function multi_function vjp composed_function

help:
	@echo "Available targets:"
	@echo "  make all"
	@echo "  make codegen"
	@echo "  make custom_function"
	@echo "  make multi_function"
	@echo "  make vjp"
	@echo "  make composed_function REPEATS=$(REPEATS)"

codegen:
	$(PYTHON) codegen/main.py

custom_function:
	$(PYTHON) custom_function/main.py

multi_function:
	$(PYTHON) multi_function/main.py

vjp:
	$(PYTHON) vjp/main.py

composed_function:
	$(PYTHON) composed_function/main.py --repeats $(REPEATS)
