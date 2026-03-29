.PHONY: all codegen custom_function multi_function vjp composed_function single_shooting help

PYTHON ?= python
REPEATS ?= 5
HORIZON ?= 5

all: codegen custom_function multi_function vjp composed_function single_shooting

help:
	@echo "Available targets:"
	@echo "  make all"
	@echo "  make codegen"
	@echo "  make custom_function"
	@echo "  make multi_function"
	@echo "  make vjp"
	@echo "  make composed_function REPEATS=$(REPEATS)"
	@echo "  make single_shooting HORIZON=$(HORIZON)"

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

single_shooting:
	$(PYTHON) single_shooting/main.py --horizon $(HORIZON)
