.PHONY: all codegen custom_function multi_function vjp map_zip zip_3 reduce_map composed_function single_shooting help

PYTHON ?= python
REPEATS ?= 5
HORIZON ?= 5
COUNT ?= 3

all: codegen custom_function multi_function vjp map_zip zip_3 reduce_map composed_function single_shooting

help:
	@echo "Available targets:"
	@echo "  make all"
	@echo "  make codegen"
	@echo "  make custom_function"
	@echo "  make multi_function"
	@echo "  make vjp"
	@echo "  make map_zip COUNT=$(COUNT)"
	@echo "  make zip_3 COUNT=$(COUNT)"
	@echo "  make reduce_map COUNT=$(COUNT)"
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

map_zip:
	$(PYTHON) map_zip/main.py --count $(COUNT)

zip_3:
	$(PYTHON) zip_3/main.py --count $(COUNT)

reduce_map:
	$(PYTHON) reduce_map/main.py --count $(COUNT)

composed_function:
	$(PYTHON) composed_function/main.py --repeats $(REPEATS)

single_shooting:
	$(PYTHON) single_shooting/main.py --horizon $(HORIZON)
