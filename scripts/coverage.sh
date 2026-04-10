coverage erase
coverage run --source=src/gradgen/ -m pytest tests
coverage html && open htmlcov/index.html
