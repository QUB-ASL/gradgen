import ast
import unittest
from pathlib import Path


class FunctionDocstringTests(unittest.TestCase):
    def test_public_function_methods_have_multiline_docstrings(self) -> None:
        source = Path("src/gradgen/function.py").read_text()
        module = ast.parse(source)

        missing: list[str] = []

        for node in module.body:
            if not isinstance(node, ast.ClassDef) or node.name != "Function":
                continue
            for child in node.body:
                if not isinstance(child, ast.FunctionDef):
                    continue
                doc = ast.get_docstring(child)
                if doc is None or len(doc.splitlines()) < 2:
                    missing.append(child.name)

        self.assertEqual(missing, [], f"Function methods missing multiline docstrings: {missing}")
