import ast
import unittest
from pathlib import Path


class SXDocstringTests(unittest.TestCase):
    def test_all_non_overload_functions_have_multiline_docstrings(self) -> None:
        source = Path("src/gradgen/sx.py").read_text()
        module = ast.parse(source)

        missing: list[str] = []

        def visit(node: ast.AST, qualname: str = "") -> None:
            body = getattr(node, "body", None)
            if body is None:
                return

            for child in body:
                if isinstance(child, ast.FunctionDef):
                    if any(
                        isinstance(decorator, ast.Name) 
                            and decorator.id == "overload"
                            for decorator in child.decorator_list
                    ):
                        continue

                    name = f"{qualname}.{child.name}" \
                        if qualname else child.name
                    doc = ast.get_docstring(child)
                    if doc is None or len(doc.splitlines()) < 2:
                        missing.append(name)

                    visit(child, name)
                elif isinstance(child, ast.ClassDef):
                    visit(child, f"{qualname}.{child.name}"
                          if qualname else child.name)

        visit(module)

        self.assertEqual(missing, [],
                         f"functions missing multiline docstrings: {missing}")
