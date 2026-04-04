import unittest

from gradgen._rust_codegen.templates import _get_template


class TemplateTests(unittest.TestCase):
    def test_known_templates_are_loadable(self) -> None:
        template = _get_template("lib.rs.j2")
        self.assertTrue(hasattr(template, "render"))
