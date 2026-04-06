import unittest

from gradgen._rust_codegen.templates import _get_template
from gradgen._rust_codegen.templates import _template_environment


class TemplateTests(unittest.TestCase):
    def test_known_templates_are_loadable(self) -> None:
        template = _get_template("lib.rs.j2")
        self.assertTrue(hasattr(template, "render"))

    def test_non_html_templates_are_not_autoescaped(self) -> None:
        environment = _template_environment()
        self.assertFalse(environment.autoescape("lib.rs.j2"))
