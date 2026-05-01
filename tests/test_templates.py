import unittest

from gradgen import Function
from gradgen import SX
from gradgen import generate_rust
from gradgen._rust_codegen.templates import _get_template
from gradgen._rust_codegen.templates import _template_environment


class TemplateTests(unittest.TestCase):
    def test_known_templates_are_loadable(self) -> None:
        template = _get_template("lib.rs.j2")
        self.assertTrue(hasattr(template, "render"))

    def test_non_html_templates_are_not_autoescaped(self) -> None:
        environment = _template_environment()
        self.assertFalse(environment.autoescape("lib.rs.j2"))

    def test_generated_rust_is_documented(self) -> None:
        x = SX.sym("x")
        function = Function(
            "doc_demo",
            [x],
            [x + 1.0],
            input_names=["x"],
            output_names=["y"],
        )

        source = generate_rust(function).source

        self.assertIn("//! Generated Rust kernels emitted by gradgen.", source)
        self.assertIn(
            "/// Errors returned by generated Rust kernels when",
            source,
        )
        self.assertIn("/// Metadata describing a generated Rust function.", source)
        self.assertIn("/// Return metadata describing [`doc_demo`].", source)
        self.assertIn("/// Evaluate the generated symbolic function `doc_demo`.", source)
