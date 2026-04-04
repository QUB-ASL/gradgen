import unittest

from gradgen import Function, SXVector
from gradgen._rust_codegen.codegen import generate_rust


class CodegenTests(unittest.TestCase):
    def test_generate_rust_produces_public_source(self) -> None:
        x = SXVector.sym("x", 1)
        function = Function(
            "demo",
            [x],
            [x[0] + 1.0],
            input_names=["x"],
            output_names=["y"],
        )

        codegen = generate_rust(function)
        self.assertIn("pub fn", codegen.source)
