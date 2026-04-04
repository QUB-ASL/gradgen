import unittest

import gradgen._rust_codegen.rendering as rendering


class RenderingPackageTests(unittest.TestCase):
    def test_package_reexports_core_helpers(self) -> None:
        self.assertTrue(hasattr(rendering, "_emit_node_expr"))
        self.assertTrue(hasattr(rendering, "_build_shared_helper_lines"))
