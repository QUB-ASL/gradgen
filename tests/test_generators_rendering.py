import unittest

from gradgen._rust_codegen.generators import rendering as rendering_generators


class GeneratorRenderingTests(unittest.TestCase):
    def test_render_context_and_renderer_are_exposed(self) -> None:
        self.assertTrue(hasattr(rendering_generators, "KernelRenderContext"))
        self.assertTrue(hasattr(rendering_generators, "render_kernel_source"))
