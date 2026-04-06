import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gradgen import Function, FunctionComposer, SX, SXVector
from gradgen.map_zip import reduce_function, zip_function


class FunctionComposerTests(unittest.TestCase):
    def test_function_composer_evaluates_named_chain(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")
        a = SX.sym("a")
        p = SX.sym("p")
        b = SX.sym("b")
        q = SX.sym("q")

        f = Function(
            "f",
            [x, y],
            [x + y],
            input_names=["x", "y"],
            output_names=["b"],
        )
        g = Function(
            "g",
            [a, b],
            [a * b],
            input_names=["a", "b"],
            output_names=["q"],
        )
        h = Function(
            "h",
            [p, q],
            [p + q],
            input_names=["p", "q"],
            output_names=["y"],
        )

        comp = (
            FunctionComposer(f)
            .feed_into(g, arg="b")
            .feed_into(h, arg="q")
            .compose(name="comp")
        )

        self.assertEqual(comp.input_names, ("x", "y", "a", "p"))
        self.assertEqual(comp.output_names, ("y",))
        self.assertEqual(comp(1.0, 2.0, 3.0, 4.0), 13.0)

    def test_function_composer_keeps_reduce_loop_in_rust(self) -> None:
        acc = SX.sym("acc")
        x = SX.sym("x")
        stage = Function(
            "stage",
            [acc, x],
            [acc + x],
            input_names=["acc", "x"],
            output_names=["acc_next"],
        )
        reduced = reduce_function(
            stage,
            3,
            accumulator_input_name="acc0",
            input_name="x_seq",
            name="sum_reduce",
        )
        post = Function(
            "post",
            [acc],
            [acc + 1],
            input_names=["acc"],
            output_names=["y"],
        )

        comp = (
            FunctionComposer(reduced)
            .feed_into(post, arg="acc")
            .compose(name="comp")
        )

        with TemporaryDirectory() as tmpdir:
            project = comp.create_rust_project(Path(tmpdir) / "comp")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("for stage_index in 0..3", lib_text)
            self.assertIn("post_0(", lib_text)
            self.assertIn("sum_reduce_1(", lib_text)
            self.assertIn("pub fn comp(", lib_text)

    def test_function_composer_keeps_zip_loop_in_rust(self) -> None:
        x = SX.sym("x")
        stage = Function(
            "stage",
            [x],
            [x + 1],
            input_names=["x"],
            output_names=["y"],
        )
        zipped = zip_function(stage, 3, input_names=["x_seq"], name="zip")
        y = SXVector.sym("y", 3)
        post = Function(
            "post",
            [y],
            [y.norm2()],
            input_names=["y"],
            output_names=["out"],
        )

        comp = (
            FunctionComposer(zipped)
            .feed_into(post, arg="y")
            .compose(name="comp")
        )

        with TemporaryDirectory() as tmpdir:
            project = comp.create_rust_project(Path(tmpdir) / "zip_comp")
            lib_text = project.lib_rs.read_text(encoding="utf-8")

            self.assertIn("for stage_index in 0..3", lib_text)
            self.assertIn("post_0(", lib_text)
            self.assertIn("zip_1(", lib_text)
            self.assertIn("pub fn comp(", lib_text)


if __name__ == "__main__":
    unittest.main()
