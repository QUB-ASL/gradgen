import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import sympy as sp

from gradgen.function import Function
from gradgen.sx import SX, SXVector
import gradgen.map_zip as map_zip


class ReduceFunctionTests(unittest.TestCase):
    @staticmethod
    def _run_cargo(project_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["cargo", *args],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def _append_rust_test(project_dir: Path, test_source: str) -> None:
        lib_rs = project_dir / "src" / "lib.rs"
        with lib_rs.open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(test_source)

    @staticmethod
    def _rust_array_literal(values: list[float], scalar_type: str) -> str:
        return "[" + ", ".join(f"{repr(float(value))}_{scalar_type}" for value in values) + "]"

    def _require_reduce_api(self):
        self.assertTrue(
            hasattr(map_zip, "reduce_function"),
            "Expected gradgen.map_zip.reduce_function to exist",
        )
        self.assertTrue(
            hasattr(map_zip, "ReducedFunction"),
            "Expected gradgen.map_zip.ReducedFunction to exist",
        )
        return map_zip.reduce_function, map_zip.ReducedFunction

    def test_reduce_api_exists(self) -> None:
        self._require_reduce_api()

    def test_reduce_validates_stage_function_signature(self) -> None:
        reduce_function, _ = self._require_reduce_api()

        x = SX.sym("x")
        y = SX.sym("y")
        z = SX.sym("z")

        unary = Function("unary", [x], [x + 1.0])
        ternary = Function("ternary", [x, y, z], [x + y + z])
        two_outputs = Function("two_outputs", [x, y], [x + y, x - y])

        with self.assertRaises(ValueError):
            reduce_function(unary, 3)
        with self.assertRaises(ValueError):
            reduce_function(ternary, 3)
        with self.assertRaises(ValueError):
            reduce_function(two_outputs, 3)

    def test_reduce_requires_output_shape_to_match_accumulator_shape(self) -> None:
        reduce_function, _ = self._require_reduce_api()

        acc = SXVector.sym("acc", 2)
        x = SX.sym("x")
        bad_stage = Function(
            "bad_stage",
            [acc, x],
            [x],
            input_names=["acc", "x"],
            output_names=["acc_next"],
        )

        with self.assertRaises(ValueError):
            reduce_function(bad_stage, 2)

    def test_reduce_to_function_matches_left_fold_scalar(self) -> None:
        reduce_function, _ = self._require_reduce_api()

        acc = SX.sym("acc")
        x = SX.sym("x")
        stage = Function(
            "stage",
            [acc, x],
            [acc + 2.0 * x],
            input_names=["acc", "x"],
            output_names=["acc_next"],
        )

        reduced = reduce_function(
            stage,
            4,
            accumulator_input_name="acc0",
            input_name="x_seq",
            name="stage_reduce",
            simplification="medium",
        )
        expanded = reduced.to_function()

        # (((1 + 2*3) + 2*(-1)) + 2*4) + 2*0.5 = 14
        result = expanded(1.0, [3.0, -1.0, 4.0, 0.5])
        self.assertAlmostEqual(result, 14.0)

    def test_reduce_respects_left_fold_order_for_non_commutative_stage(self) -> None:
        reduce_function, _ = self._require_reduce_api()

        acc = SX.sym("acc")
        x = SX.sym("x")
        stage = Function(
            "stage_non_commutative",
            [acc, x],
            [acc - x],
            input_names=["acc", "x"],
            output_names=["acc_next"],
        )

        reduced = reduce_function(stage, 3, accumulator_input_name="acc0", input_name="x_seq")
        expanded = reduced.to_function()

        self.assertAlmostEqual(expanded(10.0, [1.0, 2.0, 3.0]), 4.0)

    def test_reduce_supports_vector_accumulator_and_vector_sequence_element(self) -> None:
        reduce_function, _ = self._require_reduce_api()

        acc = SXVector.sym("acc", 2)
        x = SXVector.sym("x", 2)
        stage = Function(
            "vec_stage",
            [acc, x],
            [SXVector((acc[0] + x[0], acc[1] - x[1]))],
            input_names=["acc", "x"],
            output_names=["acc_next"],
        )

        reduced = reduce_function(
            stage,
            3,
            accumulator_input_name="acc0",
            input_name="x_seq",
            name="vec_reduce",
        )
        expanded = reduced.to_function()

        # acc0=(1,10), x_seq=[(2,3), (-1,4), (5,-2)]
        # step1 -> (3,7), step2 -> (2,3), step3 -> (7,5)
        self.assertEqual(expanded([1.0, 10.0], [2.0, 3.0, -1.0, 4.0, 5.0, -2.0]), (7.0, 5.0))

    def test_reduce_codegen_preserves_loop_structure(self) -> None:
        reduce_function, _ = self._require_reduce_api()

        acc = SX.sym("acc")
        x = SX.sym("x")
        stage = Function("stage", [acc, x], [acc + x], input_names=["acc", "x"], output_names=["acc_next"])
        reduced = reduce_function(stage, 5, accumulator_input_name="acc0", input_name="x_seq", name="sum_reduce")

        codegen = reduced.generate_rust(function_name="sum_reduce_kernel")

        self.assertIn("for stage_index in 0..5", codegen.source)

    def test_reduce_generated_rust_matches_sympy_reference(self) -> None:
        reduce_function, _ = self._require_reduce_api()

        count = 4
        acc0_symbol = sp.Symbol("acc0", real=True)
        x_symbols = sp.symbols("x0:4", real=True)

        sympy_acc = acc0_symbol
        for x_i in x_symbols:
            sympy_acc = sp.sin(sympy_acc) + (sympy_acc * x_i) + (x_i**2)

        acc0_value = 0.35
        x_values = [0.2, -0.4, 0.6, 0.1]
        substitutions = {acc0_symbol: acc0_value, **{sym: val for sym, val in zip(x_symbols, x_values)}}
        expected = float(sympy_acc.subs(substitutions).evalf())

        acc = SX.sym("acc")
        x = SX.sym("x")
        stage = Function(
            "nonlinear_stage",
            [acc, x],
            [acc.sin() + acc * x + x * x],
            input_names=["acc", "x"],
            output_names=["acc_next"],
        )
        reduced = reduce_function(
            stage,
            count,
            accumulator_input_name="acc0",
            input_name="x_seq",
            name="nonlinear_reduce",
            simplification="medium",
        )

        with TemporaryDirectory() as tmpdir:
            project = reduced.create_rust_project(Path(tmpdir) / "nonlinear_reduce_kernel")

            self._append_rust_test(
                project.project_dir,
                f"""
#[cfg(test)]
mod integration_sympy_reduce {{
    use super::*;

    #[test]
    fn generated_reduce_matches_sympy_reference() {{
        let acc0 = {self._rust_array_literal([acc0_value], "f64")};
        let x_seq = {self._rust_array_literal(x_values, "f64")};
        let expected = {self._rust_array_literal([expected], "f64")};

        let mut out = [0.0_f64; 1];
        let mut work = [0.0_f64; {project.codegen.workspace_size}];
        {project.codegen.function_name}(&acc0, &x_seq, &mut out, &mut work)
            .expect("reduce kernel evaluation failed");

        assert!((out[0] - expected[0]).abs() <= 1e-10_f64, "expected {{}} got {{}}", expected[0], out[0]);
    }}
}}
""".lstrip(),
            )

            completed = self._run_cargo(project.project_dir, "test", "--quiet")
            self.assertEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
