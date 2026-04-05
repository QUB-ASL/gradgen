import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from gradgen.map_zip import (
    _arg_size,
    _slice_packed_input,
    _flatten_arg,
    _normalize_function_result,
    _coerce_function_arg_like,
    _coerce_scalar_output,
    ZippedFunction,
    map_function,
    zip_function,
)
from gradgen.function import Function
from gradgen.sx import SX, SXVector


class MapZipExtraTests(unittest.TestCase):
    def test_arg_size_slice_flatten(self) -> None:
        x = SX.sym("x")
        v = SXVector.sym("v", 3)

        self.assertEqual(_arg_size(x), 1)
        self.assertEqual(_arg_size(v), 3)

        seq = SXVector((SX.const(1.0), SX.const(2.0), SX.const(3.0), SX.const(4.0)))

        # scalar formal returns a single SX
        self.assertIsInstance(_slice_packed_input(seq, 1, x), SX)

        # vector formal returns an SXVector slice of the expected length
        formal_v = SXVector.sym("y", 2)
        sliced = _slice_packed_input(seq, 1, formal_v)
        self.assertIsInstance(sliced, SXVector)
        self.assertEqual(len(sliced), 2)

        # flattening behavior
        self.assertEqual(_flatten_arg(x), (x,))
        self.assertEqual(_flatten_arg(sliced), sliced.elements)

    def test_normalize_function_result_multi_output_error(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")

        # When the declared outputs expect multiple values but the result is not
        # a tuple-like value, the normalizer should raise.
        with self.assertRaises(ValueError):
            _normalize_function_result(1, (x, y))

    def test_coerce_function_arg_like_and_scalar_output_errors(self) -> None:
        x = SX.sym("x")
        formal_vec = SXVector.sym("v", 2)

        # scalar formal with a non-numeric value should raise
        with self.assertRaises(TypeError):
            _coerce_function_arg_like("not numeric", x)

        # vector formal with a correct SXVector should pass and preserve length
        val = SXVector((SX.const(1.0), SX.const(2.0)))
        coerced = _coerce_function_arg_like(val, formal_vec)
        self.assertIsInstance(coerced, SXVector)
        self.assertEqual(len(coerced), 2)

        # mismatched vector lengths raise
        with self.assertRaises(ValueError):
            _coerce_function_arg_like(SXVector((SX.const(1.0),)), formal_vec)

        # bad scalar outputs raise
        with self.assertRaises(TypeError):
            _coerce_scalar_output("bad")

    def test_normalize_and_coerce_tuple_mismatches(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")

        # tuple result with wrong length should raise the 'unexpected number' error
        with self.assertRaises(ValueError):
            _normalize_function_result((1,), (x, y))

        formal_vec = SXVector.sym("v", 2)
        # tuple/list with mismatched length raises in the vector branch
        with self.assertRaises(ValueError):
            _coerce_function_arg_like((1.0,), formal_vec)

        # non-sequence, non-SXVector values for vector formals raise TypeError
        with self.assertRaises(TypeError):
            _coerce_function_arg_like(1, formal_vec)

    def test_zipped_and_jacobian_rust_proxying_and_nodes(self) -> None:
        x = SX.sym("x")
        single = Function("g", [x], [x])
        zipped = zip_function(single, 2)

        # cover the nodes property which calls to_function().nodes
        nodes = zipped.nodes
        self.assertIsInstance(nodes, tuple)

        # Patch rust_codegen hooks to ensure generate/create proxies call through
        calls = {}

        def fake_generate_rust(func, **kwargs):
            calls.setdefault("generate", []).append((func, kwargs))
            return {"ok": True}

        def fake_create_rust_project(func, path, **kwargs):
            calls.setdefault("create", []).append((func, path, kwargs))
            return {"project": path}

        with mock.patch("gradgen._rust_codegen.codegen.generate_rust", fake_generate_rust), mock.patch(
            "gradgen._rust_codegen.project.create_rust_project", fake_create_rust_project
        ):
            # Call the ZippedFunction proxies
            gen_res = zipped.generate_rust(function_name="fn")
            self.assertEqual(gen_res, {"ok": True})

            # Call the ZippedJacobianFunction proxies
            zjac = zipped.jacobian(0)
            gen_res_j = zjac.generate_rust(function_name="fn_j")
            self.assertEqual(gen_res_j, {"ok": True})

            with TemporaryDirectory() as tmpdir:
                project_path = Path(tmpdir) / "p"
                proj_res = zipped.create_rust_project(project_path)
                self.assertEqual(proj_res, {"project": project_path})

            with TemporaryDirectory() as tmpdir:
                project_path = Path(tmpdir) / "p2"
                proj_res_j = zjac.create_rust_project(project_path)
                self.assertEqual(proj_res_j, {"project": project_path})

    def test_map_and_zip_validations_and_jacobian_index(self) -> None:
        x = SX.sym("x")
        y = SX.sym("y")

        f = Function("f", [x, y], [x + y])

        # map_function requires a unary function
        with self.assertRaises(ValueError):
            map_function(f, 3)

        # zip_function with defaults resolves input sequence names
        single = Function("g", [x], [x])
        zipped = zip_function(single, 2)
        self.assertTrue(zipped.input_sequence_names[0].endswith("_seq"))

        # ZippedFunction should validate input_sequence_names length (mismatch)
        with self.assertRaises(ValueError):
            ZippedFunction(function=f, count=2, name="bad", input_sequence_names=("only_one",))

        # count must be positive (map_function triggers ZippedFunction validation)
        with self.assertRaises(ValueError):
            map_function(single, 0)

        # jacobian out-of-range index raises
        mapped = map_function(single, 2)
        with self.assertRaises(IndexError):
            mapped.jacobian(5)
