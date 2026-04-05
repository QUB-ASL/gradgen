"""Tests for the custom elementary-function registry."""

from __future__ import annotations

import unittest

from gradgen._custom_elementary import registry
from gradgen._custom_elementary.model import RegisteredElementaryFunction


class CustomRegistryTests(unittest.TestCase):
    def tearDown(self) -> None:
        registry.clear_registered_elementary_functions()

    def _register_scalar_spec(self, name: str = "square_shift") -> RegisteredElementaryFunction:
        return registry.register_elementary_function(
            name=name,
            input_dimension=1,
            parameter_dimension=1,
            parameter_defaults=[1.0],
            eval_python=lambda x, w: x * x + w[0],
            jacobian=lambda x, w: 2 * x,
            hessian=lambda x, w: 2.0,
            hvp=lambda x, v, w: 2 * v,
        )

    def test_register_returns_retrievable_spec(self) -> None:
        spec = self._register_scalar_spec()

        self.assertEqual(spec.name, "square_shift")
        self.assertEqual(spec.input_dimension, 1)
        self.assertEqual(spec.parameter_dimension, 1)
        self.assertEqual(spec.parameter_defaults, (1.0,))
        self.assertTrue(spec.is_scalar)
        self.assertIs(registry.get_registered_elementary_function("square_shift"), spec)

    def test_register_rejects_duplicate_names(self) -> None:
        self._register_scalar_spec()

        with self.assertRaises(ValueError):
            self._register_scalar_spec()

    def test_clear_removes_registered_spec(self) -> None:
        self._register_scalar_spec()

        registry.clear_registered_elementary_functions()

        with self.assertRaises(KeyError):
            registry.get_registered_elementary_function("square_shift")

    def test_register_validates_name_and_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            registry.register_elementary_function(
                name="not a valid name",
                input_dimension=1,
                eval_python=lambda x, w: x,
                jacobian=lambda x, w: x,
                hessian=lambda x, w: x,
            )

        with self.assertRaises(ValueError):
            registry.register_elementary_function(
                name="negative_input",
                input_dimension=0,
                eval_python=lambda x, w: x,
                jacobian=lambda x, w: x,
                hessian=lambda x, w: x,
            )

        with self.assertRaises(ValueError):
            registry.register_elementary_function(
                name="bad_defaults",
                input_dimension=1,
                parameter_dimension=2,
                parameter_defaults=[1.0],
                eval_python=lambda x, w: x,
                jacobian=lambda x, w: x,
                hessian=lambda x, w: x,
            )
