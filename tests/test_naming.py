import unittest

from gradgen._rust_codegen.naming import sanitize_ident, validate_rust_ident, validate_unique_rust_names


class NamingTests(unittest.TestCase):
    def test_sanitize_ident_replaces_invalid_characters(self) -> None:
        self.assertEqual(sanitize_ident("hello-world"), "hello_world")

    def test_validate_rust_ident_rejects_keywords(self) -> None:
        with self.assertRaises(ValueError):
            validate_rust_ident("fn", label="name")

    def test_validate_unique_rust_names_rejects_collisions(self) -> None:
        with self.assertRaises(ValueError):
            validate_unique_rust_names([("a", "same"), ("b", "same")], label="name")
