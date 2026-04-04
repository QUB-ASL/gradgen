import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gradgen import Function, SX, SXVector
from gradgen._rust_codegen.project import create_rust_project


class ProjectTests(unittest.TestCase):
    def test_create_rust_project_writes_expected_files(self) -> None:
        x = SXVector.sym("x", 1)
        function = Function(
            "demo",
            [x],
            [x[0] + 1.0],
            input_names=["x"],
            output_names=["y"],
        )
        with TemporaryDirectory() as tmpdir:
            project = create_rust_project(function, Path(tmpdir) / "demo")
            self.assertTrue(project.cargo_toml.exists())
            self.assertTrue(project.readme.exists())
            self.assertTrue(project.metadata_json.exists())
            self.assertTrue(project.lib_rs.exists())
