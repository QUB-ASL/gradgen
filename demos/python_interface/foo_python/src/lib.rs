use pyo3::exceptions::{PyAttributeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyFloat, PyList, PyTuple};
use std::vec::Vec;

use ::foo as gradgen_low_level;

#[pyclass]
struct Workspace {
    function_name: &'static str,
    values: Vec<f64>,
}

#[pymethods]
impl Workspace {
    #[getter]
    fn function_name(&self) -> &str {
        self.function_name
    }

    fn __len__(&self) -> usize {
        self.values.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Workspace(function_name={:?}, size={})",
            self.function_name,
            self.values.len()
        )
    }
}

fn pyerr_from_gradgen_error(error: gradgen_low_level::GradgenError) -> PyErr {
    match error {
        gradgen_low_level::GradgenError::WorkspaceTooSmall(message)
        | gradgen_low_level::GradgenError::InputTooSmall(message)
        | gradgen_low_level::GradgenError::OutputTooSmall(message) => {
            PyValueError::new_err(message)
        }
    }
}

fn build_function_info(
    py: Python<'_>,
    python_name: &str,
    metadata: gradgen_low_level::FunctionMetadata,
) -> PyResult<Py<PyAny>> {
    let info = PyDict::new(py);
    info.set_item("name", python_name)?;
    info.set_item("rust_name", metadata.function_name)?;
    info.set_item("workspace_size", metadata.workspace_size)?;
    info.set_item(
        "input_names",
        PyList::new(py, metadata.input_names.iter().copied())?,
    )?;
    info.set_item(
        "input_sizes",
        PyList::new(py, metadata.input_sizes.iter().copied())?,
    )?;
    info.set_item(
        "output_names",
        PyList::new(py, metadata.output_names.iter().copied())?,
    )?;
    info.set_item(
        "output_sizes",
        PyList::new(py, metadata.output_sizes.iter().copied())?,
    )?;
    Ok(info.into_any().unbind())
}

fn extract_values(obj: &Bound<'_, PyAny>, expected_size: usize, label: &str) -> PyResult<Vec<f64>> {
    if expected_size == 0 {
        return Ok(Vec::new());
    }

    if expected_size == 1 {
        if let Ok(value) = obj.extract::<f64>() {
            return Ok(std::iter::once(value).collect());
        }
    }

    let values: Vec<f64> = obj.extract()?;
    if values.len() != expected_size {
        return Err(PyValueError::new_err(format!(
            "{label} expected length {expected_size}"
        )));
    }
    Ok(values.into_iter().collect())
}

fn wrap_output(py: Python<'_>, values: &[f64]) -> PyResult<Py<PyAny>> {
    if values.len() == 1 {
        return Ok(PyFloat::new(py, values[0]).into_any().unbind());
    }

    Ok(PyList::new(py, values.iter().copied())?.into_any().unbind())
}

fn all_functions_impl(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(PyList::new(py, ["energy"])?.into_any().unbind())
}

fn module_all_impl(py: Python<'_>) -> PyResult<Py<PyAny>> {
    Ok(PyList::new(
        py,
        [
            "__version__",
            "Workspace",
            "all_functions",
            "function_info",
            "workspace_for_function",
            "call",
            "energy",
        ],
    )?
    .into_any()
    .unbind())
}

#[pyfunction(name = "__getattr__")]
fn module_getattr(name: &str) -> PyResult<String> {
    match name {
        "__version__" => Ok("0.5.0".to_string()),
        _ => Err(PyAttributeError::new_err(format!(
            "module has no attribute {name:?}"
        ))),
    }
}

fn workspace_for_function_impl(py: Python<'_>, function_name: &str) -> PyResult<Py<Workspace>> {
    match function_name {
        "energy" => {
            let mut values = Vec::with_capacity(2);
            values.resize(2, 0.0_f64);
            Py::new(
                py,
                Workspace {
                    function_name: "energy",
                    values,
                },
            )
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown generated function {function_name:?}"
        ))),
    }
}

fn function_info_impl(py: Python<'_>, function_name: &str) -> PyResult<Py<PyAny>> {
    match function_name {
        "energy" => {
            let metadata = gradgen_low_level::energy_meta();
            build_function_info(py, "energy", metadata)
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown generated function {function_name:?}"
        ))),
    }
}

fn call_impl(
    py: Python<'_>,
    function_name: &str,
    inputs: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    match function_name {
        "energy" => call_energy_from_tuple_impl(py, inputs),
        _ => Err(PyValueError::new_err(format!(
            "unknown generated function {function_name:?}"
        ))),
    }
}

fn call_energy_impl(
    py: Python<'_>,
    input_0: Vec<f64>,
    input_1: Vec<f64>,
    mut workspace: PyRefMut<'_, Workspace>,
) -> PyResult<Py<PyAny>> {
    let metadata = gradgen_low_level::energy_meta();
    if workspace.function_name != "energy" && workspace.function_name != metadata.function_name {
        return Err(PyValueError::new_err(format!(
            "workspace_for_function({:?}) must be used with {}",
            workspace.function_name, "energy"
        )));
    }
    if workspace.values.len() < metadata.workspace_size {
        return Err(PyValueError::new_err(format!(
            "workspace expected at least {}",
            metadata.workspace_size
        )));
    }

    let mut output_0 = [0.0_f64; 1];
    let mut output_1 = [0.0_f64; 1];

    gradgen_low_level::energy(
        &input_0[..],
        &input_1[..],
        &mut output_0[..],
        &mut output_1[..],
        workspace.values.as_mut_slice(),
    )
    .map_err(pyerr_from_gradgen_error)?;

    let result = PyDict::new(py);
    result.set_item(metadata.output_names[0], wrap_output(py, &output_0)?)?;
    result.set_item(metadata.output_names[1], wrap_output(py, &output_1)?)?;
    Ok(result.into_any().unbind())
}

fn call_energy_from_tuple_impl(py: Python<'_>, inputs: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let metadata = gradgen_low_level::energy_meta();
    let expected_arg_count = metadata.input_sizes.len() + 1;
    if inputs.len() != expected_arg_count {
        return Err(PyValueError::new_err(format!(
            "energy expected {expected_arg_count} arguments including workspace"
        )));
    }

    let input_0 = extract_values(
        &inputs.get_item(0)?,
        metadata.input_sizes[0],
        metadata.input_names[0],
    )?;
    let input_1 = extract_values(
        &inputs.get_item(1)?,
        metadata.input_sizes[1],
        metadata.input_names[1],
    )?;

    let workspace = inputs.get_item(2)?.extract::<PyRefMut<'_, Workspace>>()?;
    call_energy_impl(py, input_0, input_1, workspace)
}

#[pyfunction(name = "energy")]
fn py_energy(
    py: Python<'_>,
    arg_0: &Bound<'_, PyAny>,
    arg_1: &Bound<'_, PyAny>,
    workspace: PyRefMut<'_, Workspace>,
) -> PyResult<Py<PyAny>> {
    let input_0 = extract_values(arg_0, 2, "x")?;
    let input_1 = extract_values(arg_1, 1, "w")?;
    call_energy_impl(py, input_0, input_1, workspace)
}

#[pyfunction]
fn all_functions(py: Python<'_>) -> PyResult<Py<PyAny>> {
    all_functions_impl(py)
}

#[pyfunction]
fn workspace_for_function(py: Python<'_>, function_name: &str) -> PyResult<Py<Workspace>> {
    workspace_for_function_impl(py, function_name)
}

#[pyfunction]
fn function_info(py: Python<'_>, function_name: &str) -> PyResult<Py<PyAny>> {
    function_info_impl(py, function_name)
}

#[pyfunction(signature = (function_name, *inputs))]
fn call(py: Python<'_>, function_name: &str, inputs: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    call_impl(py, function_name, inputs)
}

#[pymodule]
#[pyo3(name = "foo")]
fn gradgen_python_interface(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Workspace>()?;
    m.add_function(wrap_pyfunction!(all_functions, m)?)?;
    m.add_function(wrap_pyfunction!(function_info, m)?)?;
    m.add_function(wrap_pyfunction!(module_getattr, m)?)?;
    m.add_function(wrap_pyfunction!(workspace_for_function, m)?)?;
    m.add_function(wrap_pyfunction!(call, m)?)?;
    m.add_function(wrap_pyfunction!(py_energy, m)?)?;
    m.dict().set_item("__all__", module_all_impl(_py)?)?;
    Ok(())
}
