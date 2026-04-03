use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyFloat, PyList, PyTuple};
use std::vec::Vec;

fn pyerr_from_gradgen_error(error: foo::GradgenError) -> PyErr {
    match error {
        foo::GradgenError::WorkspaceTooSmall(message)
        | foo::GradgenError::InputTooSmall(message)
        | foo::GradgenError::OutputTooSmall(message) => PyValueError::new_err(message),
    }
}

fn extract_values(obj: &Bound<'_, PyAny>, expected_size: usize, label: &str) -> PyResult<Vec<f64>> {
    if expected_size == 0 {
        return Ok(Vec::new());
    }

    if expected_size == 1 {
        if let Ok(value) = obj.extract::<f64>() {
            let mut values = Vec::with_capacity(1);
            values.push(value as f64);
            return Ok(values);
        }
    }

    let values: Vec<f64> = obj.extract()?;
    if values.len() != expected_size {
        return Err(PyValueError::new_err(format!(
            "{label} expected length {expected_size}"
        )));
    }
    Ok(values.into_iter().map(|value| value as f64).collect())
}

fn extract_workspace(obj: &Bound<'_, PyAny>, expected_size: usize) -> PyResult<Vec<f64>> {
    if expected_size == 0 {
        return Ok(Vec::new());
    }

    if expected_size == 1 {
        if let Ok(value) = obj.extract::<f64>() {
            let mut values = Vec::with_capacity(1);
            values.push(value as f64);
            return Ok(values);
        }
    }

    let values: Vec<f64> = obj.extract()?;
    if values.len() < expected_size {
        return Err(PyValueError::new_err(format!(
            "workspace expected at least {expected_size}"
        )));
    }
    Ok(values
        .into_iter()
        .take(expected_size)
        .map(|value| value as f64)
        .collect())
}

fn wrap_output(py: Python<'_>, values: &[f64]) -> PyResult<Py<PyAny>> {
    if values.len() == 1 {
        return Ok(PyFloat::new(py, values[0] as f64).into_any().unbind());
    }

    Ok(
        PyList::new(py, values.iter().copied().map(|value| value as f64))?
            .into_any()
            .unbind(),
    )
}

fn workspace_for_function_impl(py: Python<'_>, function_name: &str) -> PyResult<Py<PyAny>> {
    match function_name {
        "energy" => {
            let metadata = foo::energy_meta();
            Ok(
                PyList::new(py, (0..metadata.workspace_size).map(|_| 0.0_f64))?
                    .into_any()
                    .unbind(),
            )
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
        "energy" => call_energy(py, inputs),
        _ => Err(PyValueError::new_err(format!(
            "unknown generated function {function_name:?}"
        ))),
    }
}

fn call_energy(py: Python<'_>, inputs: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    let metadata = foo::energy_meta();
    let expected_arg_count = metadata.input_sizes.len() + 1;
    if inputs.len() != expected_arg_count {
        return Err(PyValueError::new_err(format!(
            "energy expected {expected_arg_count} arguments including workspace"
        )));
    }

    let x = extract_values(
        &inputs.get_item(0)?,
        metadata.input_sizes[0],
        metadata.input_names[0],
    )?;
    let w = extract_values(
        &inputs.get_item(1)?,
        metadata.input_sizes[1],
        metadata.input_names[1],
    )?;

    let workspace_values = extract_workspace(&inputs.get_item(2)?, metadata.workspace_size)?;
    let mut workspace = [0.0_f64; 2];
    workspace.copy_from_slice(&workspace_values[..2]);

    let mut cost = [0.0_f64; 1];
    let mut state = [0.0_f64; 1];

    foo::energy(
        &x[..],
        &w[..],
        &mut cost[..],
        &mut state[..],
        &mut workspace[..],
    )
    .map_err(pyerr_from_gradgen_error)?;

    let result = PyDict::new(py);
    result.set_item(metadata.output_names[0], wrap_output(py, &cost)?)?;
    result.set_item(metadata.output_names[1], wrap_output(py, &state)?)?;
    Ok(result.into_any().unbind())
}

#[pyfunction]
fn workspace_for_function(py: Python<'_>, function_name: &str) -> PyResult<Py<PyAny>> {
    workspace_for_function_impl(py, function_name)
}

#[pyfunction(signature = (function_name, *inputs))]
fn call(py: Python<'_>, function_name: &str, inputs: &Bound<'_, PyTuple>) -> PyResult<Py<PyAny>> {
    call_impl(py, function_name, inputs)
}

#[pymodule]
#[pyo3(name = "foo")]
fn gradgen_python_interface(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(workspace_for_function, m)?)?;
    m.add_function(wrap_pyfunction!(call, m)?)?;
    Ok(())
}
