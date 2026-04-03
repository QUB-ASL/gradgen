#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradgenError {
    WorkspaceTooSmall(&'static str),
    InputTooSmall(&'static str),
    OutputTooSmall(&'static str),
}

fn norm2sq(values: &[f64]) -> f64 {
    values.iter().map(|value| *value * *value).sum()
}
/// Metadata describing a generated Rust function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionMetadata {
    /// Generated Rust function name.
    pub function_name: &'static str,
    /// Minimum required length of the mutable workspace slice.
    pub workspace_size: usize,
    /// Declared input names.
    pub input_names: &'static [&'static str],
    /// Declared input slice lengths.
    pub input_sizes: &'static [usize],
    /// Declared output names.
    pub output_names: &'static [&'static str],
    /// Declared output slice lengths.
    pub output_sizes: &'static [usize],
}

/// Return metadata describing [`energy`].
pub fn energy_meta() -> FunctionMetadata {
    FunctionMetadata {
        function_name: "energy",
        workspace_size: 2,
        input_names: &["x", "w"],
        input_sizes: &[2, 1],
        output_names: &["cost", "state"],
        output_sizes: &[1, 1],
    }
}

/// Evaluate the generated symbolic function `energy`.
///
/// All numeric slices use the `f64` scalar type.
///
/// Arguments:
/// - `x`:
///   input slice for the declared argument `x`
///   Expected length: 2.
/// - `w`:
///   input slice for the declared argument `w`
///   Expected length: 1.
/// - `cost`:
///   primal output slice for the declared result `cost`
///   Expected length: 1.
/// - `state`:
///   primal output slice for the declared result `state`
///   Expected length: 1.
/// - `work`: mutable workspace slice used to store intermediate values
///   while evaluating this kernel. Expected length: at least 2.
pub fn energy(
    x: &[f64],
    w: &[f64],
    cost: &mut [f64],
    state: &mut [f64],
    work: &mut [f64],
) -> Result<(), GradgenError> {
    if work.len() < 2 {
        return Err(GradgenError::WorkspaceTooSmall("work expected at least 2"));
    };
    if x.len() != 2 {
        return Err(GradgenError::InputTooSmall("x expected length 2"));
    };
    if w.len() != 1 {
        return Err(GradgenError::InputTooSmall("w expected length 1"));
    };
    if cost.len() != 1 {
        return Err(GradgenError::OutputTooSmall("cost expected length 1"));
    };
    if state.len() != 1 {
        return Err(GradgenError::OutputTooSmall("state expected length 1"));
    };
    work[0] = norm2sq(x);
    work[0] += w[0];
    work[1] = x[0] + x[1];
    cost[0] = work[0];
    state[0] = work[1];
    Ok(())
}

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyFloat, PyList, PyTuple};
use std::vec::Vec;

impl From<GradgenError> for PyErr {
    fn from(error: GradgenError) -> Self {
        match error {
            GradgenError::WorkspaceTooSmall(message)
            | GradgenError::InputTooSmall(message)
            | GradgenError::OutputTooSmall(message) => PyValueError::new_err(message),
        }
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
        "energy" => Ok(PyList::new(py, (0..2).map(|_| 0.0_f64))?
            .into_any()
            .unbind()),
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
    let expected_arg_count = 3;
    if inputs.len() != expected_arg_count {
        return Err(PyValueError::new_err(format!(
            "energy expected {expected_arg_count} arguments including workspace"
        )));
    }

    let x = extract_values(&inputs.get_item(0)?, 2, "x")?;
    let w = extract_values(&inputs.get_item(1)?, 1, "w")?;

    let workspace_values = extract_workspace(&inputs.get_item(2)?, 2)?;
    let mut workspace = [0.0_f64; 2];
    workspace.copy_from_slice(&workspace_values[..2]);

    let mut cost = [0.0_f64; 1];
    let mut state = [0.0_f64; 1];

    energy(
        &x[..],
        &w[..],
        &mut cost[..],
        &mut state[..],
        &mut workspace[..],
    )?;

    let result = PyDict::new(py);
    result.set_item("cost", wrap_output(py, &cost)?)?;
    result.set_item("state", wrap_output(py, &state)?)?;
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
