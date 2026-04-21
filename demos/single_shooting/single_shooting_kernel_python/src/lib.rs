use pyo3::exceptions::{PyAttributeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyFloat, PyList, PyTuple};
use std::vec::Vec;

use ::single_shooting_kernel as gradgen_low_level;

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
    Ok(PyList::new(
        py,
        [
            "mpc_cost_f_states",
            "mpc_cost_grad_states_u_seq",
            "mpc_cost_hvp_states_u_seq",
            "mpc_cost_f_grad_states_u_seq",
        ],
    )?
    .into_any()
    .unbind())
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
            "mpc_cost_f_states",
            "mpc_cost_grad_states_u_seq",
            "mpc_cost_hvp_states_u_seq",
            "mpc_cost_f_grad_states_u_seq",
        ],
    )?
    .into_any()
    .unbind())
}

#[pyfunction(name = "__getattr__")]
fn module_getattr(name: &str) -> PyResult<String> {
    match name {
        "__version__" => Ok("0.4.0".to_string()),
        _ => Err(PyAttributeError::new_err(format!(
            "module has no attribute {name:?}"
        ))),
    }
}

fn workspace_for_function_impl(py: Python<'_>, function_name: &str) -> PyResult<Py<Workspace>> {
    match function_name {
        "mpc_cost_f_states" | "single_shooting_kernel_mpc_cost_f_states" => {
            let mut values = Vec::with_capacity(8);
            values.resize(8, 0.0_f64);
            Py::new(
                py,
                Workspace {
                    function_name: "mpc_cost_f_states",
                    values,
                },
            )
        }
        "mpc_cost_grad_states_u_seq" | "single_shooting_kernel_mpc_cost_grad_states_u_seq" => {
            let mut values = Vec::with_capacity(14);
            values.resize(14, 0.0_f64);
            Py::new(
                py,
                Workspace {
                    function_name: "mpc_cost_grad_states_u_seq",
                    values,
                },
            )
        }
        "mpc_cost_hvp_states_u_seq" | "single_shooting_kernel_mpc_cost_hvp_states_u_seq" => {
            let mut values = Vec::with_capacity(32);
            values.resize(32, 0.0_f64);
            Py::new(
                py,
                Workspace {
                    function_name: "mpc_cost_hvp_states_u_seq",
                    values,
                },
            )
        }
        "mpc_cost_f_grad_states_u_seq" | "single_shooting_kernel_mpc_cost_f_grad_states_u_seq" => {
            let mut values = Vec::with_capacity(15);
            values.resize(15, 0.0_f64);
            Py::new(
                py,
                Workspace {
                    function_name: "mpc_cost_f_grad_states_u_seq",
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
        "mpc_cost_f_states" | "single_shooting_kernel_mpc_cost_f_states" => {
            let metadata = gradgen_low_level::single_shooting_kernel_mpc_cost_f_states_meta();
            build_function_info(py, "mpc_cost_f_states", metadata)
        }
        "mpc_cost_grad_states_u_seq" | "single_shooting_kernel_mpc_cost_grad_states_u_seq" => {
            let metadata =
                gradgen_low_level::single_shooting_kernel_mpc_cost_grad_states_u_seq_meta();
            build_function_info(py, "mpc_cost_grad_states_u_seq", metadata)
        }
        "mpc_cost_hvp_states_u_seq" | "single_shooting_kernel_mpc_cost_hvp_states_u_seq" => {
            let metadata =
                gradgen_low_level::single_shooting_kernel_mpc_cost_hvp_states_u_seq_meta();
            build_function_info(py, "mpc_cost_hvp_states_u_seq", metadata)
        }
        "mpc_cost_f_grad_states_u_seq" | "single_shooting_kernel_mpc_cost_f_grad_states_u_seq" => {
            let metadata =
                gradgen_low_level::single_shooting_kernel_mpc_cost_f_grad_states_u_seq_meta();
            build_function_info(py, "mpc_cost_f_grad_states_u_seq", metadata)
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
        "mpc_cost_f_states" | "single_shooting_kernel_mpc_cost_f_states" => {
            call_single_shooting_kernel_mpc_cost_f_states_from_tuple_impl(py, inputs)
        }
        "mpc_cost_grad_states_u_seq" | "single_shooting_kernel_mpc_cost_grad_states_u_seq" => {
            call_single_shooting_kernel_mpc_cost_grad_states_u_seq_from_tuple_impl(py, inputs)
        }
        "mpc_cost_hvp_states_u_seq" | "single_shooting_kernel_mpc_cost_hvp_states_u_seq" => {
            call_single_shooting_kernel_mpc_cost_hvp_states_u_seq_from_tuple_impl(py, inputs)
        }
        "mpc_cost_f_grad_states_u_seq" | "single_shooting_kernel_mpc_cost_f_grad_states_u_seq" => {
            call_single_shooting_kernel_mpc_cost_f_grad_states_u_seq_from_tuple_impl(py, inputs)
        }
        _ => Err(PyValueError::new_err(format!(
            "unknown generated function {function_name:?}"
        ))),
    }
}

fn call_single_shooting_kernel_mpc_cost_f_states_impl(
    py: Python<'_>,
    input_0: Vec<f64>,
    input_1: Vec<f64>,
    input_2: Vec<f64>,
    mut workspace: PyRefMut<'_, Workspace>,
) -> PyResult<Py<PyAny>> {
    let metadata = gradgen_low_level::single_shooting_kernel_mpc_cost_f_states_meta();
    if workspace.function_name != "mpc_cost_f_states"
        && workspace.function_name != metadata.function_name
    {
        return Err(PyValueError::new_err(format!(
            "workspace_for_function({:?}) must be used with {}",
            workspace.function_name, "mpc_cost_f_states"
        )));
    }
    if workspace.values.len() < metadata.workspace_size {
        return Err(PyValueError::new_err(format!(
            "workspace expected at least {}",
            metadata.workspace_size
        )));
    }

    let mut output_0 = [0.0_f64; 1];
    let mut output_1 = [0.0_f64; 12];

    gradgen_low_level::single_shooting_kernel_mpc_cost_f_states(
        &input_0[..],
        &input_1[..],
        &input_2[..],
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

fn call_single_shooting_kernel_mpc_cost_f_states_from_tuple_impl(
    py: Python<'_>,
    inputs: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let metadata = gradgen_low_level::single_shooting_kernel_mpc_cost_f_states_meta();
    let expected_arg_count = metadata.input_sizes.len() + 1;
    if inputs.len() != expected_arg_count {
        return Err(PyValueError::new_err(format!(
            "mpc_cost_f_states expected {expected_arg_count} arguments including workspace"
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
    let input_2 = extract_values(
        &inputs.get_item(2)?,
        metadata.input_sizes[2],
        metadata.input_names[2],
    )?;

    let workspace = inputs.get_item(3)?.extract::<PyRefMut<'_, Workspace>>()?;
    call_single_shooting_kernel_mpc_cost_f_states_impl(py, input_0, input_1, input_2, workspace)
}

#[pyfunction(name = "mpc_cost_f_states")]
fn py_single_shooting_kernel_mpc_cost_f_states(
    py: Python<'_>,
    arg_0: &Bound<'_, PyAny>,
    arg_1: &Bound<'_, PyAny>,
    arg_2: &Bound<'_, PyAny>,
    workspace: PyRefMut<'_, Workspace>,
) -> PyResult<Py<PyAny>> {
    let input_0 = extract_values(arg_0, 2, "x0")?;
    let input_1 = extract_values(arg_1, 5, "u_seq")?;
    let input_2 = extract_values(arg_2, 2, "p")?;
    call_single_shooting_kernel_mpc_cost_f_states_impl(py, input_0, input_1, input_2, workspace)
}

fn call_single_shooting_kernel_mpc_cost_grad_states_u_seq_impl(
    py: Python<'_>,
    input_0: Vec<f64>,
    input_1: Vec<f64>,
    input_2: Vec<f64>,
    mut workspace: PyRefMut<'_, Workspace>,
) -> PyResult<Py<PyAny>> {
    let metadata = gradgen_low_level::single_shooting_kernel_mpc_cost_grad_states_u_seq_meta();
    if workspace.function_name != "mpc_cost_grad_states_u_seq"
        && workspace.function_name != metadata.function_name
    {
        return Err(PyValueError::new_err(format!(
            "workspace_for_function({:?}) must be used with {}",
            workspace.function_name, "mpc_cost_grad_states_u_seq"
        )));
    }
    if workspace.values.len() < metadata.workspace_size {
        return Err(PyValueError::new_err(format!(
            "workspace expected at least {}",
            metadata.workspace_size
        )));
    }

    let mut output_0 = [0.0_f64; 5];
    let mut output_1 = [0.0_f64; 12];

    gradgen_low_level::single_shooting_kernel_mpc_cost_grad_states_u_seq(
        &input_0[..],
        &input_1[..],
        &input_2[..],
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

fn call_single_shooting_kernel_mpc_cost_grad_states_u_seq_from_tuple_impl(
    py: Python<'_>,
    inputs: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let metadata = gradgen_low_level::single_shooting_kernel_mpc_cost_grad_states_u_seq_meta();
    let expected_arg_count = metadata.input_sizes.len() + 1;
    if inputs.len() != expected_arg_count {
        return Err(PyValueError::new_err(format!(
            "mpc_cost_grad_states_u_seq expected {expected_arg_count} arguments including workspace"
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
    let input_2 = extract_values(
        &inputs.get_item(2)?,
        metadata.input_sizes[2],
        metadata.input_names[2],
    )?;

    let workspace = inputs.get_item(3)?.extract::<PyRefMut<'_, Workspace>>()?;
    call_single_shooting_kernel_mpc_cost_grad_states_u_seq_impl(
        py, input_0, input_1, input_2, workspace,
    )
}

#[pyfunction(name = "mpc_cost_grad_states_u_seq")]
fn py_single_shooting_kernel_mpc_cost_grad_states_u_seq(
    py: Python<'_>,
    arg_0: &Bound<'_, PyAny>,
    arg_1: &Bound<'_, PyAny>,
    arg_2: &Bound<'_, PyAny>,
    workspace: PyRefMut<'_, Workspace>,
) -> PyResult<Py<PyAny>> {
    let input_0 = extract_values(arg_0, 2, "x0")?;
    let input_1 = extract_values(arg_1, 5, "u_seq")?;
    let input_2 = extract_values(arg_2, 2, "p")?;
    call_single_shooting_kernel_mpc_cost_grad_states_u_seq_impl(
        py, input_0, input_1, input_2, workspace,
    )
}

fn call_single_shooting_kernel_mpc_cost_hvp_states_u_seq_impl(
    py: Python<'_>,
    input_0: Vec<f64>,
    input_1: Vec<f64>,
    input_2: Vec<f64>,
    input_3: Vec<f64>,
    mut workspace: PyRefMut<'_, Workspace>,
) -> PyResult<Py<PyAny>> {
    let metadata = gradgen_low_level::single_shooting_kernel_mpc_cost_hvp_states_u_seq_meta();
    if workspace.function_name != "mpc_cost_hvp_states_u_seq"
        && workspace.function_name != metadata.function_name
    {
        return Err(PyValueError::new_err(format!(
            "workspace_for_function({:?}) must be used with {}",
            workspace.function_name, "mpc_cost_hvp_states_u_seq"
        )));
    }
    if workspace.values.len() < metadata.workspace_size {
        return Err(PyValueError::new_err(format!(
            "workspace expected at least {}",
            metadata.workspace_size
        )));
    }

    let mut output_0 = [0.0_f64; 5];
    let mut output_1 = [0.0_f64; 12];

    gradgen_low_level::single_shooting_kernel_mpc_cost_hvp_states_u_seq(
        &input_0[..],
        &input_1[..],
        &input_2[..],
        &input_3[..],
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

fn call_single_shooting_kernel_mpc_cost_hvp_states_u_seq_from_tuple_impl(
    py: Python<'_>,
    inputs: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let metadata = gradgen_low_level::single_shooting_kernel_mpc_cost_hvp_states_u_seq_meta();
    let expected_arg_count = metadata.input_sizes.len() + 1;
    if inputs.len() != expected_arg_count {
        return Err(PyValueError::new_err(format!(
            "mpc_cost_hvp_states_u_seq expected {expected_arg_count} arguments including workspace"
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
    let input_2 = extract_values(
        &inputs.get_item(2)?,
        metadata.input_sizes[2],
        metadata.input_names[2],
    )?;
    let input_3 = extract_values(
        &inputs.get_item(3)?,
        metadata.input_sizes[3],
        metadata.input_names[3],
    )?;

    let workspace = inputs.get_item(4)?.extract::<PyRefMut<'_, Workspace>>()?;
    call_single_shooting_kernel_mpc_cost_hvp_states_u_seq_impl(
        py, input_0, input_1, input_2, input_3, workspace,
    )
}

#[pyfunction(name = "mpc_cost_hvp_states_u_seq")]
fn py_single_shooting_kernel_mpc_cost_hvp_states_u_seq(
    py: Python<'_>,
    arg_0: &Bound<'_, PyAny>,
    arg_1: &Bound<'_, PyAny>,
    arg_2: &Bound<'_, PyAny>,
    arg_3: &Bound<'_, PyAny>,
    workspace: PyRefMut<'_, Workspace>,
) -> PyResult<Py<PyAny>> {
    let input_0 = extract_values(arg_0, 2, "x0")?;
    let input_1 = extract_values(arg_1, 5, "u_seq")?;
    let input_2 = extract_values(arg_2, 2, "p")?;
    let input_3 = extract_values(arg_3, 5, "v_u_seq")?;
    call_single_shooting_kernel_mpc_cost_hvp_states_u_seq_impl(
        py, input_0, input_1, input_2, input_3, workspace,
    )
}

fn call_single_shooting_kernel_mpc_cost_f_grad_states_u_seq_impl(
    py: Python<'_>,
    input_0: Vec<f64>,
    input_1: Vec<f64>,
    input_2: Vec<f64>,
    mut workspace: PyRefMut<'_, Workspace>,
) -> PyResult<Py<PyAny>> {
    let metadata = gradgen_low_level::single_shooting_kernel_mpc_cost_f_grad_states_u_seq_meta();
    if workspace.function_name != "mpc_cost_f_grad_states_u_seq"
        && workspace.function_name != metadata.function_name
    {
        return Err(PyValueError::new_err(format!(
            "workspace_for_function({:?}) must be used with {}",
            workspace.function_name, "mpc_cost_f_grad_states_u_seq"
        )));
    }
    if workspace.values.len() < metadata.workspace_size {
        return Err(PyValueError::new_err(format!(
            "workspace expected at least {}",
            metadata.workspace_size
        )));
    }

    let mut output_0 = [0.0_f64; 1];
    let mut output_1 = [0.0_f64; 5];
    let mut output_2 = [0.0_f64; 12];

    gradgen_low_level::single_shooting_kernel_mpc_cost_f_grad_states_u_seq(
        &input_0[..],
        &input_1[..],
        &input_2[..],
        &mut output_0[..],
        &mut output_1[..],
        &mut output_2[..],
        workspace.values.as_mut_slice(),
    )
    .map_err(pyerr_from_gradgen_error)?;

    let result = PyDict::new(py);
    result.set_item(metadata.output_names[0], wrap_output(py, &output_0)?)?;
    result.set_item(metadata.output_names[1], wrap_output(py, &output_1)?)?;
    result.set_item(metadata.output_names[2], wrap_output(py, &output_2)?)?;
    Ok(result.into_any().unbind())
}

fn call_single_shooting_kernel_mpc_cost_f_grad_states_u_seq_from_tuple_impl(
    py: Python<'_>,
    inputs: &Bound<'_, PyTuple>,
) -> PyResult<Py<PyAny>> {
    let metadata = gradgen_low_level::single_shooting_kernel_mpc_cost_f_grad_states_u_seq_meta();
    let expected_arg_count = metadata.input_sizes.len() + 1;
    if inputs.len() != expected_arg_count {
        return Err(PyValueError::new_err(format!(
            "mpc_cost_f_grad_states_u_seq expected {expected_arg_count} arguments including workspace"
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
    let input_2 = extract_values(
        &inputs.get_item(2)?,
        metadata.input_sizes[2],
        metadata.input_names[2],
    )?;

    let workspace = inputs.get_item(3)?.extract::<PyRefMut<'_, Workspace>>()?;
    call_single_shooting_kernel_mpc_cost_f_grad_states_u_seq_impl(
        py, input_0, input_1, input_2, workspace,
    )
}

#[pyfunction(name = "mpc_cost_f_grad_states_u_seq")]
fn py_single_shooting_kernel_mpc_cost_f_grad_states_u_seq(
    py: Python<'_>,
    arg_0: &Bound<'_, PyAny>,
    arg_1: &Bound<'_, PyAny>,
    arg_2: &Bound<'_, PyAny>,
    workspace: PyRefMut<'_, Workspace>,
) -> PyResult<Py<PyAny>> {
    let input_0 = extract_values(arg_0, 2, "x0")?;
    let input_1 = extract_values(arg_1, 5, "u_seq")?;
    let input_2 = extract_values(arg_2, 2, "p")?;
    call_single_shooting_kernel_mpc_cost_f_grad_states_u_seq_impl(
        py, input_0, input_1, input_2, workspace,
    )
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
#[pyo3(name = "single_shooting_kernel")]
fn gradgen_python_interface(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Workspace>()?;
    m.add_function(wrap_pyfunction!(all_functions, m)?)?;
    m.add_function(wrap_pyfunction!(function_info, m)?)?;
    m.add_function(wrap_pyfunction!(module_getattr, m)?)?;
    m.add_function(wrap_pyfunction!(workspace_for_function, m)?)?;
    m.add_function(wrap_pyfunction!(call, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_single_shooting_kernel_mpc_cost_f_states,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_single_shooting_kernel_mpc_cost_grad_states_u_seq,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_single_shooting_kernel_mpc_cost_hvp_states_u_seq,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_single_shooting_kernel_mpc_cost_f_grad_states_u_seq,
        m
    )?)?;
    m.dict().set_item("__all__", module_all_impl(_py)?)?;
    Ok(())
}
