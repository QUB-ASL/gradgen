
<h1 align="center">
 <img alt="cgapp logo" src="https://i.postimg.cc/G3M2szz5/Logo-Makr-4z-HKa0.png" width="224px"/><br/>


![PyPI - Downloads](https://img.shields.io/pypi/dm/gradgen?color=blue&style=flat-square)

</h1>

**GradGen** can determine the derivative   
(or gradient, or Jacobian) of a given cost function in optimization problems，reducing calculation time.  
### Table of contents  
[Features](#features)

[Installation instructions](#installation-instructions)

[Code Generation Example](#code-generation-example)

[Core team](#core-team)

### Features  
**GradGen** is based on sequential backward-in-time algorithm. You can see more information in this [paper]().

**GradGen** is ideal for accelerating computation process in these problems:
- Nonlinear programming problem in Nonlinear Model Predictive - Control (NMPC)
- Gradient method in optimization problem
- Other control, machine learning, and engineering application



### Installation instructions  
To build this project locally:  
- Install `Rust`. You can find install instructions [here](https://www.rust-lang.org).
- Create a virtual environment: `virtualenv -p python3 venv`  
- Activate the virtual environment. On Linux/MacOS, run `source venv/bin/activate`  
- Install the project:  `pip install gradgen `  
  

### Code Generation Example
Here is a simple example of calculating gradient of total cost function in ball-and-beam model. (read the [docs]() for details)

 - First you should define your system dynamics and total cost function in Python.


```python
import os
import unittest
import casadi.casadi as cs
from gradgen.cost_gradient import *
import logging
import numpy as np
import subprocess
  
logger = logging.getLogger(__name__)  
  
  
class GradgenTestCase(unittest.TestCase):  
  
    @classmethod  
  def create_example(cls):  
        nx, nu = 4, 1  
  m, I, g, ts= 1, 0.0005,9.81,0.01  
  
  x = cs.SX.sym('x', nx)  
        u = cs.SX.sym('u', nu)  
  
        # System dynamics, f  
  f = cs.vertcat(  
            x[0] + ts * x[1],  
  x[1] + ts * ((5/7)*x[0]*x[3]**2-g * cs.sin(x[2])),  
  x[2] + ts * x[3],  
  x[3] + ts * ((u[0] - m * g * x[0] * cs.cos(x[2]) - 2 * m * x[0] * x[1] * x[2] ) / (m*x[0]**2+I)))  
  
        # Stage cost function, ell  
  ell = 5*x[0]**2 + 0.01*x[1]**2 + 0.01*x[2]**2 + 0.05*x[3]**2 + 2.2*u**2  
  
  # terminal cost function, vf  
  vf = 0.5 * (x[0]**2 + 50 * x[1]**2 + 100 * x[2]**2)  
        return x, u, f, ell, vf  
  
    def test_generate_code_and_build(self):  
        x, u, f, ell, vf = GradgenTestCase.create_example()  
        N = 15  
  gradObj = CostGradient(x, u, f, ell, vf, N).with_name(  
            "ball_test").with_target_path(".")  
        gradObj.build(no_rust_build=True)  
  
  
if __name__ == '__main__':  
    logger.setLevel(logging.ERROR)  
    unittest.main()

```

 - Then create a Rust file.



```Rust
fn main() {  
  
 # Define initial variables
 let n_pred = 5;   
 let mut ws = ball_test::BackwardGradientWorkspace::new(n_pred);  
 let nu =ball_test::num_inputs();  
 let nx = ball_test::num_states();  
 let x0 = vec![0.0; nx];  
 let mut u_seq = vec! [1.0; nu * n_pred];  
 let mut grad = vec! [0.0; nu * n_pred];  
    
 # Print gradient 
 ball_test::total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws, n_pred);  
 println!("{:?}",grad);  
}
```
>**Note that:** you should keep the consistence of test file name.


## Core Team

<table>
  <tbody>
    <tr>
      <td align="center" valign="top">
        <img width="150" height="150" src="https://github.com/alphaville.png?s=100">
        <br>
        <a href="https://alphaville.github.io">Pantelis Sopasakis</a> 
      </td>
      <td align="center" valign="top">
        <img width="150" height="150" src="https://i.postimg.cc/m2Q3Qtpq/IMG-3356.jpg">
        <br>
        <a href="https://github.com/inLimonL">Jie Lin</a>     
      </td>
     </tr>
  </tbody>
</table>