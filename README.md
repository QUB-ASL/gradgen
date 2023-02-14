
  
    
<div align="center">    
 <img alt="cgapp logo" src="https://i.postimg.cc/G3M2szz5/Logo-Makr-4z-HKa0.png" width="224px"/><br/>    
    
    
![PyPI - Downloads](https://img.shields.io/pypi/dm/gradgen?color=blue&style=flat-square)    
    
</div>    
    
**GradGen** can determine the gradient of the total cost function of an optimal control problem with respect to the sequence of control actions.  
  
In particular, consider the following optimal control problem  
  
$$\begin{aligned}  
\operatorname*{Minimise}_{u_0, u_1, \ldots, u_{N-1}}&\sum_{t=0}^{N-1}\ell(x_t, u_t) + V_f(x_N)\\  
\text { subject to: } x_{t+1}& =f\left(x_t, u_t\right), t=0, \ldots, N-1\\  
x_0& =x.  
\end{aligned}$$  
  
Let $u=(u_0, u_1, \ldots, u_{N-1})$. The state of the system at time $t$ starting from the initial state $x_0=x$ and with the action of these control actions is $x_t = \phi(t; x, u)$. Then, the total cost function is defined as   
  
$$V_N(u) = \sum_{t=0}^{N-1}\ell(x_t, u_t) + V_f(x_N).$$  
  
**Gradgen** is a Python module that generates Rust code that can be used to compute $\nabla V_N(u)$.  
  
  
### Table of contents      
  
- [Features](#features)  
- [Installation instructions](#installation-instructions)    
- [Code Generation Example](#code-generation-example)    
- [Core team](#core-team)    
    
    
### Features   
**GradGen** is based on the sequential backward-in-time algorithm. You can see more information in this [paper]().    
    
**GradGen** is ideal for computing the     
  
- Nonlinear programming problem in Nonlinear Model Predictive Control (NMPC)    
- Gradient-based numerical optimisation methods     
- Other control, machine learning, and engineering applications    
    
    
    
### Installation instructions   
To install gradgen:      
- Install `Rust`. You can find install instructions [here](https://www.rust-lang.org).    
- Create a virtual environment: `virtualenv -p python3 venv` - Activate the virtual environment. On Linux/MacOS, run `source venv/bin/activate` - Install the project:  `pip install gradgen `      
 ### Code Generation Example   
Here is a simple example of calculating gradient of total cost function in ball-and-beam model. (read the [docs]() for details)    
[![bnp.png](https://i.postimg.cc/ydfNFBYQ/bnp.png)](https://postimg.cc/Q9Ts3271)  
Consider A ball of mass m is placed on a beam which is poised on a fulcrum at its middle. We can control the system by applying a torque $u$ with respect to the fulcrum point. The moment of inertia of the beam is denoted by $I$. The displacement $x$ of the ball from the midpoint can be measured with an optical sensor. The dynamical system is described by the following nonlinear differential equations  
  
$$
\begin{align}  
\dot{x}_1 {}={}& x_2  
\\  
\dot{x}_2 {}={}& \tfrac{5}{7}(x_1x_4^2 - g\sin(x_3))  
\\  
\dot{x}_3 {}={}& x_4  
\\  
\dot{x}_4 {}={}& \frac{u - mgx_1\cos(x_3) - 2mx_1x_2x_3}{m x_1^2 + I}  
\end{align}  
$$
  
where $x_1=x$, $x_2=\dot{x}$, $x_3=\theta$, $x_4 = \dot{\theta}$.   
We can discretise this system with the Euler method with sampling time $T_s$. This yields

$$
\begin{align}  
x_{1, t+1} {}={}& x_{1, t} + T_s x_{2, t}  
\\  
x_{2, t+1} {}={}& x_{2, t} + T_s \tfrac{5}{7}(x_{1, t}x_{4, t}^2 - g\sin(x_{3,t}))  
\\  
x_{3,t+1} {}={}& x_{3,t} + T_s x_{4,t}  
\\  
x_{4,t+1} {}={}& x_{4, t} + T_s \frac{u_t - mg x_{1,t}\cos(x_{3,t}) - 2 m x_{1,t} x_{2,t} x_{3,t}}{m x_{1,t}^2 + I}  
\end{align}  
$$

The above dynamical system has four states and one input and can be written in the form $x_{t+1} = f(x_t, u_t)$, where $f:\mathbb{R}^4 \times \mathbb{R} \to \mathbb{R}^4$ is defined by

$$
\begin{align}  
    f(x, u)  
    {}={}  
    \begin{bmatrix}  
    x_{1} + T_s x_{2}  
    \\  
    x_{2} + T_s \tfrac{5}{7}(x_{1}x_{4}^2 - g\sin(x_{3}))  
    \\  
     x_{3} + T_s x_{4}  
     \\  
     x_{4} + T_s \frac{u - mg x_{1}\cos(x_{3}) - 2 m x_{1} x_{2} x_{3}}{m x_{1}^2 + I}  
    \end{bmatrix}.  
\end{align} 
$$

In the simulations, we can use the numerical values $m = 1$, $I = 0.0005$, $g = 9.81$, $T_s = 0.01$.  
We define state cost function:

$$   
\begin{align}  
\ell =  5*{x_{0}}^2  +  0.01*{x_{1}}^2 +  0.01*{x_{2}}^2 +  0.05*{x_{3}}^2 +  2.2*u^2,
\end{align} 
$$

and terminal cost function:

$$
\begin{align}  
V_f = 0.5 * ({x_{0}}^2+ 50*{x_{1}}^2+ 100 *{x_{2}}^2).    
\end{align} 
$$

  
  
 First, you should define your system dynamics and total cost function in Python.    
    
    
```python 
import casadi.casadi as cs 
from gradgen.cost_gradient import * 
import numpy as np 

N = 15   
nx, nu = 4, 1      
m, I, g, ts= 1, 0.0005,9.81,0.01      
      
x = cs.SX.sym('x', nx)      
u = cs.SX.sym('u', nu)      
      
# System dynamics, f      
f = cs.vertcat(      
      x[0] + ts * x[1],      
      x[1] + ts * ((5/7)*x[0]*x[3]**2-g * cs.sin(x[2])),      
      x[2] + ts * x[3],      
      x[3] + ts * ((u[0] - m * g * x[0] * cs.cos(x[2]) - 2 * m * x[0] * x[1] * x[2] ) / (m*x[0]**2+I))
      )      
      
# Stage cost function, ell      
ell = 5*x[0]**2 + 0.01*x[1]**2 + 0.01*x[2]**2 + 0.05*x[3]**2 + 2.2*u**2      
      
# terminal cost function, vf      
vf = 0.5 * (x[0]**2 + 50 * x[1]**2 + 100 * x[2]**2)        
 
gradObj = CostGradient(x, u, f, ell, vf, N).with_name(      
            "ball_and_beam").with_target_path(".")      
gradObj.build(no_rust_build=True)      
``` 
Above Python codes automatically generate the Rust interface, which include Jacobian of $f$ with respects to state variable $x$, $f_{x}(x, u)$, Jacobian of $\ell$ with respects to state variable $x$, $\ell_{x}(x, u)$, and other functions we needed to generate gradient by sequential backward-in-time method.
You can simply call these functions to realize Rust implementation.
An example is shown below. 
    
```rust 
fn main() {       
 # Define initial variables    
 let n_pred = 5; 
 let mut ws = ball_and_beam::BackwardGradientWorkspace::new(n_pred);      
 let nu = ball_and_beam::num_inputs();      
 let nx = ball_and_beam:num_states();      
 let x0 = vec![0.0; nx];      
 let mut u_seq = vec! [1.0; nu * n_pred];      
 let mut grad = vec! [0.0; nu * n_pred];      
        
 # Print gradient     
 ball_and_beam::total_cost_gradient_bw(&x0, &u_seq, &mut grad, &mut ws, n_pred); 
 println!("{:?}",grad); 
} 
```
 >**Note that:** you should keep the consistency of test file name. For example, in the first step, we create a folder called 'ball_and_beam'. Backward gradient workspace is stored in this folder. So every time you use 'total_cost_gradient_bw' to call workspace structure, you should find it from 'ball_and_beam' folder.
    
    
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