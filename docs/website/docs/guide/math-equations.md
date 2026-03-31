---
sidebar_position: 5
---

# Writing Math Equations

Gradgen documentation supports mathematical equations using LaTeX/KaTeX notation.

## Inline Math

Wrap equations in single `$` signs for inline math:

```md
The derivative of $f(x) = x^2$ is $f'(x) = 2x$.
```

Result: The derivative of $f(x) = x^2$ is $f'(x) = 2x$.

## Display Math (Block Equations)

Wrap equations in double `$$` signs on separate lines for display equations:

```md
$$
\frac{d}{dx}(x^2) = 2x
$$
```

Result:

$$
\frac{d}{dx}(x^2) = 2x
$$

## Common Mathematical Notation

### Derivatives

```md
$$
\frac{\partial f}{\partial x} = 2x + y
$$
```

$$
\frac{\partial f}{\partial x} = 2x + y
$$

### Integrals

```md
$$
\int_0^1 x^2 \, dx = \frac{1}{3}
$$
```

$$
\int_0^1 x^2 \, dx = \frac{1}{3}
$$

### Matrices

```md
$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$
```

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

### Greek Letters and Symbols

```md
$\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \eta, \theta$

$\sum, \prod, \int, \partial, \nabla, \forall, \exists$
```

Result: $\alpha, \beta, \gamma, \delta, \epsilon, \zeta, \eta, \theta$

$\sum, \prod, \int, \partial, \nabla, \forall, \exists$

### Exponents and Subscripts

```md
$x^2$, $x_i$, $x_{i,j}$, $x^{2^2}$
```

Result: $x^2$, $x_i$, $x_{i,j}$, $x^{2^2}$

### Fractions

```md
$\frac{a}{b}$, $\cfrac{a}{b}$
```

Result: $\frac{a}{b}$, $\cfrac{a}{b}$

### Roots

```md
$\sqrt{x}$, $\sqrt[3]{x}$
```

Result: $\sqrt{x}$, $\sqrt[3]{x}$

## Example: Gradgen AD Concepts

### Jacobian-Vector Product (JVP)

Forward-mode automatic differentiation computes the Jacobian-vector product:

$$
J_{\mathbf{f}}(\mathbf{x}) \cdot \mathbf{v} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \mathbf{v}
$$

### Vector-Jacobian Product (VJP)

Reverse-mode automatic differentiation computes the vector-Jacobian product:

$$
\mathbf{u}^\top \cdot J_{\mathbf{f}}(\mathbf{x}) = \mathbf{u}^\top \frac{\partial \mathbf{f}}{\partial \mathbf{x}}
$$

### Gradient

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the gradient is:

$$
\nabla f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

### Hessian

The Hessian matrix of second partial derivatives for a scalar function:

$$
H_f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

## Tips

- Use `\\` to create a new line in display equations
- Use `&=` for equation alignment in multi-line displays
- For complex equations, consider breaking them into multiple simpler equations
- Test your math rendering by building: `npm run build`

## Resources

- [KaTeX Documentation](https://katex.org/)
- [LaTeX/Math Mode Guide](https://en.wikibooks.org/wiki/LaTeX/Mathematics)
- [Docusaurus Math Support](https://docusaurus.io/docs/markdown-features/math-equations)
