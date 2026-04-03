---
sidebar_position: 7
---

# High-order functions

Gradgen supports high-order functions such as map, zip, and reduce.

## Map

For an integer $N$, the **map** function consumes one packed sequence

$$\mathbf{x} = (x^{(1)}, x^{(2)}, \dots, x^{(N)}),$$

and returns the outputs

$$\mathbf{y} = (u(x^{(1)}), u(x^{(2)}), \dots, u(x^{(N)})),$$

i.e., it applies $u$ to each of the vectors of the sequence.
In other words, it is the mapping $\mathrm{map}_{u, N}: \mathbf{x} \mapsto \mathbf{y}$. This operation is illustrated in the figure below.


<div align="center">
<img src="/gradgen/img/map.png" width="50%" />
</div>

## Zip

Similarly, for an integer $M$, the **zip** function consumes two packed sequences

$$\mathbf{a} = (a^{(1)}, \dots, a^{(M)}), \qquad
c_{\mathrm{seq}} = (c^{(1)}, \dots, c^{(M)}),$$

and returns

$$\mathbf{z} = (g(a^{(1)}, c^{(1)}), \dots, g(a^{(M)}, c^{(M)})).$$

In other words, it is the mapping $\mathrm{zip}_{b, M}: \mathbf{a} \mapsto \mathbf{b}.$ The zip operation is illustrated in the figure below.

<div align="center">
<img src="/gradgen/img/zip.png" width="50%" />
</div>

## Reduce

See the [demos](https://github.com/QUB-ASL/gradgen/tree/main/demos).

