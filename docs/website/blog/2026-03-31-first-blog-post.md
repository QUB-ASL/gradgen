---
slug: first-blog-post
title: Gradgen revival
authors: [pantelis]
tags: [release]
---

Welcome to this brand new website of gradgen: your Python module for automatic differentiation and truly embedded Rust code generation.

Gradgen v0.3 brings two significant changes:

1. It has been redesigned to generate **Rust crates**, optionally marked with `#[no_std]` so you can use them directly in your embedded applications
2. It comes with specialised code generation mechanisms for **optimal control** leading to Rust code that **does not increase in size** with the prediction horizon

<!-- truncate -->

We are already working on gradgen v0.3 which will bring one more significant change: support for high-order functions like **map**, **zip**, and **reduce**, which will improve code generation even more. 

Our goal is to exploit the structure of a function, its gradient, and Hessian to generate more efficient, elegant, and fast Rust code.
