# A Safe Exploration Approach to Constrained Markov Decision Processes
[Project Page / Paper](https://proceedings.mlr.press/v258/ni25a.html)

## Overview

This codebase builds upon the framework introduced in  
**Identifiability and Generalizability in Constrained Inverse Reinforcement Learning**  
([arXiv:2306.00629](https://arxiv.org/pdf/2306.00629.pdf)).

It provides implementations of gridworld environments, core solvers for constrained Markov decision processes (CMDPs), visualization utilities, and several safe reinforcement learning algorithms for empirical evaluation.

---

## Code Structure

### `algs/` and `env/`

These directories define the gridworld environment and the core algorithmic components.

- The gridworld environment and transition dynamics follow the standard formulation in  
  **Sutton, R. S. and Barto, A. G. _Reinforcement Learning: An Introduction_. MIT Press, 2018.**
- The code includes utility functions for computing:
  - reward and constraint value functions,
  - Q-functions,
  - advantage functions.
- A linear programming (LP) solver is provided for computing optimal policies in CMDPs.

---

### `visualization/`

This directory contains utilities for visualizing learned policies in the gridworld environment, including state–action preferences and policy structure.

---

### `example/`

This directory contains implementations of several constrained reinforcement learning algorithms used for comparison:

1. **RPG-PD**  
   Ding, D., et al.  
   *Last-iterate convergent policy gradient primal-dual methods for constrained MDPs.*  
   NeurIPS, 2023.

2. **IPO**  
   Liu, Y., Halev, A., and Liu, X.  
   *Policy learning with constraints in model-free reinforcement learning: A survey.*  
   IJCAI, 2021.

3. **Log-Barrier**  
   Ni, T. and Kamgarpour, M.  
   *A safe exploration approach to constrained Markov decision processes.*  
   AISTATS, 2025.

---
