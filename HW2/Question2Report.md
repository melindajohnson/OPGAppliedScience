# HW2 Question 2: Velocity Data Curve Fitting — Report

## Problem

We are given 31 velocity readings (meters/second) measured over a 30-second interval at 1-second steps (t = 0 to 30). The goal is to fit the least-squares curve:

f(t) = A*cos(B*t) + C*t + D

and evaluate it for t = 0:0.01:30.

## Data

The velocity data ranges from 30 m/s at t=0 to 53 m/s at t=30, with an overall upward trend and periodic oscillations. 
The model f(t) = A*cos(B*t) + C*t + D  is a nonlinear model (B is inside the cosine), so it cannot be solved with linear algebra like polyfit. We use `scipy.optimize.fmin` to search for the best parameters.
Initial guesses are provided
- **A = 3**
- **B = pi/4 ≈ 0.785**
- **C = 2/3 ≈ 0.667**
- **D = 32**

## Optimization with FMIN

`fmin` takes the initial guess [3, 0.785, 0.667, 32] and iteratively adjusted all four parameters to minimize the E2 error. After 149 iterations and 260 function evaluations, it converged to:

- **A = 2.1717**
- **B = 0.9093**
- **C = 0.7325**
- **D = 31.4529**

Final equation: **f(t) = 2.1717*cos(0.9093*t) + 0.7325*t + 31.4529**

## E2 Error

E2 = sum((v_data[i] - f(t_data[i]))²) = **78.64**

## Curve Evaluation

The fitted curve was evaluated at t = 0, 0.01, 0.02, ..., 30.00 (3001 points) and saved as a (3001, 1) column vector.
