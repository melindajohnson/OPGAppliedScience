"""
HW2 Question 2: Velocity Data Curve Fitting

Fit velocity (m/s) vs time (s) data with:
  f(t) = A*cos(B*t) + C*t + D
using least-squares optimization (FMIN).
"""

import numpy as np
from scipy.optimize import fmin

# Velocity data: time 0 to 30 seconds (31 data points)
t_data = np.arange(0, 31)

v_data = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41,
                    40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])

# Fine grid for evaluation: t = 0:0.01:30
t_eval = np.arange(0, 30.01, 0.01)

# ============================================================
# Part (a): Least-Squares fit f(t) = A*cos(B*t) + C*t + D
# ============================================================

# Define the E2 error function that fmin will minimize
def velocity_error(params):
    A, B, C, D = params
    v_pred = A * np.cos(B * t_data) + C * t_data + D
    return np.sum((v_data - v_pred)**2)

# Initial guess (given in the problem):
#   A = 3      
#   B = pi/4   
#   C = 2/3   
#   D = 32   
initial_guess = [3, np.pi/4, 2/3, 32]

# Run fmin to find best A, B, C, D
result = fmin(velocity_error, initial_guess, disp=True)
A_fit, B_fit, C_fit, D_fit = result

print()
print("Part (a): Cosine + Linear Fit f(t) = A*cos(B*t) + C*t + D")
print(f"A = {A_fit}")
print(f"B = {B_fit}")
print(f"C = {C_fit}")
print(f"D = {D_fit}")

# Calculate E2 error
v_predicted = A_fit * np.cos(B_fit * t_data) + C_fit * t_data + D_fit
E2 = np.sum((v_data - v_predicted)**2)
print(f"E2 error: {E2}")

# Evaluate curve for t = 0:0.01:30 and save as column vector
v_eval = (A_fit * np.cos(B_fit * t_eval) + C_fit * t_eval + D_fit).reshape(-1, 1)
print(f"Curve column vector shape: {v_eval.shape}")
