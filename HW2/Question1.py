"""
HW2 Question 1: Temperature Data Curve Fitting

Consider temperature data taken over a 24-hour (military time) cycle.
(a) Fit with parabola f(x) = Ax^2 + Bx + C, calculate E2 error
(b) Use INTERP1 and SPLINE to generate interpolated approximations
(c) Least-Squares fit for y = A*cos(B*x) + C using FMIN
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import fmin

# Temperature data: hour vs temperature
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

y_data = np.array([75, 77, 76, 73, 69, 68, 63, 59, 57, 55, 54, 52,
                    50, 50, 49, 49, 49, 50, 54, 56, 59, 63, 67, 72])

# Fine grid for evaluation: x = 1:0.01:24
x_eval = np.arange(1, 24.01, 0.01)

# ============================================================
# Part (a): Parabolic fit f(x) = Ax^2 + Bx + C
# ============================================================

# Fit parabola using polyfit (degree 2)
coeffs = np.polyfit(x_data, y_data, 2)
A, B, C = coeffs

print("Part (a): Parabolic Fit f(x) = Ax^2 + Bx + C")
print(f"A = {A}")
print(f"B = {B}")
print(f"C = {C}")

# Calculate E2 error at the original 24 data points
y_predicted = np.polyval(coeffs, x_data)
E2 = np.sum((y_data - y_predicted)**2)
print(f"E2 error: {E2}")

# Evaluate the curve for x = 1:0.01:24
y_eval = np.polyval(coeffs, x_eval)

# Save errors at the 24 data points as a column vector
errors = (y_data - y_predicted).reshape(-1, 1)
print(f"Error column vector shape: {errors.shape}")
print("Errors at each data point:")
for i in range(len(x_data)):
    print(f"  Hour {x_data[i]:2d}: error = {errors[i, 0]:.4f}")

# ============================================================
# Part (b): INTERP1 and SPLINE interpolation
# ============================================================

# Linear interpolation (INTERP1)
y_interp1 = np.interp(x_eval, x_data, y_data).reshape(-1, 1)

# Cubic spline interpolation (SPLINE)
cs = CubicSpline(x_data, y_data)
y_spline = cs(x_eval).reshape(-1, 1)

print()
print("Part (b): INTERP1 and SPLINE")
print(f"INTERP1 column vector shape: {y_interp1.shape}")
print(f"SPLINE column vector shape:  {y_spline.shape}")
print()
print("Sample values at x = 1, 5, 12.5, 18, 24:")
for x_sample in [1.0, 5.0, 12.5, 18.0, 24.0]:
    idx = int(round((x_sample - 1) / 0.01))
    print(f"  x={x_sample:5.1f}: INTERP1={y_interp1[idx, 0]:.4f}, SPLINE={y_spline[idx, 0]:.4f}")

# ============================================================
# Part (c): Least-Squares fit for y = A*cos(B*x) + C using FMIN
# ============================================================

# Define the E2 error function that fmin will minimize
def cosine_error(params):
    A, B, C = params
    y_pred = A * np.cos(B * x_data) + C
    return np.sum((y_data - y_pred)**2)

# Initial guess:
#   A ≈ (max - min) / 2 = (77 - 49) / 2 = 14  (amplitude)
#   B ≈ pi / 16 ≈ 0.196  (minimum at hour ~16, cos(Bx) = -1 when Bx = pi)
#   C ≈ (max + min) / 2 = (77 + 49) / 2 = 63  (vertical offset)
initial_guess = [14, 0.196, 63]

# Run fmin (Nelder-Mead optimization) to find best A, B, C
result = fmin(cosine_error, initial_guess, disp=True)
A_cos, B_cos, C_cos = result

print()
print("Part (c): Cosine Fit y = A*cos(B*x) + C")
print(f"A = {A_cos}")
print(f"B = {B_cos}")
print(f"C = {C_cos}")

# E2 error (A5)
y_cos_predicted = A_cos * np.cos(B_cos * x_data) + C_cos
E2_cos = np.sum((y_data - y_cos_predicted)**2)
print(f"E2 error (A5): {E2_cos}")

# Evaluate curve for x = 1:0.01:24 as column vector (A6)
y_cos_eval = (A_cos * np.cos(B_cos * x_eval) + C_cos).reshape(-1, 1)
print(f"Curve column vector shape (A6): {y_cos_eval.shape}")
