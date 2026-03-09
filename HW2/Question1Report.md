# HW2 Question 1: Temperature Data Curve Fitting — Report

## Problem

We are given 24 temperature readings taken over a military-time cycle (hours 1 through 24). The goal is to:
- (a) Fit the data with a parabolic curve and calculate the E2 error
- (b) Generate interpolated approximations using INTERP1 and SPLINE
- (c) Fit the data with a cosine model using least-squares optimization (FMIN)

## Data

The temperature data was stored as two NumPy arrays. The independent variable `x_data` represents the hours 1 through 24, and `y_data` contains the corresponding temperatures in degrees Fahrenheit. Temperatures range from a high of 77°F at hour 2 down to a low of 49°F at hours 15-17, then rise again to 72°F at hour 24 — forming a U-shaped pattern.

---

## Part (a): Parabolic Fit f(x) = Ax² + Bx + C

### Fitting with POLYFIT

`np.polyfit(x_data, y_data, 2)` performs a least-squares regression with a degree-2 polynomial. This function finds the coefficients A, B, and C that minimize the sum of squared differences between the actual data and the fitted curve. The resulting coefficients are:

- **A = 0.1849** 
- **B = -5.2642**
- **C = 88.2955**

This gives us the equation: **f(x) = 0.1849x² - 5.2642x + 88.2955**

### E2 Error

The E2 error is the sum of squared errors at all 24 original data points:

E2 = Σ (y_data[i] - f(x_data[i]))² = **172.36**

### Curve Evaluation

Using `np.arange(1, 24.01, 0.01)`, we created 2301 evenly spaced x-values from 1 to 24 and evaluated the parabola at each point using `np.polyval`. The errors at the 24 data points were saved as a (24, 1) column vector. The largest errors occur at hour 1 (-8.22°F) and hour 6 (+4.63°F), while the best fit is at hour 12 (+0.24°F).

---

## Part (b): INTERP1 and SPLINE Interpolation

### Linear Interpolation (INTERP1)

`np.interp` done to perform linear interpolation — drawing straight lines between each pair of consecutive data points. The result passes exactly through every data point, with straight-line segments connecting them. The output is a (2301, 1) column vector.

### Cubic Spline Interpolation (SPLINE)

We used `scipy.interpolate.CubicSpline` to fit smooth cubic polynomials between each pair of data points. Unlike linear interpolation, the spline produces a smooth curve with no sharp corners. The output is also a (2301, 1) column vector.

### Key Difference

Both methods pass exactly through all 24 data points (unlike polyfit which approximates). The difference is what happens between points — INTERP1 uses straight lines while SPLINE uses smooth curves. For example, at x = 12.5, INTERP1 gives 51.0°F (midpoint of neighbors) while SPLINE gives 50.79°F (smooth curve estimate).

---

## Part (c): Cosine Fit y = A*cos(B*x) + C

### Why FMIN is Needed

The cosine model is nonlinear in parameter B (B is inside the cosine function), so it cannot be solved directly with linear algebra like polyfit. Instead, we use `scipy.optimize.fmin` to search for the best A, B, C by iteratively minimizing the E2 error.

### Initial Guess

The initial guess is critical for fmin to find the correct solution. A bad guess can cause fmin to get stuck in a local minimum (a wrong answer that it can't escape from). We derived our guess by analyzing the data:

**A ≈ 14 (amplitude):**
The cosine function oscillates between +A and -A, so the total swing is 2A. In our data, the highest temperature is 77°F (hour 2) and the lowest is 49°F (hours 15-17). The total swing is 77 - 49 = 28°F. Therefore:
```
2A = 28  →  A = 28/2 = 14
```

**C ≈ 63 (vertical offset):**
The cosine wave oscillates equally above and below its center. The center of our data is the average of the max and min:
```
C = (max + min) / 2 = (77 + 49) / 2 = 63
```
This means the temperature swings between 63 + 14 = 77 and 63 - 14 = 49, which matches our data.

**B ≈ 0.196 (frequency):**
This is the trickiest parameter. B controls how fast the cosine oscillates. We know that cos(x) hits its minimum value (-1) when x = π. In our model, the input to cosine is B*x, so:
```
cos(B*x) = -1  when  B*x = π
```
The minimum temperature occurs around hour 16, so:
```
B * 16 = π
B = π / 16 ≈ 3.14159 / 16 ≈ 0.196
```
This means the cosine completes half a cycle in 16 hours (from max at hour ~0 to min at hour ~16), giving a full period of ~32 hours.

### Optimized Result

After 66 iterations and 119 function evaluations, fmin converged to:

- **A = 14.6122**
- **B = 0.2145**
- **C = 62.9877**

This gives us the equation: **y = 14.6122 * cos(0.2145x) + 62.9877**

### E2 Error (A5)

E2 = Σ (y_data[i] - f(x_data[i]))² = **37.58**

### Curve (A6)

The cosine curve was evaluated at x = 1:0.01:24 and saved as a (2301, 1) column vector.

---

## Comparison of Methods

| Method | E2 Error | Passes Through Data? |
|--------|----------|---------------------|
| Parabola (a) | 172.36 | No (approximation) |
| INTERP1 (b) | 0 | Yes (interpolation) |
| SPLINE (b) | 0 | Yes (interpolation) |
| Cosine (c) | 37.58 | No (approximation) |

The cosine model (E2 = 37.58) fits the data significantly better than the parabola (E2 = 172.36) because temperature over a 24-hour cycle naturally follows a cosine-like wave pattern. INTERP1 and SPLINE have zero error at data points since they pass exactly through them, but they don't provide a compact equation describing the data.
