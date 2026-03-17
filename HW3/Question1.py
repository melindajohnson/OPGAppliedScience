"""
HW3: Feature Spaces — Yale Face Database

(a) Compute 100x100 correlation matrix from first 100 images, plot with pcolor
(b) Find most/least correlated face pairs, plot them
(c) Compute 10x10 correlation matrix for specific images
(d) Compute Y = XX^T, find first 6 eigenvectors with largest eigenvalues
(e) Compute SVD of X, find first 6 principal component directions
(f) Compare eigenvector v1 from (d) with SVD mode u1 from (e)
(g) Compute variance captured by first 6 SVD modes
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load the Yale Faces data
results = loadmat('HW3/yalefaces.mat')
X = results['X']
print(f"Matrix X shape: {X.shape}")  # should be 1024 x 2414

# ============================================================
# Part (a): 100x100 Correlation Matrix
# ============================================================

# Take the first 100 face images (columns)
X100 = X[:, :100]

# Compute correlation matrix: C[j,k] = dot product of column j and column k
# X100.T @ X100 computes all dot products at once
C = X100.T @ X100
print(f"Correlation matrix C shape: {C.shape}")  # should be 100 x 100

# Plot the correlation matrix using pcolor
plt.figure(figsize=(8, 6))
plt.pcolormesh(C)
plt.colorbar(label='Correlation')
plt.title('100x100 Correlation Matrix (First 100 Faces)')
plt.xlabel('Face Index')
plt.ylabel('Face Index')
plt.savefig('HW3/part_a_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Part (a) plot saved: HW3/part_a_correlation_matrix.png")

# ============================================================
# Part (b): Most and Least Correlated Faces
# ============================================================

# Copy C and set diagonal to NaN so we ignore self-comparisons
C_offdiag = C.copy()
np.fill_diagonal(C_offdiag, np.nan)

# Most correlated pair (max off-diagonal value)
max_val = np.nanmax(C_offdiag)
max_idx = np.unravel_index(np.nanargmax(C_offdiag), C_offdiag.shape)
print(f"\nMost correlated: Face {max_idx[0]+1} and Face {max_idx[1]+1}, correlation = {max_val:.4f}")

# Most uncorrelated pair (min off-diagonal value)
min_val = np.nanmin(C_offdiag)
min_idx = np.unravel_index(np.nanargmin(C_offdiag), C_offdiag.shape)
print(f"Most uncorrelated: Face {min_idx[0]+1} and Face {min_idx[1]+1}, correlation = {min_val:.4f}")

# Plot the faces
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# Most correlated pair
axes[0, 0].imshow(X[:, max_idx[0]].reshape(32, 32), cmap='gray')
axes[0, 0].set_title(f'Face {max_idx[0]+1} (Most Correlated)')
axes[0, 0].axis('off')

axes[0, 1].imshow(X[:, max_idx[1]].reshape(32, 32), cmap='gray')
axes[0, 1].set_title(f'Face {max_idx[1]+1} (Most Correlated)')
axes[0, 1].axis('off')

# Most uncorrelated pair
axes[1, 0].imshow(X[:, min_idx[0]].reshape(32, 32), cmap='gray')
axes[1, 0].set_title(f'Face {min_idx[0]+1} (Most Uncorrelated)')
axes[1, 0].axis('off')

axes[1, 1].imshow(X[:, min_idx[1]].reshape(32, 32), cmap='gray')
axes[1, 1].set_title(f'Face {min_idx[1]+1} (Most Uncorrelated)')
axes[1, 1].axis('off')

plt.suptitle('Part (b): Most and Least Correlated Face Pairs')
plt.tight_layout()
plt.savefig('HW3/part_b_face_pairs.png', dpi=150, bbox_inches='tight')
plt.close()
print("Part (b) plot saved: HW3/part_b_face_pairs.png")

# ============================================================
# Part (c): 10x10 Correlation Matrix for specific images
# ============================================================

# Specific images (1-based indexing from the problem)
image_labels = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]

# Convert to 0-based indexing for Python
image_indices = [i - 1 for i in image_labels]

# Extract those 10 columns from X
X10 = X[:, image_indices]

# Compute 10x10 correlation matrix
C10 = X10.T @ X10
print(f"\nPart (c): 10x10 Correlation matrix shape: {C10.shape}")
print("Correlation matrix:")
print(C10)

# Plot the 10x10 correlation matrix
plt.figure(figsize=(8, 6))
plt.pcolormesh(C10)
plt.colorbar(label='Correlation')
plt.xticks(np.arange(10) + 0.5, image_labels, rotation=45)
plt.yticks(np.arange(10) + 0.5, image_labels)
plt.title('10x10 Correlation Matrix (Selected Images)')
plt.xlabel('Image')
plt.ylabel('Image')
plt.savefig('HW3/part_c_correlation_matrix_10x10.png', dpi=150, bbox_inches='tight')
plt.close()
print("Part (c) plot saved: HW3/part_c_correlation_matrix_10x10.png")

# ============================================================
# Part (d): Eigenvectors of Y = XX^T
# ============================================================

# Compute Y = X @ X^T (1024 x 1024 matrix)
Y = X @ X.T
print(f"\nPart (d): Y = XX^T shape: {Y.shape}")

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(Y)

# eigh returns eigenvalues in ascending order, so we reverse to get largest first
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]

# Take the first 6 eigenvectors (largest eigenvalues)
top6_eigenvalues = eigenvalues[:6]
top6_eigenvectors = eigenvectors[:, :6]

print("First 6 eigenvalues:")
for i in range(6):
    print(f"  lambda_{i+1} = {top6_eigenvalues[i]:.4f}")

# Plot the first 6 eigenvectors as 32x32 images (eigenfaces)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i in range(6):
    row, col = i // 3, i % 3
    axes[row, col].imshow(top6_eigenvectors[:, i].reshape(32, 32), cmap='gray')
    axes[row, col].set_title(f'Eigenvector {i+1} (λ={top6_eigenvalues[i]:.1f})')
    axes[row, col].axis('off')

plt.suptitle('Part (d): First 6 Eigenvectors of Y = XX^T (Eigenfaces)')
plt.tight_layout()
plt.savefig('HW3/part_d_eigenfaces.png', dpi=150, bbox_inches='tight')
plt.close()
print("Part (d) plot saved: HW3/part_d_eigenfaces.png")

# ============================================================
# Part (e): SVD of X — First 6 Principal Component Directions
# ============================================================

# Compute SVD: X = U @ diag(S) @ Vt
# full_matrices=False gives the "economy" SVD (faster, smaller)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
print(f"\nPart (e): SVD of X")
print(f"U shape: {U.shape}")       # 1024 x 1024
print(f"S shape: {S.shape}")       # 1024 singular values
print(f"Vt shape: {Vt.shape}")     # 1024 x 2414

# First 6 columns of U are the first 6 principal component directions
top6_U = U[:, :6]

print("First 6 singular values:")
for i in range(6):
    print(f"  sigma_{i+1} = {S[i]:.4f}")

# Plot the first 6 SVD modes as 32x32 images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i in range(6):
    row, col = i // 3, i % 3
    axes[row, col].imshow(top6_U[:, i].reshape(32, 32), cmap='gray')
    axes[row, col].set_title(f'SVD Mode {i+1} (σ={S[i]:.1f})')
    axes[row, col].axis('off')

plt.suptitle('Part (e): First 6 SVD Modes (Principal Component Directions)')
plt.tight_layout()
plt.savefig('HW3/part_e_svd_modes.png', dpi=150, bbox_inches='tight')
plt.close()
print("Part (e) plot saved: HW3/part_e_svd_modes.png")

# ============================================================
# Part (f): Compare v1 (eigenvector) with u1 (SVD mode)
# ============================================================

v1 = top6_eigenvectors[:, 0]  # First eigenvector from part (d)
u1 = top6_U[:, 0]             # First SVD mode from part (e)

# Compute norm of difference (and norm of sum, in case of sign flip)
norm_diff = np.linalg.norm(v1 - u1)
norm_sum = np.linalg.norm(v1 + u1)

print(f"\nPart (f): Comparing v1 (eigenvector) with u1 (SVD mode)")
print(f"  ||v1 - u1|| = {norm_diff:.6f}")
print(f"  ||v1 + u1|| = {norm_sum:.6f}")

if norm_diff < 0.01:
    print("  Result: v1 and u1 are essentially identical")
elif norm_sum < 0.01:
    print("  Result: v1 and u1 are identical but with opposite signs (v1 = -u1)")
else:
    print("  Result: v1 and u1 differ slightly due to numerical precision")

# ============================================================
# Part (g): Variance Captured by First 6 SVD Modes
# ============================================================

# Variance captured by each mode = sigma_i^2 / sum(all sigma^2)
total_variance = np.sum(S**2)
variance_each = S[:6]**2 / total_variance * 100  # percentage

print(f"\nPart (g): Variance captured by first 6 SVD modes")
print(f"Total variance (sum of all sigma^2): {total_variance:.2f}")
cumulative = 0
for i in range(6):
    cumulative += variance_each[i]
    print(f"  Mode {i+1}: {variance_each[i]:.2f}% (cumulative: {cumulative:.2f}%)")

# Plot variance captured
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart of individual variance
ax1.bar(range(1, 7), variance_each)
ax1.set_xlabel('SVD Mode')
ax1.set_ylabel('Variance Captured (%)')
ax1.set_title('Individual Variance per Mode')
ax1.set_xticks(range(1, 7))

# Cumulative variance
cumulative_variance = np.cumsum(variance_each)
ax2.plot(range(1, 7), cumulative_variance, 'o-', linewidth=2)
ax2.set_xlabel('Number of SVD Modes')
ax2.set_ylabel('Cumulative Variance (%)')
ax2.set_title('Cumulative Variance Captured')
ax2.set_xticks(range(1, 7))
ax2.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90%')
ax2.legend()

plt.suptitle('Part (g): Variance Captured by First 6 SVD Modes')
plt.tight_layout()
plt.savefig('HW3/part_g_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Part (g) plot saved: HW3/part_g_variance.png")
