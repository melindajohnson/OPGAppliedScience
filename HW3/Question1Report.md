# HW3: Feature Spaces — Yale Face Database

## Overview

This homework explores face recognition concepts using the Yale Face Database. The dataset contains 2,414 face images, each 32x32 pixels, stored as columns in a matrix X (1024 x 2414). We use correlation matrices, eigenvalue decomposition, and SVD to analyze the structure of the face data and discover that it can be efficiently represented in a low-dimensional feature space.

---

## Part (a): 100x100 Correlation Matrix

We take the first 100 face images and compute their correlation matrix using C = X100^T @ X100. Each entry C[j,k] is the dot product between face j and face k — a measure of how similar they are.

Here is how the matrix multiplication works visually:

```
 X100^T (faces as rows)     @     X100 (faces as columns)

  ┌─── Face 1 ───┐              │F1│F2│...│F100│
  ├─── Face 2 ───┤              │  │  │   │    │
  ├─── Face 3 ───┤      @       │  │  │   │    │
  │     ...      │              │  │  │   │    │
  └─── Face100 ──┘              │  │  │   │    │

  = 100x100 matrix where C[j,k] = Face j dot Face k
```

Each row of X100^T is a face (1024 pixels), and each column of X100 is also a face. When we multiply row j by column k, we get the dot product between Face j and Face k. This gives us the full 100x100 correlation matrix in one operation.

The correlation matrix is plotted using pcolormesh. Bright spots indicate high correlation (similar faces), and dark spots indicate low correlation (dissimilar faces). The bright diagonal confirms that every face is most similar to itself. Bright bands or blocks indicate groups of faces that look alike (likely the same person or similar lighting).

## Part (b): Most and Least Correlated Face Pairs

From the 100x100 correlation matrix, we find:

- **Most correlated pair**: Face 87 and Face 89, correlation = 260.78
  - These faces look nearly identical — likely the same person with similar lighting
- **Most uncorrelated pair**: Face 55 and Face 65, correlation = 0.002
  - These faces are almost orthogonal — their pixel patterns have virtually no overlap. One is likely a very dark image.

We set the diagonal to NaN before searching, so we don't accidentally pick a face compared to itself.

## Part (c): 10x10 Correlation Matrix for Selected Images

We compute a 10x10 correlation matrix for images [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]. This smaller matrix lets us inspect specific face pairs. The brightest off-diagonal entry is between images 87 and 5 (correlation = 150.81), indicating strong similarity. Image 512 has low correlation with most others, suggesting it looks quite different from the rest.

## Part (d): Eigenfaces — Eigenvectors of Y = XX^T

### What are we doing and why?

In parts (a)-(c), we compared faces using dot products. But we were comparing individual faces to each other. Now we ask a bigger question: **what are the most important patterns across ALL 2,414 faces?**

Think of it like this: if you had to describe every face using only 6 "building block" images that you mix and match, what would those 6 building blocks be? That's what eigenvectors give us.

### Step by step

**Step 1: Compute Y = X @ X^T**

This is different from parts (a)-(c) where we did X^T @ X. The difference matters:

```
X^T @ X = (2414 x 1024) @ (1024 x 2414) = 2414 x 2414  (face vs face)
X @ X^T = (1024 x 2414) @ (2414 x 1024) = 1024 x 1024  (pixel vs pixel)
```

X^T @ X tells us how faces relate to each other. X @ X^T tells us how **pixels** relate to each other — which pixel patterns tend to appear together across all faces. That's what we want here.

**Step 2: Find eigenvectors and eigenvalues of Y**

An eigenvector v of Y satisfies: Y @ v = lambda * v. This means when you multiply the matrix Y by v, you get the same vector v back, just scaled by lambda. The eigenvector is a direction that the matrix doesn't rotate — it only stretches.

We use `np.linalg.eigh(Y)` which is optimized for symmetric matrices (Y = XX^T is always symmetric). It returns eigenvalues in ascending order, so we reverse them to get largest first.

**Step 3: The eigenvalues tell us importance**

The eigenvalue lambda tells you how much of the total variance in the data is "explained" by that eigenvector. A large eigenvalue means that pattern appears strongly across many faces.

**First 6 eigenvalues:**

| Eigenvector | Eigenvalue | What it means |
|-------------|------------|---------------|
| 1 | 234,020.5 | Dominant pattern — explains most of the data |
| 2 | 49,038.3 | Second most important pattern |
| 3 | 8,236.5 | Already much smaller — diminishing returns |
| 4 | 6,024.9 | |
| 5 | 2,051.5 | |
| 6 | 1,901.1 | |

The sharp drop (234k -> 49k -> 8k) means the first few eigenvectors carry almost all the information. The rest are mostly noise.

**Step 4: Reshape eigenvectors into images (eigenfaces)**

Each eigenvector is a 1024-length vector. Since our images are 32x32 = 1024 pixels, we can reshape each eigenvector back into a 32x32 image to visualize it:

- **Eigenface 1**: Looks like a blurry "average face" — bright forehead, dark eye sockets. This is the most common pattern shared by all faces. Makes sense: most faces have eyes, nose, and mouth in roughly the same places.
- **Eigenface 2**: Shows a left-right lighting gradient. This captures the biggest source of *difference* between faces after removing the average — some photos are lit from the left, some from the right.
- **Eigenfaces 3-6**: Capture progressively finer details — vertical lighting, specific facial features, asymmetries. Each one explains less and less.

### The key insight: faces as weighted combinations

Any face in the dataset can be approximated as a weighted sum of these eigenfaces:

```
face ≈ w1 * eigenface_1 + w2 * eigenface_2 + ... + w6 * eigenface_6
```

Each face gets its own unique set of weights [w1, w2, w3, w4, w5, w6]. Those 6 numbers become the face's coordinates in a 6-dimensional "feature space." Instead of comparing 1,024 pixel values, we can compare just 6 weights — a massive simplification.

### What are principal components?

The eigenfaces ARE the principal components. "Principal component" is just another name for these optimal building-block patterns. They're called "principal" because they capture the most variance possible. No other set of 6 patterns could explain more of the data than the top 6 eigenvectors.

They are also "features" — but not human-designed features like "nose size" or "eye color." They're mathematically optimal features that the algorithm discovered on its own. Each one is a whole 32x32 image pattern, not a single measurement.

## Part (e): SVD of X — Principal Component Directions

### What is SVD?

SVD (Singular Value Decomposition) is a way to break any matrix into three pieces:

```
X = U @ S @ V^T
```

Where:
- **U** (1024 x 1024): columns are directions in "pixel space" — the same eigenfaces from part (d)
- **S** (diagonal, 1024 values): singular values — how important each direction is
- **V^T** (1024 x 2414): how each face is expressed in terms of U's directions

### How SVD relates to part (d)

Part (d) was a two-step process:
1. Compute Y = X @ X^T
2. Find eigenvectors of Y

SVD does this in one step — it decomposes X directly without computing XX^T first. Mathematically:

```
If X = U @ S @ V^T, then:
XX^T = U @ S @ S^T @ U^T
```

This means:
- The columns of U = the eigenvectors of XX^T (same as part d)
- The singular values squared = the eigenvalues of XX^T

```
sigma^2 = lambda

483.76^2 = 234,022  ≈  eigenvalue 234,020 from part (d)
221.45^2 =  49,038  ≈  eigenvalue  49,038 from part (d)
```

### Why use SVD if eigenvalue decomposition gives the same answer?

Two reasons:
1. **Numerical stability**: SVD works directly on X and is more numerically stable than computing XX^T first (which can amplify rounding errors)
2. **You get V^T for free**: SVD gives you both U (pixel patterns) and V^T (how faces combine those patterns), while eigenvalue decomposition of XX^T only gives you U

**First 6 singular values:**

| SVD Mode | Singular Value (sigma) | sigma^2 (= eigenvalue) |
|----------|----------------------|----------------------|
| 1 | 483.76 | 234,022 |
| 2 | 221.45 | 49,038 |
| 3 | 90.76 | 8,237 |
| 4 | 77.62 | 6,025 |
| 5 | 45.29 | 2,052 |
| 6 | 43.60 | 1,901 |

The 6 SVD mode images look identical to the 6 eigenfaces from part (d), confirming this relationship.

## Part (f): Comparing v1 (Eigenvector) with u1 (SVD Mode)

### What we're checking

Part (d) gave us eigenvector v1. Part (e) gave us SVD mode u1. They should be the same vector. We verify this by computing the norm (length) of their difference:

```
||v1 - u1|| = sqrt(sum of (v1[i] - u1[i])^2 for all i))
```

If this is zero, the vectors are identical.

### Results

- **||v1 - u1|| = 0.000000** — they are exactly the same vector
- **||v1 + u1|| = 2.000000** — this is what you'd expect for two identical unit vectors (length 1 + length 1 = 2)

This confirms that eigenvalue decomposition and SVD are two roads to the same destination.

### Why we also check ||v1 + u1||

Eigenvectors have a quirk: if v is an eigenvector, then -v is also a valid eigenvector (flipping all signs doesn't change the direction, just which way it points). Different algorithms might return v or -v. If that happens:
- ||v1 - u1|| would be ~2 (far apart because one points up, one points down)
- ||v1 + u1|| would be ~0 (they cancel out because v1 = -u1)

In our case this didn't happen — both methods returned the same sign.

## Part (g): Variance Captured by First 6 SVD Modes

### What is "variance captured"?

Variance measures how spread out the data is. Each SVD mode captures some portion of the total spread. The formula is:

```
variance captured by mode i = sigma_i^2 / (sum of ALL sigma^2) * 100%
```

We square the singular values because variance is measured in squared units (just like in HW2 where we used E2 = sum of squared errors).

### Results

| Mode | Individual Variance | Cumulative Variance | What it captures |
|------|-------------------|-------------------|------------------|
| 1 | 72.93% | 72.93% | Average face structure |
| 2 | 15.28% | 88.21% | Left-right lighting |
| 3 | 2.57% | 90.78% | Vertical lighting |
| 4 | 1.88% | 92.65% | Facial structure details |
| 5 | 0.64% | 93.29% | Finer details |
| 6 | 0.59% | 93.89% | Even finer details |

### What this means practically

- **Mode 1 alone captures 73%**: Most faces look roughly similar (eyes, nose, mouth in similar places). The "average face" pattern is the single most informative feature.
- **2 modes get 88%**: The biggest way faces differ from the average is lighting direction. Adding this one extra feature almost doubles the useful information.
- **3 modes cross 90%**: With just 3 numbers per face, you capture 90% of the information that was in 1,024 pixels.
- **6 modes reach 94%**: Diminishing returns — each additional mode adds less and less.

### The big picture: dimensionality reduction

This is the core idea of PCA (Principal Component Analysis):

```
Original: each face = 1,024 numbers (one per pixel)
Reduced:  each face = 6 numbers (one weight per eigenface)
Lost:     only 6% of the information
```

This is powerful because:
1. **Storage**: 6 numbers vs 1,024 — that's 170x compression
2. **Speed**: Comparing 6 numbers is much faster than comparing 1,024
3. **Noise removal**: The discarded 6% is mostly noise, so the 6-number representation can actually be *better* for tasks like face recognition

---

## Summary

| Part | Method | Key Result |
|------|--------|------------|
| (a) | X^T @ X correlation matrix | 100x100 matrix showing face similarities |
| (b) | Max/min off-diagonal search | Most similar: Face 87 & 89 (260.78), Most different: Face 55 & 65 (0.002) |
| (c) | Subset correlation matrix | 10x10 matrix for selected images |
| (d) | Eigenvalue decomposition of XX^T | 6 eigenfaces with eigenvalues from 234,020 to 1,901 |
| (e) | SVD of X | 6 SVD modes with singular values from 483.8 to 43.6 |
| (f) | Vector norm comparison | ||v1 - u1|| = 0, confirming equivalence |
| (g) | Variance analysis | 6 modes capture 93.9% of total variance |
