---
title: "CARSTM (Spatiotemporal GLM) in Julia/Turing with GP, factorials, CAR/AR1, PCAs, etc"
header: "CARSTM in Julia"
keyword: |
	Keywords - Guassian Process / CAR , CARSTM Spatiotemporal models
abstract: |
	CARSTM in Julia simple data.

metadata-files:
  - _metadata.yml

params:
  todo: [nothing,add,more,here]
---

<!-- Quarto formatted: To create/render document:

make quarto FN=carstm_julia.md DOCTYPE=html PARAMS="-P todo:[nothing,add,more,here]" --directory=~/projects/model_covariance/docs
 
-->

<!-- To include a common file:
{{< include _common.qmd >}}  
-->

<!-- To force landscape (eg. in presentations), surround the full document or pages with the following fencing:
::: {.landscape}
  ... document ...
:::
-->



## Abstract

This document provides a rigorous technical overview of the **CARSTM** (Conditional Autoregressive Spacetime Model) framework implemented in this Julia environment. CARSTM serves as a high-dimensional Bayesian hierarchical structure designed to decompose complex spatio-temporal phenomena into interpretable latent components: spatial clustering (BYM2), temporal autocorrelation (AR1/RFF), and non-linear interactions. 

This document also shows how to build a CARSTM regression, optionally with a
pPCA, using bottom temperature (Gaussian, space-time, with time as
Fourier harmonics; (temporal processes)\[./temporal_processes.md\] ) and
species composition (Multivariate Normal, latent with Householder
transform, space-time; [spatial processes](./spatial_processes.md)) and snow crab (Hurdle,
binomial and Poisson, space-time) in the Maritimes Region of Canada. 

Key advancements presented here include the integration of **Random Fourier Features (RFF)** to approximate Gaussian Processes, **Deep Gaussian Processes** for non-stationary surface estimation, and a **Mixed-Sampler Gibbs** inference strategy that optimizes sampling efficiency by pairing specific parameters with their mathematically optimal algorithms (e.g., ESS for Gaussian priors, NUTS for regression coefficients). This framework is designed for applications requiring high precision in ecological, epidemiological, or environmental data science.

---

## The Purpose and Rationale


1.  **Define Flexible Spatial Units:** Allow for data-driven partitioning of spatial domains into meaningful units.
2.  **Implement Diverse Spatio-Temporal Models:** Provide a suite of nine different CARSTM models capable of handling various data types (Gaussian, LogNormal, Binomial, Poisson, Negative Binomial) and incorporating different spatial (BYM2), temporal (AR1, RFF), and spatio-temporal interaction structures, as well as categorical and continuous covariates.
3.  **Support Advanced Techniques:** Incorporate modern Bayesian statistical computing methods like PC priors, GMRF constructions, and Random Fourier Features for Gaussian Processes.


### The Spatio-Temporal Challenge
Real-world data often exhibit dependencies across both space and time that violate the independence assumptions of standard GLMs. Traditional methods often treat space and time as separable; however, CARSTM explicitly models the **Space-Time Interaction (Type IV)**, allowing for the discovery of localized anomalies that evolve dynamically.

### Why CARSTM?
The 'Conditional' aspect refers to the Markovian property where the state of a spatial unit depends only on its immediate neighbors. This allows for the construction of **Sparse Precision Matrices**, transforming $O(N^3)$ Gaussian Process problems into computationally tractable $O(N)$ or $O(N \log N)$ GMRF problems.
 
### Core Assumptions

To ensure valid inference, the CARSTM framework relies on several core axioms:

*   **Markov Property:** The spatial effect of a unit is independent of all non-neighbors given its immediate neighbors, enabling sparse precision matrix $Q$. **Conditional Spatial Independence (Markov Property)** assumes that the latent spatial effect $\phi_i$ of a unit is independent of all other units given its immediate neighbors $\mathcal{N}(i)$. This allows the joint distribution of the spatial field to be represented as a **Gaussian Markov Random Field (GMRF)** with a sparse precision matrix $Q$, where $Q_{ij} \neq 0$ only if $i$ and $j$ are neighbors.

*   **Additivity:** The log-linear predictor $\eta$ is a linear combination of separable spatial, temporal, and interaction components. This **Additivity of Latent Components** means the log-linear predictor $\eta$ is assumed to be a linear combination of separable effects: $\eta = \alpha + \text{Space} + \text{Time} + \text{Interaction} + \text{Covariates}$. This implies that while the components can interact (Type IV), their prior structures are defined independently.

*   **Stationarity:** Temporal processes (AR1/RFF) assume constant mean and variance over the standardized $[0, 1]$ interval. **Stationarity of the Temporal Process** is assumed in models using **AR1**, that is, with a constant correlation $\rho$. In **RFF** variants, the kernel (e.g., Matern or Squared Exponential) is assumed to be stationary, meaning the correlation between two time points depends only on their distance, not their absolute position.
 However, this is relaxed with Deep GP models.

*   **Rank-Deficiency:** Intrinsic priors (ICAR, RW2) are singular and require sum-to-zero constraints ($\sum u_i = 0$) for parameter identifiability against the global intercept. **Intrinsic Rank-Deficiency (RW2 and ICAR)** means smoothing priors like the **Second-Order Random Walk (RW2)** and the **Intrinsic CAR (ICAR)** are assumed to be 'intrinsic,' meaning their precision matrices are singular (rank-deficient). We assume a sum-to-zero constraint ($\sum u_i = 0$) to ensure the intercept and the structured effects are identifiable.

Impact of the Sum-to-Zero Constraint on Identifiability: 

  - The Rank-Deficiency Problem: Priors like the **ICAR** (spatial) and **RW2** (temporal/smoothing) define the distribution of *differences* between adjacent points rather than their absolute levels. Consequently, the precision matrix $Q$ for these priors is singular (it has a rank deficiency of at least 1). 

  - **The Null Space:** For a spatial ICAR model, adding any constant $c$ to the entire vector $\mathbf{u}$ (i.e., $\mathbf{u} + c\mathbf{1}$) results in the same prior log-density.

  - **The Mathematical Consequence:** Without a constraint, the model cannot distinguish between the **global intercept** ($\alpha$) and the **mean level** of the spatial field. This leads to an improper posterior where the intercept could drift to $+\infty$ while the spatial field drifts to $-\infty$.

  - Ensuring Identifiability: By enforcing $\sum u_i = 0$, we effectively 'pin' the latent field to a mean of zero. This ensures:
    *   **Unique Estimates:** The global intercept $\alpha$ captures the overall mean of the response.
    *   **Interpretation:** The spatial vector $\mathbf{u}$ captures only the *deviations* from that mean due to geography.
    *   **Sampler Stability:** MCMC samplers (like NUTS or ESS) will fail to converge if this constraint is missing, as they will attempt to explore the infinite 'ridge' of equally likely values in the likelihood surface.

  - Implementation: Here, we handle this through **soft constraints** (adding a tight normal penalty to the sum) or by **re-centering** the vector within the model block, which shifts the mass back to the identifiable region of the parameter space. There are two primary ways to enforce the $\sum u_i = 0$ constraint: 
    *  **Soft Constraint (Penalty Method):** We treat the sum as an observed variable with a very tight prior around zero.
    *  **Explicit Re-centering (Mean-Shift):** We subtract the empirical mean from the vector during every iteration.


* **Bochner’s Theorem for RFF:** For the Random Fourier Feature approximation, it is assumed that the chosen kernel is continuous and positive definite, allowing it to be represented as the Fourier transform of a non-negative spectral density. We sample from this density (typically a Normal distribution) to approximate the infinite-dimensional 

### References

*   **Besag, J. (1974):** Spatial interaction and the statistical analysis of lattice systems.
*   **Rue, H., & Held, L. (2005):** Gaussian Markov Random Fields: Theory and Applications.
*   **Sørbye, S. H., & Rue, H. (2014):** Scaling intrinsic Gaussian Markov random field priors.
*   **Rahimi, A., & Recht, B. (2007):** Random features for large-scale kernel machines.
*   **Hoffman, M. D., & Gelman, A. (2014):** The No-U-Turn Sampler (NUTS).

### Areal units: partitioning of space

The choice of spatial partition directly impacts the identifiability of temporal and spatial latent effects. A well-constructed partition must balance geometric compactness with statistical information density.

Multiple methods exist for partitioning spatial data. A few of the simpler ones are examined.

Specific criteria:
  - **Geometric Regularity**: How compact and convex the tiles are.
  - **Information Balance**: The variance of points per tile (lower is better for CAR stability).
  - **Temporal Coverage**: The minimum number of unique time points present in any single tile.


The issues we are looking for:

Identifiability issues** in hierarchical models (like ICAR or CARSTM) are important .
  - regions with no data make estimating local parameters like temporal persistence ($\rho$) a challenge
  - Poisson Weighted CVT: "shrink" tiles where data is abundant and "stretch" them where data is scarce. Every areal unit is informative.


#### Spatial Partitioning Methodology

The `assign_spatial_units` function allows for several algorithmic approaches to discretize the spatial domain:

1.  **Binary Vector Tree (BVT):** A recursive splitting method that divides the space along the axis of maximum variance. It is computationally efficient and useful for ensuring roughly equal point counts in each unit.
2.  **Quadrant Voronoi Tessellation (QVT):** A quadtree-like approach that recursively divides the domain into four quadrants. Best for capturing multi-scale spatial density variations.
3.  **Agglomerative Voronoi Tessellation (AVT):** Starts with an over-partitioned space and iteratively merges the smallest units until a minimum point threshold is met. This is the most robust method for preventing units with insufficient data slices.
4.  **Centroidal Voronoi Tessellation (CVT):** Uses Lloyd's algorithm to iteratively move centroids to the geometric center of their Voronoi cells, resulting in a highly regularized grid that adapts to the data density.

Each of these methods permits additional constraints in the form of number and density, etc.



### Basis  

Delaunay Triangulation and Voronoi Tesselation

Given a set of centroids $S = \{s_1, s_2, ..., s_n\}$, a Voronoi cell $V_i$ is defined as the set of all points $x$ in the domain such that the distance to $s_i$ is less than or equal to any other centroid $s_j$:
$V_i = \{x \in \mathcal{D} \mid \|x - s_i\| \leq \|x - s_j\|, \forall j \neq i\}$  

**Algorithm:** Uses `DelaunayTriangulation.jl` to compute the dual graph, followed by clipping against the domain boundary using `LibGEOS.jl`.



- Place $k$ seeds across the domain.
- Define boundaries using Delaunay Triangulation and Voronoi tesselation.
- No adaptation to the underlying point process intensity. Seed Placement: needs to be defined.
- High variance of points per unit, meaning some units might be very dense while others are sparse.
- Geometric Regularity: While it produces compact, convex cells, their distribution can be uneven due to the random seed placement.

References

* **Okabe, A., et al. (2000):** Spatial Tessellations: Concepts and Applications of Voronoi Diagrams.
* See information in https://en.wikipedia.org/wiki/Delaunay_triangulation


### Seed or centroid placement

The number and location of such starting points is influential for methods that start with many possibilities and then decrease the number. We use KDE to assist in initiating from an slighly informative basis. Alternatively, random or regular grid placement can also be used.




#### Centroidal Voronoi Tessellation (CVT)

CVT is a spatial partitioning method that creates a mesh by aligning seeds with the centers of their respective regions. It transforms a standard Voronoi diagram into a stable, balanced structure where every "cell" is at its geometric or density-weighted equilibrium. This is achieved with Lloyd’s Algorithm:

-  Partition: Generate Voronoi regions based on current seed locations (usually some regular grid)
-  Shift: Move each seed to the center of mass of its region.
-  Converge: Repeat until the system minimizes the "quantization error" (energy functional):
-  Purpose: for a uniform grid where geography is the priority. 
-  
    $$\mathcal{H}(S, V) = \sum_{j=1}^k \int_{V_j} \rho(x) ||x - s_j||^2 \, dx$$
    
    *(Note: In Standard CVT, the density $\rho(x)$ is treated as a constant 1).*

- Geometric Uniformity. Tiles of roughly equal size/shape, "honeycomb" mesh (often hexagonal).
- Unweighted: Treats all geographic space as equally important
- Minimizes spatial autocorrelation variance **within** units.


#### Adaptive CVT

- Adaptive Resolution. Adjusts tile size based on data distribution. 
- Density-Weighted - Seeds migrate toward dense data clusters. 
- Smaller, denser tiles in high-activity areas; larger tiles in sparse areas. 
- Balances information content; prevents "data-poor" units in sparse areas. 
- Local Split Control - Targeted subdivision/splits in specific tiles based on local metrics, providing high-resolution detail only where it is required.
- Purpose: data density varies significantly and partitions need to adapt to data

- Density-Weighted Centroid. Seeds migrate toward areas with more data points based on an intensity function $\lambda(x)$, typically estimated via Kernel Density Estimation (KDE).

- The position of each seed $s_j$ is calculated as:

$$s_j = \frac{\int_{V_j} x \lambda(x) \, dx}{\int_{V_j} \lambda(x) \, dx}$$

- This ensures the expected number of points per tile remains roughly constant across the entire map, regardless of whether a region is crowded or sparse.
- Prevents "Data Starvation": In standard methods, sparse regions often get stuck with "empty" tiles. This method expands tile size in sparse areas and shrinks it in dense areas, ensuring every tile has sufficient observations to identify temporal trends (e.g., $AR(1)$ processes).
- Minimizes Boundary Artifacts: Standard tiles often "split" a high-density cluster in half. The Hybrid CVT migrates seeds to the modes (peaks) of density, ensuring a single data feature stays within a single tile.
- Improves Model Convergence: By balancing the number of observations ($n_j$) across tiles, it stabilizes the precision matrix for spatial models (like ICAR/CAR) and ensures MCMC chains converge faster.

- KDE Estimation: Map the data intensity $\lambda(s)$ across the domain.
- Seeds at the highest density peaks.
- **Algorithm:** Weighted Lloyd's Algorithm  
  1. Estimate a continuous intensity surface $\lambda(s)$ using KDE (Poisson) or Gaussian Process (GP) regression.
  2. Calculate the **Density-Weighted Centroid** for each unit $V_j$:
   $$s_j = \frac{\int_{V_j} x \lambda(x) dx}{\int_{V_j} \lambda(x) dx}$$
  1. This forces seeds toward data-rich regions, resulting in smaller tiles in high-density areas.

- Produces "statistically significant" tiles. It ensures that no tile is a "data desert," which directly leads to higher reliability in spatio-temporal modeling.


- CVT minimizes the functional $F(s, V) = \sum_{i=1}^n \int_{V_i} \rho(x) \|x - s_i\|^2 dx$, where **$\rho(x)$ is the Kernel Density Estimation (KDE)**.

- **Pros:** Produces the most regular/compact shapes; naturally adapts to point density (low CV of density).
- **Cons:** Computationally expensive (iterative); does not guarantee a minimum number of points per unit.

#### 2. Binary Vector Tree (BVT)
BVT is a hierarchical partitioning method that recursively splits the point set along the axis of maximum variance.
- **Pros:** Extremely fast; ensures very balanced point counts across all units.
- **Cons:** Resulting shapes can be very elongated (high aspect ratio); does not follow local spatial clusters as closely as Voronoi methods.

#### 3. Quadrant Voronoi Tessellation (QVT)
QVT recursively divides the domain into four quadrants based on local means.
- **Pros:** Efficiently captures multi-scale clustering; very intuitive spatial hierarchy.
- **Cons:** Can produce very small/empty units in sparse areas unless strictly constrained.
- A top-down approach in which each internal node has exactly four children. Quadtrees are most often used to partition a two-dimensional space by recursively subdividing it into four quadrants or regions. The Quadtree is built recursively such that If a node has fewer points than its capacity, it becomes a leaf node. If a node has more points than its capacity, it subdivides itself into four child nodes (quadrants).
Points from the parent node are then redistributed into the appropriate child nodes.
- **Algorithm:** Recursive Spatial Decomposition  
  1. Start with a bounding box containing all points.
  2. Recursively divide the box into four equal quadrants if the number of points exceeds the `capacity`.
  3. **Post-Processing:** Filter leaf nodes based on `min_area` and `min_time_slices`.
  4. **Classification:** Re-assign all original points to the centroid of the nearest valid leaf node.


#### 4. Agglomerative Voronoi Tessellation (AVT)
AVT is an iterative method used to enforce minimum sample size constraints (`min_pts`).
- **Pros:** Guaranteed compliance with `min_pts` for statistical power; preserves topological adjacency.
- **Cons:** Can result in highly irregular shapes as units are merged; computationally intensive due to re-triangulation.
- **Algorithm:** Agglomerative Dissolution  
  1. Initialize each data point as its own Voronoi cell.
  2. Construct an adjacency graph via Delaunay triangulation.
  3. **Prioritized Merging:** Calculate the cumulative Poisson intensity for each adjacent pair. Sort pairs by combined density.
  4. Dissolve the edge between the two units with the **lowest** combined intensity first.
  5. Stop when the target number of units is reached or the point-count variance stabilizes (asymptotic convergence).


#### Summary of Findings


Based on the benchmarking results and spatial visualizations, we evaluated four distinct partitioning strategies:

1.  **Centroidal Voronoi Tessellation (CVT):** Consistently achieved the lowest **Coefficient of Variation (CV)** in point density. By iteratively minimizing the variance of point-to-centroid distances, it creates the most spatially "fair" and compact units. This is the gold standard for projects requiring uniform statistical power across units.
2.  **Binary Vector Tree (BVT):** Demonstrated the highest computational speed. By splitting strictly on median coordinates, it guarantees almost perfectly balanced point counts, making it ideal for large-scale data pre-processing where processing time and count-balancing outweigh shape regularity.
3.  **Quadrant Voronoi Tessellation (QVT):** Excels at identifying multi-scale spatial clusters. Its recursive 4-way split naturally matches the hierarchical nature of spatial data (e.g., urban vs. rural density transitions).
4.  **Agglomerative Voronoi Tessellation (AVT):** The only method that provides a hard guarantee on the `min_pts` constraint. While it may produce less regular shapes during the merging phase, it ensures that every resulting unit meets the sample size requirements for downstream statistical modeling.

Ultimately, the criteria for selection of areal units are based upon considerations of:
  

- **Geometric vs. Statistical Adaptation**: 
   - **`cvt`**  prioritize geometric regularlity and compactness. This is excellent for simple spatial models but can lead to 'sparse' units in low-density areas.
   - **`avt`** adaptively shrink tiles in high-density regions. This ensures that even the smallest units have enough points for stable estimation of local effects.

- **Information Balance & Convergence**: 
   - The **`avt`** method achieved the lowest variance in point counts. This 'information balance' is a critical prerequisite for the CARSTM model's precision matrix to be well-conditioned.
   - The asymptotic convergence at 19 units for the density-aware methods suggests that 19 is the 'natural' granularity for this specific point process dataset.

- **Temporal Identifiability**: 
   - All methods successfully maintained full temporal coverage (15 slices) across all units. This confirms that the partitioning logic, combined with the point-count constraints, ensures that the longitudinal (AR1) component of the CARSTM model is identifiable in every areal unit.

- **Edge Handling**: 
   - The automated `boundary_hull` calculation (with dynamic buffering) correctly prevents 'infinite' Voronoi cells at the domain edges, which previously caused artifacts in the adjacency graph (W matrix).


Recommendations:

- **For General Spatial Analysis:** Use **CVT** with a KDE-informed initial seed. It provides the most visually and mathematically robust areal units for mapping and regionalization.
- **For Small Sample Sizes:** Always apply **AVT** as a post-processing step to ensure that no unit is too sparse to produce reliable estimates.
- **For Real-time/Large Data:** Use **BVT** if the dataset contains millions of points and the primary goal is rapid parallelization of the workload across balanced partitions.
- **For Hierarchical Modeling:** Use **QVT** to capture the nested structure of local and regional spatial effects.

#### References
* **Lloyd, S.** (1982). *Least squares quantization in PCM*. IEEE Transactions on Information Theory, 28(2), 129-137.
* **Du, Q., Faber, V., & Gunzburger, M.** (1999). *Centroidal Voronoi Tessellations*. SIAM Review, 41(4), 637-676.
* **Okabe, A. et al.** (2000). *Spatial Tessellations*. John Wiley & Sons.
* **Samet, H.** (2006). *Foundations of Multidimensional and Metric Data Structures*. Morgan Kaufmann.

  


### GMRF Components

The core of the spatial models relies on the **Besag-York-Mollié (BYM2)** specification. We utilize `build_laplacian_precision` and `scale_precision!` to ensure that the structured spatial component has a unit marginal variance. This scaling (Sørbye & Rue, 2014) is critical for prior interpretability, ensuring that the `phi` parameter truly represents the proportion of variance explained by spatial structure.

GMRFs focus upon the precision matrix ($Q$) as this speeds up computations (by not having to invert the covariance matrix).

Many developments have helped enable this approach, especially through INLA (Rue & Held, 2005).

#### Standard AR1 


A standard stationary AR(1) process is defined by: $x_t = \rho x_{t-1} + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma^2)$. For a **cyclic** sequence of length $n$ ($x_0 = x_n$), the joint density $P(\mathbf{x})$ follows:
$$\text{Log} P(\mathbf{x}) \propto -\frac{1}{2\sigma^2(1-\rho^2)} \sum_{i=1}^{n} (x_i - \rho x_{i-1})^2$$
Expanding $(x_i - \rho x_{i-1})^2 = x_i^2 - 2\rho x_i x_{i-1} + \rho^2 x_{i-1}^2$. Summing over $i$:
- **Diagonal ($Q_{ii}$):** Every $x_i$ appears once as $x_i^2$ and once as $\rho^2 x_i^2$, resulting in $1+\rho^2$.
- **Off-Diagonal ($Q_{i, i\pm1}$):** Adjacent pairs receive $-\rho$.
- **Boundary ($Q_{1,n}$):** The cyclic constraint links $x_1$ and $x_n$ with $-\rho$.

With "open" boundaries at $t=1$ and $t=n$:

- **Boundary Condition:** The first element $x_1$ has no predecessor, so its marginal variance is $\sigma^2/(1-\rho^2)$, implying $Q_{11} = 1$. The last element $x_n$ only appears once as a square and once as a lagged term, but since there is no $x_{n+1}$, its diagonal is also $1$ (or effectively $1+\rho^2$ minus the missing link).
- **Structure:** $Q$ is tridiagonal. This is the standard prior for temporal autocorrelation in time-series data.

Unlike the cyclic version, the standard AR1 precision matrix accounts for the "open" boundaries at $t=1$ and $t=n$.
- **Boundary Condition:** The first element $x_1$ has no predecessor, so its marginal variance is $\sigma^2/(1-\rho^2)$, implying $Q_{11} = 1$. The last element $x_n$ only appears once as a square and once as a lagged term, but since there is no $x_{n+1}$, its diagonal is also $1$ (or effectively $1+\rho^2$ minus the missing link).
- **Structure:** $Q$ is tridiagonal. This is the standard prior for temporal autocorrelation in time-series data.


See example: build_ar1_precision()

#### AR(1) Process with Periodic Boundary Conditions

For a **First-Order Autoregressive (AR1) process** with **periodic boundary conditions**. This transforms a standard linear AR(1) sequence into a **Cyclic GMRF**.

The first-order AR(1) process is defined by:

$x_t = \rho x_{t-1} + \epsilon_t, \quad \epsilon_t \sim N(0, \sigma^2)$. 

For a **cyclic** sequence of length $n$ ($x_0 = x_n$), the joint density $P(\mathbf{x})$ follows:

$$\text{Log} P(\mathbf{x}) \propto -\frac{1}{2\sigma^2(1-\rho^2)} \sum_{i=1}^{n} (x_i - \rho x_{i-1})^2$$

Expanding $(x_i - \rho x_{i-1})^2 = x_i^2 - 2\rho x_i x_{i-1} + \rho^2 x_{i-1}^2$. 

Summing over $i$:

- **Diagonal ($Q_{ii}$):** Every $x_i$ appears once as $x_i^2$ and once as $\rho^2 x_i^2$, resulting in $1+\rho^2$.
- **Off-Diagonal ($Q_{i, i\pm1}$):** Adjacent pairs receive $-\rho$.
- **Boundary ($Q_{1,n}$):** The cyclic constraint links $x_1$ and $x_n$ with $-\rho$.


The `build_cyclic_ar1_precision` function constructs the precision matrix ($Q$) for a **First-Order Autoregressive (AR1) process** with **periodic boundary conditions**. This transforms a standard linear AR(1) sequence into a **Cyclic GMRF**.

See example: build_cyclic_ar1_precision()


####  Second-Order Random Walk (RW2)

The RW2 is a smoothing prior used for categorical levels or non-parametric trends. It penalizes the second-order differences:
$$\text{Penalty} \propto \sum (x_i - 2x_{i+1} + x_{i+2})^2$$
- **Precision Matrix:** $Q = D^T D$, where $D$ is the second-difference operator matrix. 
- **Justification:** It acts as a stochastic spline. It is an **intrinsic GMRF (IGMRF)** of rank $n-2$, meaning it is singular (improper) and requires a sum-to-zero constraint for identifiability.

`build_rw2_precision()` (Second-Order Random Walk) 


#### Scaled Precision 

In a BYM2 model, the marginal variance of a spatial effect depends on the graph structure (Sørbye & Rue). The geometric mean of the generalized variances of the IGMRF is used to scale the precision matrix $Q$ such that the structured spatial component has an approximate marginal variance of 1. This allows the `phi` parameter in BYM2 to represent the actual proportion of variance explained by space, making priors on `sigma` interpretable across different geographies.

`scale_precision!()` (Sørbye & Rue Scaling)
In BYM2 and GMRF models, the marginal variance of a spatial effect depends on the graph structure. `scale_precision!` calculates the geometric mean of the generalized variances of the IGMRF.
- **Purpose:** It scales $Q$ such that the structured spatial component has an approximate marginal variance of 1. This allows the `phi` parameter in BYM2 to represent the actual proportion of variance explained by space, making priors on `sigma` interpretable across different geographies.
 
#### Log-likelihood from a precision matrix

Calculating the log-likelihood of a state vector $\mathbf{x}$ given a sparse precision matrix $Q$:

$$\log p(\mathbf{x}|Q) = \frac{1}{2}\log(|Q|) - \frac{1}{2}\mathbf{x}^T Q \mathbf{x} - \frac{n}{2}\log(2\pi)$$

by using a Cholesky decomposition ($Q = LL^T$), efficiently compute the log-determinant as $2 \sum \log(L_{ii})$ and the quadratic form as $||L^T \mathbf{x}||^2$.

`logpdf_gmrf()`
This calculates the log-likelihood of a state vector $\mathbf{x}$ given a sparse precision matrix $Q$:
$$\log p(\mathbf{x}|Q) = \frac{1}{2}\log(|Q|) - \frac{1}{2}\mathbf{x}^T Q \mathbf{x} - \frac{n}{2}\log(2\pi)$$
- **Implementation:** Uses a Cholesky decomposition ($Q = LL^T$) to efficiently compute the log-determinant as $2 \sum \log(L_{ii})$ and the quadratic form as $||L^T \mathbf{x}||^2$.

  
#### References
* **Rue, H., & Held, L. (2005).** *Gaussian Markov Random Fields: Theory and Applications*. CRC Press.
* **Sørbye, S. H., & Rue, H. (2014).** "Scaling intrinsic Gaussian Markov random field priors in spatial statistics."
* **Lindgren, F., et al. (2011).** "An explicit link between Gaussian fields and GMRFs, the SPDE approach." *JRSS-B*.
 



### Smoothing via RW2
Categorical covariates are smoothed using a **Second-Order Random Walk (RW2)**. This acts as a stochastic spline, penalizing the second-order differences between adjacent levels, effectively allowing for non-linear effects in ordered categorical data (e.g., age groups or income brackets).

---

### Model Taxonomy and Implementation

The notebook contains ten distinct model variants, categorized by likelihood family and latent structure. They are examples of implementation that are reasonably well optimized:


| Model | Likelihood Family | Key Feature | Best Use Case |
| :--- | :--- | :--- | :--- |
| **v1** | Gaussian | AR1 + BYM2 | Standard continuous data with linear temporal trends. |
| **v2** | Gaussian | RFF + BYM2 | Continuous data with multi-scale seasonality or complex cycles. |
| **v3** | LogNormal | AR1 + BYM2 | Strictly positive, right-skewed data (e.g., rainfall, income). |
| **v4** | Binomial | AR1 + BYM2 | Proportions, binary trials, or prevalence data. |
| **v5** | Poisson | AR1 + BYM2 | Count data (optional Zero-Inflation for sparse counts). |
| **v6** | Neg-Binomial | AR1 + BYM2 | Over-dispersed count data where Variance > Mean. |
| **v7** | Binomial | Deep GP (RFF) | Proportions with highly non-linear, non-stationary space-time surfaces. |
| **v8** | Gaussian | Deep GP (RFF) | Gold standard for high-dimensional, non-stationary continuous phenomena. |
| **v9** | Gaussian | Continuous RFF | Models with non-linear effects for continuous covariates (Matern-like). |
| **v10**| Gaussian | 3-Layer Deep GP | Experimental: Maximum flexibility for extreme 


#### Gaussian Variants (v1, v2, v8, v9, v10)
*   **v1 (Foundational):** Uses AR1 for time and BYM2 for space. Recommended for well-behaved continuous data.
*   **v2 (RFF-Gaussian):** Replaces AR1 with Random Fourier Features. This is superior for capturing multi-scale seasonality and long-term trends simultaneously.
*   **v8/v10 (Deep GP):** Uses hierarchical RFF layers to model the latent field. This is the 'gold standard' for high-dimensional non-stationarity.

#### Count and Discrete Variants (v4, v5, v6, v7)
*   **v4 (Binomial):** Designed for proportions and binary trial data.
*   **v5/v6 (Poisson/NegBin):** Optimized for count data with optional **Zero-Inflation (ZI)**. The Negative Binomial variant (v6) is specifically recommended when over-dispersion is present ($Var > Mean$).

---


### Random Fourier Features (RFF) 
  
Random Fourier Features provide a way to approximate a kernel function $k(\mathbf{x}, \mathbf{x}')$ with a low-dimensional feature mapping $\phi(\mathbf{x})$. This transforms a Gaussian Process (GP) problem into a linear Bayesian regression problem, which is computationally much more efficient ($O(nm^2)$ vs $O(n^3)$).

#### Bochner's Theorem
Bochner's Theorem states that a continuous, stationary kernel $k(\mathbf{x} - \mathbf{x}')$ is positive definite if and only if it is the Fourier transform of a non-negative measure $p(\mathbf{\omega})$:
$$k(\mathbf{x} - \mathbf{x}') = \int_{\mathbb{R}^d} p(\mathbf{\omega}) e^{i\mathbf{\omega}^T(\mathbf{x} - \mathbf{x}')} d\mathbf{\omega} = E_{\mathbf{\omega}}[e^{i\mathbf{\omega}^T\mathbf{x}} (e^{i\mathbf{\omega}^T\mathbf{x}'})^*]$$

#### The RFF Approximation

Random Fourier Features provide a way to approximate a kernel function $k(\mathbf{x}, \mathbf{x}')$ with a low-dimensional feature mapping $\phi(\mathbf{x})$. This transforms a Gaussian Process (GP) problem into a linear Bayesian regression problem, which is computationally much more efficient ($O(nm^2)$ vs $O(n^3)$).

#### Bochner's Theorem
Bochner's Theorem states that a continuous, stationary kernel $k(\mathbf{x} - \mathbf{x}')$ is positive definite if and only if it is the Fourier transform of a non-negative measure $p(\mathbf{\omega})$:
$$k(\mathbf{x} - \mathbf{x}') = \int_{\mathbb{R}^d} p(\mathbf{\omega}) e^{i\mathbf{\omega}^T(\mathbf{x} - \mathbf{x}')} d\mathbf{\omega} = E_{\mathbf{\omega}}[e^{i\mathbf{\omega}^T\mathbf{x}} (e^{i\mathbf{\omega}^T\mathbf{x}'})^*]$$

#### The RFF Approximation
By Monte Carlo sampling $\mathbf{\omega}_1, \dots, \mathbf{\omega}_m$ from the spectral density $p(\mathbf{\omega})$ (which is a Gaussian distribution for the Squared Exponential kernel), we can approximate the expectation:
$$\phi(\mathbf{x}) = \sqrt{\frac{2}{m}} [\cos(\mathbf{\omega}_1^T\mathbf{x} + b_1), \dots, \cos(\mathbf{\omega}_m^T\mathbf{x} + b_m)]^T$$
where $b_i \sim \text{Uniform}(0, 2\pi)$. The inner product $\phi(\mathbf{x})^T \phi(\mathbf{x}')$ then converges to $k(\mathbf{x}, \mathbf{x}')$ as $m \to \infty$.

By Monte Carlo sampling $\mathbf{\omega}_1, \dots, \mathbf{\omega}_m$ from the spectral density $p(\mathbf{\omega})$ (which is a Gaussian distribution for the Squared Exponential kernel), we can approximate the expectation:
$$\phi(\mathbf{x}) = \sqrt{\frac{2}{m}} [\cos(\mathbf{\omega}_1^T\mathbf{x} + b_1), \dots, \cos(\mathbf{\omega}_m^T\mathbf{x} + b_m)]^T$$
where $b_i \sim \text{Uniform}(0, 2\pi)$. The inner product $\phi(\mathbf{x})^T \phi(\mathbf{x}')$ then converges to $k(\mathbf{x}, \mathbf{x}')$ as $m \to \infty$.

#### References
* **Rahimi, A., & Recht, B. (2007).** "Random features for large-scale kernel machines." *NIPS*.

 
### Deep Gaussian Processes

A Deep GP is a hierarchical composition of GPs. In our implementation (`model_v7`/`model_v8`), we use the RFF approximation for each layer to maintain tractability.

#### Hierarchical Structure
For a two-layer model, the process is defined as:
1.  **Layer 1:** $\mathbf{h} = f_1(\mathbf{x})$, where $f_1 \sim GP(0, k_1)$.
2.  **Layer 2:** $y = f_2(\mathbf{h})$, where $f_2 \sim GP(0, k_2)$.

#### Justification
- **Non-Stationarity:** While each individual GP layer might use a stationary kernel (like RFF-Matern), the composition $f_2(f_1(\mathbf{x}))$ is highly non-stationary and non-Gaussian. This allows the model to adapt its lengthscale locally.
- **Feature Extraction:** The first layer acts as a non-linear dimensionality reduction or warping of the input space $(x, y, t)$, allowing the second layer to discover complex spatio-temporal interactions that a single-layer GP could not capture.

#### References
* **Damianou, A., & Lawrence, N. (2013).** "Deep Gaussian Processes." *AISTATS*.
* **Cutajar, K., et al. (2017).** "Random Feature Expansions for Deep Gaussian Processes." *ICML*.



### Optimal Inference: The Mixed-Sampler Strategy (Gibbs Sampling)
  
Efficient inference for Conditional Autoregressive Spacetime Models (CARSTM) requires a **Mixed-Sampler Gibbs** approach. Because the parameter space includes high-dimensional latent fields, smooth categorical effects, and simple variance scalars, a single algorithm is rarely optimal for the entire model.

### 1. Latent Gaussian Fields
**Recommended: Elliptical Slice Sampling (ESS())**
*   **Use Case:** `u_icar`, `u_iid`, `f_tm_raw`, `st_int_raw`.
*   **Justification:** ESS is specifically designed for parameters with Gaussian priors. It requires **no tuning** (no step size or path length) and is extremely stable in the high-dimensional spaces typical of spatial and interaction effects.
*   **Optimization:** Ensure variables are zero-centered in their priors. ESS is analytically exact for Gaussian priors and requires no manual tuning parameters.
*   **Original Paper:** Murray, I., Adams, R. P., & MacKay, D. J. (2010). Elliptical slice sampling. *AISTATS*.
*   **Context in CARSTM:** ESS is optimal for latent Gaussian fields (ICAR/AR1) because it leverages the properties of the multivariate normal prior. Unlike HMC, it requires no gradient information or step-size tuning, making it robust for high-dimensional spatial dependencies.


### 2. Differentiable Regression Coefficients
**Recommended: NUTS()**
*   **Use Case:** `beta_cov`, fixed effects.
*   **Justification:** The No-U-Turn Sampler excels at navigating complex posterior geometries of regression weights by adaptively finding the optimal path length, avoiding random walk behavior.
*   **Original Paper:** Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*.
*   **DynamicHMC Implementation:** Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. *arXiv preprint arXiv:1701.02434*.
*   **Context in CARSTM:** Used for categorical regression coefficients (`beta_cov`). NUTS adaptively finds the optimal path length, avoiding the random walk behavior of MH while being more user-friendly than standard HMC.
*   **`target_acceptance` (Default: 0.65)**: Increase to `0.8` or `0.9` if you see 'divergent transitions'. This forces the sampler to take smaller, more cautious steps.
*   **`max_depth` (Default: 10)**: If the sampler hits the maximum depth (1024 steps), increase this. It allows the sampler to explore longer trajectories in complex posteriors.


### 3. Discrete & Non-Differentiable Structures
**Particle Gibbs (PG)** is a powerful algorithm for sampling from models with complex, non-differentiable, or discrete latent structures. In the CARSTM framework, PG is most beneficial for:
**Recommended: Particle Gibbs (PG())**
*   **Use Case:** Zero-inflation indicators (`phi_zi`), discrete latent states.
*   **Justification:** PG uses a particle filter to sample from latent trajectories that are discrete or non-differentiable, allowing them to be integrated into the broader Gibbs framework.
*   **Particle Count:** Usually `10` to `50`. Increasing particles improves the approximation of the latent state but increases computational cost linearly. `PG(40)` is a robust production default.
*   **Zero-Inflation components**: Handling the discrete latent state $z_i \in \{0, 1\}$ in ZIP or ZINB models.
*   **Discrete Latent Regimes**: If the spatiotemporal interaction follows a hidden Markov structure.
*   **Non-linear State-Space trajectories**: When the temporal process $\rho$ is part of a non-Gaussian state-space representation.
* PG is often paired with HMC/NUTS for the continuous parameters while it handles the 'difficult' discrete or highly non-linear parts.
*   **Original Paper:** Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle markov chain monte carlo methods. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*.
*   **Context in CARSTM:** Critical for non-differentiable or discrete parameters, such as zero-inflation indicators in Poisson or Negative Binomial variants, as it uses sequential Monte Carlo to update latent paths.

 
### 4. HMC (Standard Hamiltonian Monte Carlo) and HMC with Dual Averaging (HMCDA)
*   **Reference:** Andrieu, C., & Thoms, J. (2008). A tutorial on adaptive MCMC. *Statistics and Computing*.
*   **Context in CARSTM:** Provides stable sampling for latent fields with strict geometric constraints by using dual averaging to adapt the step size while maintaining a fixed trajectory length.
* HMC:
  *   **`epsilon` (Step Size)**: Too large causes instability; too small makes sampling slow. A good start is often `0.05` to `0.1`.
  *   **`n_leapfrog` (Steps)**: Usually between `10` and `50`. The goal is to ensure the trajectory is long enough to reach a new area of the posterior but short enough to avoid looping back.
* HMCDA:
  *   **`trajectory_length`**: This is the total distance traveled ($ \epsilon \times n_{\text{leapfrog}} $). A value of `1.0` is a robust default for standardized models.
  *   **`target_acceptance`**: Similar to NUTS, `0.65` is standard, while `0.8+` is safer for highly correlated parameters.

 

### 5. Variance and Correlation Parameters
**Recommended: Metropolis-Hastings (MH())**
*   **Use Case:** `sigma_y`, `sigma_sp`, `phi_sp`, `rho_tm`.
*   **Justification:** These are typically low-dimensional scalars. MH is computationally cheap and often converges faster for these parameters than gradient-based methods, which may struggle with the bounded nature of variances.
*   **Proposal Distribution:** For variance parameters, provide an explicit proposal standard deviation if acceptance rates fall outside the 20-50% range. 
*   *Example:* `MH(:sigma_y => Normal(0, 0.05))`.


### 6. Mixed-Sampler Gibbs Framework
*   **Reference:** Gelfand, A. E., & Smith, A. F. (1990). Sampling-based approaches to calculating marginal densities. *Journal of the American Statistical Association*.
*   **Context in CARSTM:** The meta-strategy of combining specialized samplers (ESS for space, NUTS for fixed effects, MH for variance) is rooted in the conditional independence properties of the Gibbs sampler, allowing for highly efficient inference in hierarchical models.
 
 
#### Sampler Theory
*   **Murray, I., et al. (2010).** *Elliptical slice sampling.* AISTATS. (Foundational for high-dimensional Gaussian latents).
*   **Hoffman, M. D., & Gelman, A. (2014).** *The No-U-Turn sampler.* JMLR. (Justification for adaptive HMC path lengths).
*   **Andrieu, C., et al. (2010).** *Particle Markov chain Monte Carlo methods.* JRSS-B. (Derivation of Particle Gibbs for latent states).

#### Spatiotemporal Modeling
*   **Rue, H., & Held, L. (2005).** *Gaussian Markov Random Fields: Theory and Applications.* CRC Press.
*   **Sørbye, S. H., & Rue, H. (2014).** *Scaling intrinsic Gaussian Markov random field priors in spatial statistics.* (Justification for BYM2 scaling used in `Q_sp`).
*   **Rahimi, A., & Recht, B. (2007).** *Random features for large-scale kernel machines.* NIPS. (Theoretic basis for RFF approximations).

---

### Advanced Optimization: MAP and ADVI Tuning

For production-grade point estimates and fast posterior approximations, we use the following optimized configurations:

1. **MAP with L-BFGS**: This quasi-Newton method uses second-order curvature information to converge faster than standard gradient methods.
2. **ADVI with Multi-Sample Gradients**: By increasing the samples per iteration, we reduce the variance of the stochastic gradient, ensuring the ELBO converges to a more stable local optimum.

- increasing the number of samples used to estimate the ELBO gradient (e.g., ADVI(10, 1000)) significantly reduces noise and stabilizes convergence, especially when dealing with complex spatial interactions.  

The Role of `n_samples` in `ADVI(n_samples, n_iterations)`:

In Automatic Differentiation Variational Inference (ADVI), we aim to maximize the **Evidence Lower Bound (ELBO)**. Since the ELBO involves an expectation over the variational distribution $q(\theta)$, its gradient must be estimated stochastically.

#### 1. Stochastic Gradient Estimation
`n_samples` (the first argument) specifies how many samples are drawn from the variational posterior $q$ to calculate the empirical mean of the gradient at each optimization step. 

*   **Low `n_samples` (e.g., 1):** The gradient estimate is highly computationally efficient but has **high variance**. This can lead to "chatter" where the optimization bounces around the optimum or fails to converge in complex landscapes.
*   **High `n_samples` (e.g., 10-50):** The gradient estimate is much **smoother and more stable** (reduced variance). This is often necessary for Spatiotemporal models where the interaction between latent fields and categorical effects creates a very complex energy surface.

#### 2. The Trade-off
Increasing `n_samples` improves the accuracy of each step but increases the computational cost of each iteration linearly. 

**Recommendation for CARSTM:**
Start with `ADVI(1, 1000)` for quick smoke tests. If the ELBO plot looks extremely jagged or the results are inconsistent across runs, move to `ADVI(10, 2000)`. The extra samples help the optimizer "see through the noise" of high-dimensional spatial dependencies.


### Recommended Optimizers for `optimize()`

*   **`LBFGS()`**: Default choice for most differentiable models. Scalable and fast.
*   **`Newton()`**: Use if you have a smaller number of parameters and want exact second-order convergence (requires Hessian).
*   **`NelderMead()`**: Use only if the model is non-differentiable (e.g., contains discrete parameters), though it is much slower.


### Alternative Approaches and Research Gaps

While the CARSTM framework implemented here via `Turing.jl` is robust, it is important to consider alternative methodologies and acknowledge the current limitations of this implementation.


#### Alternative Modeling Paradigms

*   **Integrated Nested Laplace Approximations (INLA):** 
    *   *Approach:* A deterministic alternative to MCMC for Latent Gaussian Models (LGMs).
    *   *Comparison:* INLA is significantly faster than NUTS/ESS for large spatial grids but is less flexible for non-Gaussian latent components (e.g., Deep GPs) or custom likelihoods not pre-defined in the `R-INLA` or `inlabru` ecosystems.
*   **Fixed Rank Kriging (FRK):**
    *   *Approach:* Uses a fixed set of basis functions to reduce dimensionality.
    *   *Comparison:* Similar to our RFF approach, but typically focuses on spatial covariance functions directly rather than the precision-matrix (GMRF) approach. It is often preferred for very smooth, large-scale geophysical processes.
*   **Stochastic Partial Differential Equations (SPDE):**
    *   *Approach:* Represents Matern GPs as solutions to linear SPDEs on triangular meshes.
    *   *Comparison:* This is the 'gold standard' for continuous spatial surfaces. While our RFF approximates this, an explicit SPDE-GMRF link (Lindgren et al., 2011) provides more rigorous boundary condition handling.

#### Identified Gaps and Future Work

1.  **Scalability to 'Big' Space-Time Data:**
    *   *Gap:* As the number of spatial units ($S$) and time points ($T$) grows, the interaction term ($S \times T$) creates a massive latent field. 
    *   *Solution Path:* Future iterations should explore **Kronecker Product decomposition** of precision matrices to maintain $O(N)$ memory complexity.

2.  **Non-Gaussian Interactions:**
    *   *Gap:* Current Type IV interactions assume Gaussianity in the latent space. 
    *   *Solution Path:* Implementing Copula-based interactions would allow for tail-dependence in extreme events (e.g., synchronized flood risks across regions).

3.  **Automated Prior Sensitivity Analysis:**
    *   *Gap:* Hierarchical models are sensitive to hyper-priors on variance components ($\sigma_{sp}, \sigma_{tm}$).
    *   *Solution Path:* Integrating automated Robustness Checks (e.g., using `SimulationBasedCalibration.jl`) to ensure priors do not dominate the posterior in data-sparse regions.

4.  **Dynamic Graph Structures:**
    *   *Gap:* The spatial adjacency matrix $W$ is currently static.
    *   *Solution Path:* For applications like epidemiology, allowing $W$ to evolve (Dynamic Network CAR) would capture changing connectivity due to infrastructure or policy shifts.

5. Key Information & Decisions
*   **Model Architecture:** The framework successfully integrates **BYM2 spatial effects**, **AR1 temporal processes**, **Random Fourier Features (RFF)** for seasonality, and **Type IV interactions** for space-time anomalies.
*   **Identifiability Constraints:** To separate the latent field from the global intercept, we utilize **Explicit Re-centering** (mean subtraction) within the model block. This is preferred over soft penalty constraints for superior NUTS sampler stability.
*   **Scaling and Priors:** Implementation of **Sørbye & Rue scaling** ensures that the spatial precision matrix is numerically conditioned, while **Penalized Complexity (PC) Priors** provide principled shrinkage for variance components.

6. Computational Trade-offs
*   **Gaussian CARSTM (GMRF-based):** High efficiency ($O(N)$ to $O(N \log N)$ complexity) due to sparse precision matrices. Ideal for large spatial grids but limited by assumptions of stationarity.
*   **Deep GP (RFF-based):** Captures non-stationary and non-linear relationships by warping the input space. It incurs higher memory costs ($O(N \cdot m^2)$) and requires higher `target_acceptance` in NUTS to navigate the complex posterior landscape.

7. Unresolved Tasks & Research Gaps
*   **Scalability:** Implementation of **Kronecker Product decomposition** for interaction matrices to handle massive datasets without $O(N^2)$ memory growth.
*   **Advanced Modeling:** Exploration of **Copula-based interactions** for non-Gaussian tail dependencies and **Dynamic Network CAR** for time-varying spatial connectivity (e.g., infrastructure changes).
*   **Inference Calibration:** Systematic benchmarking of **ADVI** (Variational Inference) vs. **MCMC** to determine optimal iteration counts for varying levels of model complexity.

### Recommendations 

#### Data Preparation
*   **Time Standardization:** Raw time indices should be mapped to $[0, 1]$ to prevent trigonometric overflow in RFF basis generation.
*   **Graph Connectivity:** Always use `ensure_connected!` on your spatial graph. Disconnected components in a GMRF can lead to singular precision matrices and sampler failure.

#### Prior Selection
We implement **PC Priors (Penalized Complexity)** for standard deviations. These priors are designed to be 'informative' in their pull toward a simpler base model (e.g., zero variance) unless the data strongly suggest otherwise, preventing over-fitting in the interaction terms.

---

### Conclusions
The CARSTM framework presented here represents a robust, scalable solution for spatio-temporal Bayesian inference. By combining the computational efficiency of GMRFs with the flexibility of Deep GPs and the stability of Mixed-Sampler Gibbs, this environment allows for the modeling of highly complex datasets that were previously computationally prohibitive.

---

### References
*   **Rue & Held (2005):** Gaussian Markov Random Fields.
*   **Hoffman & Gelman (2014):** The No-U-Turn Sampler.
*   **Murray et al. (2010):** Elliptical Slice Sampling.
*   **Rahimi & Recht (2007):** Random Features for Large-Scale Kernel Machines.
*   **Sørbye & Rue (2014):** Scaling intrinsic GMRF priors.



## CARSTM in Julia


We use a wide array of Julia package involving statistics, machine learning, and spatial analysis:

*   **General Purpose:** `Random`, `Statistics`, `LinearAlgebra`, `DataFrames`, `StatsBase`, `JLD2`
*   **Spatial Analysis:** `LibGEOS`, `Graphs`, `DelaunayTriangulation` (for Voronoi tessellation)
*   **Machine Learning/Bayesian Modeling:** `Distributions`, `MCMCChains`, `SparseArrays`, `StaticArrays`, `FillArrays`, `Bijectors`, `DynamicPPL`, `AdvancedVI`, `Optimisers`, `PosteriorStats`, `Turing` (the primary Bayesian modeling framework)
*   **Plotting:** `Plots`, `StatsPlots`
*   **Numerical/Optimization:** `FFTW` (Fast Fourier Transforms, likely for spectral analysis or signal processing in temporal models).



### Functional Breakdown

The functions are organized into several logical blocks:

#### 1. Spatial Unit Partitioning and Graph Construction

*   `expand_hull_v0`, `get_coords_from_geom_v0`, `expand_hull`, `get_coords_from_geom`: These functions are for geometric manipulation using `LibGEOS`. They compute convex hulls and extract coordinates from various `LibGEOS` geometry types. The `_v0` versions appear to be initial attempts, while the later ones are more refined, especially in handling coordinate extraction. The use of `LibGEOS` implies precise geometric operations.
*   `get_bvt_centroids`, `get_qvt_centroids`, `get_avt_centroids`: These implement different spatial partitioning algorithms (Binary Vector Tree, Quadrant Voronoi Tessellation, Agglomerative Voronoi Tessellation) to define spatial units based on point density and distribution. This is a crucial step for defining the 'areas' in CARSTM models.
*   `assign_spatial_units`: A high-level wrapper that orchestrates the spatial partitioning process. It takes raw points, a chosen method (e.g., `:cvt`, `:bvt`, `:qvt`, `:avt`), and various parameters like target/max units, minimum time slices, and buffering distance. It also returns a spatial graph (`SimpleGraph`). This is a complex function integrating several sub-processes.
*   `get_voronoi_polygons_and_edges`: Generates Voronoi polygons based on centroids and clips them to the convex hull. Crucially, it also identifies adjacency edges between polygons, which are essential for constructing spatial adjacency matrices (`W_sym`). It includes robust checks for adjacency using buffering.
*   `check_connectivity`, `ensure_connected!`: These functions deal with ensuring the generated spatial graph is connected. `ensure_connected!` modifies the graph in-place by adding edges between nearest neighbors of disconnected components, which is important for GMRF-based spatial models.
*   `plot_spatial_graph`: Visualizes the spatial units, centroids, points, and adjacency graph using `Plots.jl`. This is a utility for inspecting the spatial partitioning results.

#### 2. Data Generation and Initial Exploration

*   `generate_sim_data`: Creates synthetic spatio-temporal data, including `(x,y)` coordinates, `time_idx`, `y_sim` (continuous), `y_binary`, `weights`, `trials`, and `cov_indices`. This function is fundamental for demonstrating and testing the CARSTM models without real-world data.
*   `estimate_local_kde_with_extrapolation`, `plot_kde_simple`: Implement and visualize Kernel Density Estimation (KDE) for spatial intensity. This helps understand point density, which is often a factor in spatial partitioning.
*   `calculate_metrics`: Computes density metrics (mean, SD, CV) for the partitioned spatial units. This is useful for evaluating the quality of the spatial partitioning (e.g., homogeneity of unit sizes/densities).

#### 3. Bayesian Model Utilities

*   `init_params_extract`, `init_params_copy`: Facilitate parameter initialization for Turing models, either by extracting means from a previous chain or loading from a file. This is a practical utility for improving MCMC sampling efficiency, especially when warm-starting.
*   `PCPriorSigma`: A custom `ContinuousUnivariateDistribution` for PC (Penalized Complexity) priors on standard deviations, allowing for more flexible prior specification in the Bayesian models.
*   `build_laplacian_precision`, `scale_precision!`, `build_rw2_precision`, `build_ar1_precision`, `build_cyclic_ar1_precision`: These are critical functions for constructing precision matrices for various Gaussian Markov Random Field (GMRF) components (e.g., BYM2 spatial effects, RW2 for smooth categorical effects, AR1 for temporal effects). `scale_precision!` is particularly important for identifiability and numerical stability.
*   `logpdf_gmrf`: Calculates the log-probability density for a GMRF given its precision matrix and a vector of values. This is used extensively within the Turing models for custom likelihood contributions.
*   `summarize_array`: A generic function to summarize MCMC samples (mean, median, credible intervals) across the last dimension.
*   `reconstruct_posteriors`: A comprehensive function to process MCMC chains and reconstruct meaningful posterior estimates for spatial, temporal, categorical, and interaction effects, as well as predictions. It handles different model families (Gaussian, Poisson, Binomial, etc.) and integrates various effect types. This is a very complex function that makes the raw MCMC output interpretable.
*   `detect_model_family`: Infers the likelihood family from the Turing model's function name, used by `reconstruct_posteriors`.
*   `plot_model_fit`, `posterior_predictive_check`: Functions to evaluate model fit visually and quantitatively (RMSE, Pearson, Kendall Tau). `posterior_predictive_check` uses `HypothesisTests` for statistical rigor.
*   `plot_posterior_results`: Visualizes specific model effects (spatial maps, categorical effects) from the `reconstruct_posteriors` output. This is a crucial diagnostic tool.
*   `plot_posterior_vs_prior`: Compares posterior and prior densities for a given parameter, helping to assess the impact of the data on prior beliefs.
*   `calculate_st_intervals`: A placeholder for calculating credible intervals for spatio-temporal effects.
*   `NegativeBinomial2`: A re-parametrization of the Negative Binomial distribution, commonly used in ecological/count data modeling.
*   `calculate_waic`: Computes the Watanabe-Akaike Information Criterion, a widely used metric for Bayesian model comparison, leveraging `Turing.pointwise_loglikelihoods`.
*   `get_rff_deep2D_basis`, `get_rff_trend_basis`, `get_rff_seasonal_basis`: Implement Random Fourier Features (RFF) for approximating Gaussian Processes in various dimensions (2D spatial/temporal, 1D trend, 1D seasonal). These are used in the more advanced RFF-based models.

#### 4. Turing Models (CARSTM Implementations)

Nine `Turing.@model` definitions are present, each representing a specific CARSTM variant:

*   **`pca_carstm` & `carstm_pca`**: These models explore latent factor analysis (PCA via Householder transforms) combined with CARSTM effects. `pca_carstm` applies PCA first, then CARSTM on factor scores, while `carstm_pca` attempts the reverse (though noted as incomplete/slow). They deal with multivariate data.
*   **`carstm_temperature`**: A specific application model using ICAR (spatial) and Fourier (temporal) processes, likely for continuous temperature data.
*   **`model_v1_gaussian`**: A foundational CARSTM with Gaussian likelihood, integrating BYM2 spatial, AR1 temporal, and a Type IV space-time interaction, along with RW2-smoothed categorical covariates. This represents a robust spatio-temporal modeling framework.
*   **`model_v2_carstm_rff`**: Similar to V1 but replaces explicit AR1 temporal and spatial structures with Random Fourier Features for temporal trends and seasonality, suggesting a more flexible (and potentially computationally intensive) approach.
*   **`model_v3_lognormal`**: Adapts the V1 structure for LogNormal likelihood, suitable for positive, right-skewed data.
*   **`model_v4_binomial`**: Adapts the V1 structure for Binomial likelihood (logit link), suitable for binary or proportion data.
*   **`model_v5_poisson`**: Adapts the V1 structure for Poisson likelihood (log link), with an option for Zero-Inflated Poisson (`use_zi`), suitable for count data.
*   **`model_v6_negativebinomial`**: Adapts the V1 structure for Negative Binomial likelihood (log link), also with an option for Zero-Inflated Negative Binomial (`use_zi`), providing more flexibility for over-dispersed count data than Poisson.
*   **`model_v7_deep_gaussianprocess_binomial` & `model_v8_deep_gaussianprocess_gaussian`**: These are advanced models incorporating deep Gaussian Processes (approximated with RFFs) to capture complex non-linear spatio-temporal relationships, combined with RW2-smoothed categorical covariates. They represent a significant jump in model complexity.
*   **`model_v9_continuous_gaussian`**: Extends the V1 framework to include continuous covariates modeled using RFFs (Matern-like kernels), providing another layer of flexibility for handling various covariate types.
 

### Start environment

**WARNING**: if this is the first run, this can take up to 1 hour to install and precompile libraries and their dependencies

```{julia}
 
# For Areal Units
pkgs_au = ["Random", "Statistics", "LinearAlgebra", "DataFrames",
       "StatsBase", "SparseArrays", "Plots", "StatsPlots", 
        "JLD2", "LibGEOS", "Graphs", "DelaunayTriangulation" ]
 

# For CARSTM 
pkgs_carstm = ["Random",   "Distributions", "Statistics", "MCMCChains", "DataFrames",
        "LinearAlgebra", "Clustering", "StatsBase", "HypothesisTests",
        "JLD2", "FFTW",  "SparseArrays", "StaticArrays", "FillArrays",
         "Bijectors", "DynamicPPL", "AdvancedVI", "Optimisers", "PosteriorStats",  "Turing" ]
 

pkgs = unique( vcat( pkgs_au, pkgs_carstm ))

using Pkg
Pkg.add(pkgs)
Pkg.precompile()
# Pkg.instantiate()
# Pkg.gc()

for pk in pkgs
  @eval using $(Symbol(pk))
end

# using Pkg
# Pinning LibGEOS to the latest available package version to resolve API inconsistencies
# Pkg.add(name="LibGEOS", version="0.9.7")
# Pkg.precompile()



# define 'project_directory' as the location of the repository -- required

if Sys.iswindows()
    project_directory = joinpath( "C:\\", "home", "jae", "projects", "model_covariance")  
elseif Sys.islinux()
    project_directory = joinpath( "/home", Sys.username(), "projects", "model_covariance")
else
    project_directory = joinpath( "C:\\", "Users", "choij", "projects", "model_covariance")  # examples
end


include( joinpath( project_directory, "src", "spatial_partitioning_functions.jl" ) )   
include( joinpath( project_directory, "src", "carstm_functions.jl" ) )    

 

```


### Simulated data

```{julia}

# Data Generation
n_pts = 100
n_time = 15
 
(pts, y_sim, y_binary, time_idx, weights, trials, cov_indices) =
  generate_sim_data(n_pts, n_time; rndseed=42);


# Use an existing time slice for demonstration
target_ts = 5 # Example time slice

x_grid_kde, y_grid_kde, intensity_kde = estimate_local_kde_with_extrapolation(pts, time_idx, target_ts; grid_res=100, sd_extension_factor=1.0)

# Filter points for the target time slice for plotting
filtered_pts_for_plot = pts[findall(==(target_ts), time_idx)]

# Create a heatmap of the KDE intensity
p_kde = Plots.heatmap(x_grid_kde, y_grid_kde, intensity_kde';
                      colorbar_title="Intensity",
                      title="Local KDE for Time Slice $target_ts",
                      xlabel="X Coordinate", ylabel="Y Coordinate",
                      aspect_ratio=:equal,
                      c=:viridis)

# Overlay the actual points for that time slice
Plots.scatter!(p_kde, [p[1] for p in filtered_pts_for_plot], [p[2] for p in filtered_pts_for_plot];
               marker=:circle, markersize=3, markeralpha=0.6, markercolor=:white, label="Points in TS $target_ts")

display(p_kde)



# Define common constraints
common_min_area_points = 2.0
common_max_area_points = 30.0
min_total_units_benchmark = 5
max_total_units_benchmark = 15
tolerance = 1e-1 

# Final verification run with robust BVT/Quadtree and cleaned AVT adjacency
test_configs_expanded = [
    (:cvt, 8), 
    (:qvt, 6),
    (:bvt, 10),
    (:avt, 20)
]

results_expanded = []
plots_expanded = []

for (m, max_u) in test_configs_expanded
    local spatial_res
    spatial_res = assign_spatial_units(pts, m;
        max_total_arealunits=max_u,
        buffer_dist=0.8,
        min_pts=15)

    push!(results_expanded, (method=m, requested_max=max_u, actual_units=length(spatial_res.centroids)))

    p = plot_spatial_graph(pts, spatial_res; title="Method: $m", domain_boundary=spatial_res.hull_coords)
    push!(plots_expanded, p)
end

display(DataFrame(results_expanded))
display(Plots.plot(plots_expanded..., layout=(3, 2), size=(900, 1000)))



benchmark_results = []
for (m, max_u) in test_configs_expanded
    res = assign_spatial_units(pts, m; max_total_arealunits=max_u, buffer_dist=0.8, min_pts=15)
    met = calculate_metrics(res, pts)
    push!(benchmark_results, (method=m, units=length(res.centroids), mean_dens=met.mean_density, sd_dens=met.sd_density, cv_dens=met.cv_density))
end

display(DataFrame(benchmark_results))


```


 


## Example 0: Simulated data

```{julia}

# Data 
n_pts = 100
n_time = 15

(pts, y_sim, y_binary, time_idx, weights, trials, cov_indices) =
  generate_sim_data(n_pts, n_time; rndseed=42)

plot_kde_simple(pts, sd_extension_factor=1.0, title="Spatial Intensity (KDE)")

# Define common constraints
common_min_area_points = 2.0
common_max_area_points = 30.0
min_total_units = 5
max_total_units = 15
min_pts = 3
tolerance = 1e-1

# Ensure we are using the simulation data generated earlier
area_method = :avt  # avt, cvt, or bvt, qvt
spatial_res = assign_spatial_units(pts, area_method;
        max_total=max_total_units,
        buffer_dist=0.8,  # fraction of mean distances
        min_pts=min_pts)

actual_units=length(spatial_res.centroids)

plt = plot_spatial_graph(pts, spatial_res; title="Method: $area_method", domain_boundary=spatial_res.hull_coords)

display(plt)
   

# classify locations and adjacency
g_v1 = get_spatial_graph(spatial_res)
W_sym = Float64.( Graphs.adjacency_matrix(g_v1) )

area_idx = spatial_res.assignments ; 

# Ensure cov_indices is correctly shaped as an N_obs x 4 matrix
cov_indices_mat = hcat(cov_indices, cov_indices, cov_indices, cov_indices)

trials_sim = ones(Int, length(y_binary)); # For binary outcome, 1 trial per observation
class1_sim = rand(1:13, length(y_binary)); # A categorical variable with 13 levels
class2_sim = rand(1:2, length(y_binary)) ; # A categorical variable with 2 levels
weights_sim = ones(Float64, length(y_binary)); # Assign equal weight to all observations
  

# Pre-compute static features outside of Turing models:
using BenchmarkTools

# Configuration
n_bench_iters = 500
bench_results = Dict{String, Float64}()

# 1. Setup shared precomputations
n_categories = 13
MARGS = precompute_model_inputs(y_sim, pts, area_idx, time_idx, W_sym, n_categories)
cont_covs_dummy = randn(length(y_sim), 2)

# Ensure count data is strictly non-negative integers for count models
y_counts = abs.(Int.(round.(MARGS.y)))
precomp_counts = merge(MARGS, (y = y_counts,))
using BenchmarkTools

# Configuration
n_bench_iters = 50
bench_results = Dict{String, Float64}()

# Ensure count data is strictly non-negative integers for count models
y_counts = abs.(Int.(round.(MARGS.y)))
precomp_counts = merge(MARGS, (y = y_counts,))

println("Starting Suite Benchmark (MH, $n_bench_iters iterations)...\n")

# Define the full model set with corrected data inputs for count families
models_to_bench = Dict(
    "v1_gaussian"         => () -> model_v1_gaussian(MARGS),
    "v2_rff_gaussian"     => () -> model_v2_rff_gaussian(MARGS),
    "v3_lognormal"        => () -> model_v3_lognormal(MARGS),
    "v4_binomial"         => () -> model_v4_binomial(MARGS; trials=ones(Int, length(y_sim))),
    "v5_poisson"          => () -> model_v5_poisson(precomp_counts),
    "v6_negativebinomial" => () -> model_v6_negativebinomial(precomp_counts),
    "v7_deep_gp_binomial" => () -> model_v7_deep_gp_binomial(MARGS; trials=ones(Int, length(y_sim))),
    "v8_deep_gp_gaussian" => () -> model_v8_deep_gp_gaussian(MARGS),
    "v9_continuous_gaussian" => () -> model_v9_continuous_gaussian(cont_covs_dummy, MARGS),
    "v10_deep_gp_3layer"  => () -> model_v10_deep_gp_3layer_gaussian(MARGS)
)

for m_key in sort(collect(keys(models_to_bench)))
    print("Benchmarking $m_key... ")
    try
        m_instance = models_to_bench[m_key]()
        t = @elapsed sample(m_instance, NUTS(), n_bench_iters; progress=false)
        bench_results[m_key] = t
        println("$(round(t, digits=2))s")
    catch e
        println("FAILED")
        bench_results[m_key] = NaN
    end
end

# --- Display Summary Table ---
println("\n" * "="^35)
println(rpad("Model", 25), " | ", "Time (s)")
println("-"^35)
for m_key in sort(collect(keys(bench_results)))
    time_val = isnan(bench_results[m_key]) ? "Error" : string(round(bench_results[m_key], digits=2))
    println(rpad(m_key, 25), " | ", time_val)
end
println("="^35)



# ---------- to view so plots:

mod_fns =  collect(keys(model_registry))

i = 1 # 1:9

@load_carstm_state( mod_fns[i] )

using Turing

# 1. Instantiate Model v2
mod_v2 = model_v2_rff_gaussian(MARGS)

# 2. Define the Optimized Gibbs Sampler
# We partition the parameters by their mathematical properties
optimal_gibbs = Gibbs(
    # Gaussian Latents: Use ESS for optimal movement without tuning
    (:u_icar, :u_iid, :w_trend, :w_seas, :st_int_raw) => ESS(),

    # Regression Coefficients: Use NUTS for adaptive gradient-based exploration
    (:beta_cov) => NUTS(),

    # Variance Components: Use MH for simple scalar parameters
    (:sigma_y, :sigma_sp, :phi_sp, :sigma_trend, :sigma_seas, :sigma_int, :sigma_rw2) => MH()
)

# 3. Execute Production-Scale Sampling
# Running a moderate chain for verification; scale iterations to 2000+ for production
println("Starting Optimized Mixed-Sampler Gibbs for Model 2...")
chain_v2_optimized = sample(mod_v2, optimal_gibbs, 10; progress=true)
 
 

# --- MAP Optimization Benchmarking ---
# Maximum A Posteriori (MAP) provides a point estimate by maximizing the posterior density.

using Turing, Optim

println("Running MAP Optimization for model_v1_gaussian...")

# 1. Instantiate the model using the stable precomputations
m_v1 = model_v1_gaussian(MARGS)

# 2. Perform MAP optimization using the correct library path
# Using Optim.optimize directly to avoid ambiguity and fix the module nesting error
t_map = @elapsed begin
    map_res_v1 = maximum_a_posteriori(m_v1 )
end

println("Optimization finished in $(round(t_map, digits=2)) seconds.")

# 3. Display summary of estimates
println("\n--- MAP Estimates for Model V1 ---")
display(map_res_v1)


chain = sample(m_v1, NUTS(), 1_000; initial_params=InitFromParams(map_res_v1))



# ---------

map_results = Dict{String, Any}()

println("Starting Suite MAP Optimization...\n")

for m_key in sort(collect(keys(models_to_bench)))
    print("Optimizing $m_key (MAP)... ")
    try
        m_instance = models_to_bench[m_key]()
        # Use LBFGS for numerical optimization of the log-joint
        t = @elapsed begin
            map_res = optimize(m_instance, MAP())
        end
        map_results[m_key] = (result = map_res, time = t)
        println("$(round(t, digits=2))s")
    catch e
        println("FAILED")
        map_results[m_key] = nothing
    end
end

# --- Display MAP Summary ---
println("\n" * "="^45)
println(rpad("Model", 25), " | ", rpad("Time (s)", 10), " | ", "LP")
println("-"^45)
for m_key in sort(collect(keys(map_results)))
    if !isnothing(map_results[m_key])
        res = map_results[m_key]
        lp = round(res.result.lp, digits=2)
        println(rpad(m_key, 25), " | ", rpad(string(round(res.time, digits=2)), 10), " | ", lp)
    else
        println(rpad(m_key, 25), " | ", "Error")
    end
end
println("="^45)



# ----- 
# Variational Inference
 using Turing, AdvancedVI

using AdvancedVI

# 1. Setup the ADVI algorithm
# Using 1 sample for the gradient estimate and 1000 iterations
advi = ADVI(1, 1000)

# 2. Run the optimization
println("Starting ADVI for model_v1_gaussian...")
t_vi = @elapsed begin
    q_v1 = vi(m_v1, advi)
end

println("ADVI finished in $(round(t_vi, digits=2)) seconds.")


using Optim

# 1. Optimized MAP with L-BFGS
# We pass specific Optim options to allow for better convergence monitoring
println("Starting Optimized MAP (L-BFGS)...")
map_res_optimized = optimize(m_v1, MAP(), LBFGS())

# 2. Optimized ADVI (Multi-sample gradient)
# ADVI(n_samples, n_iterations)
# Increasing samples to 10 reduces noise in high-dimensional ST fields
println("Starting Optimized ADVI (10 samples per grad)...")
advi_optimized = ADVI(10, 1500)
q_v1_opt = vi(m_v1, advi_optimized)

println("Optimization Complete.")




vi_results = Dict{String, Any}()
n_vi_iters = 1000

println("Starting Suite Variational Inference (ADVI)...")

for m_key in sort(collect(keys(models_to_bench)))
    print("Running VI for $m_key... ")
    try
        m_instance = models_to_bench[m_key]()

        # Standard Mean-Field ADVI
        advi = ADVI(1, n_vi_iters)

        t = @elapsed begin
            # Solve for the variational posterior
            q = vi(m_instance, advi)
        end

        vi_results[m_key] = (dist = q, time = t)
        println("$(round(t, digits=2))s")
    catch e
        println("FAILED: $e")
        vi_results[m_key] = nothing
    end
end

# --- Display VI Summary Table ---
println("\n" * "="^45)
println(rpad("Model", 25), " | ", rpad("Time (s)", 10))
println("-"^45)
for m_key in sort(collect(keys(vi_results)))
    if !isnothing(vi_results[m_key])
        res = vi_results[m_key]
        println(rpad(m_key, 25), " | ", rpad(string(round(res.time, digits=2)), 10))
    else
        println(rpad(m_key, 25), " | ", "Error")
    end
end
println("="^45)

```

  

## Example 1: Bottom temperatures

See the INLA-based (Laplace-Approximation) implementation here:
<https://github.com/jae0/carstm/blob/master/inst/scripts/example_temperature_carstm.md>

Here we re-implement this as a fully Bayesian process with Julia, Turing
and the [supporting functions in this repository](https://github.com/jae0/model_covariance/)

The main idea is to model spatial variability via a [Conditional
Autoregressive Process or CAR](./spatial_processes.md) and [temporal variability via Fourier terms](./temporal_processes.md). 

First, we begin with a basic regression model with overall mean (intercept) and a linear trend in time ($X=[1, t]$ and any other linear effects, $\beta$), in order to make it spatially and temporally (first order) "stationary": 

$$y \sim N(  \mathbf{\beta} \mathbf{X}, \: \sigma^2)$$

and some random errors $\sigma$. We can decompose the mean process as a [Gaussian covariate process](./gaussian_process.md) associated with depth, $\textbf{GP}(z)$ and potentially any other nonlinear process (we are careful to minimize such processes as they are computationally expensive) with an expected value of zero:

$$y \sim N(  \mathbf{\beta} \mathbf{X} + \textbf{GP}(z) , \: \sigma^2)$$

The mean process can be further decomposed into a [spatial effect](./spatial_processes.md). There are a number of possible forms/parameterizations, the most common being a spatial covariance process (through e.g, a Matern form or an SPDE and so akin to kriging). However, here we use the even simpler ICAR process that only depends upon immediate neighbours in space $s$:

$$y \sim N(  \mathbf{\beta} \mathbf{X} + \textbf{GP}(z) + \textbf{ICAR}(s) , \: \sigma^2)$$

The error can further be decomposed into a periodic time-component. This is modelled simply as either an AR1 or RW1, or in this case as a Fourier terms that model seasonal (period = 1 year) and potentially longer-term periodicities (El Nino - La Nina, etc.), to give:

$$y \sim N( \mathbf{\beta} \mathbf{X} + \textbf{GP}(z) + \textbf{ICAR}(s) + \textit{F}(t)  , \: \sigma^2)$$

Finally, to express different dynamics across space (i.e., space-time interaction, $\textit{F}(t) + \textbf{ICAR}(s,t) $), it is assumed that temporal variability is nested within space:


$$y \sim N( \mathbf{\beta} \mathbf{X} + \textbf{GP}(z) + \textbf{ICAR}(s) + \textit{F}(t) + \textbf{ICAR}(s,t) \otimes \textit{F}(s,t)   , \: \sigma^2)$$

Conditioning of the Fourier parameters across space as a spatial ICAR or other spatial form is also possible, but here not considered as it is more computationally expensive. 

 


#### Data

The data come from various sources. It is a small subset of real data
for the area close to Halifax, Nova Scotia, Canada.

The example data is bounded by longitudes (-65, -62) and latitudes (45,
43). It is stored as test data for carstm. It can be created in R with the sequence in [https://github.com/jae0/carstm/blob/1d5df20e6ee876e78f2a1e66dc1a2f91e90838b8/inst/scripts/example_temperature_carstm.md](example_temperature_carstm.md). Load into julia as follows:

```julia

project_directory = joinpath( homedir(), "projects", "model_covariance"  )

funcs = ( "startup.jl", "pca_functions.jl",  "regression_functions.jl", "car_functions.jl", "carstm_functions.jl" )

download_directly = false
if download_directly
  using Downloads
  project_url = "https://raw.githubusercontent.com/jae0/model_covariance/master/"

  for f in funcs
    include( Downloads.download( string(project_url, f) ))
  end

else 

  for f in funcs
    include( joinpath( project_directory, "src", f) )
  end

end


# include( joinpath( project_directory, "src", "bijectors_override.jl") )

Random.seed!(1); # Set a seed for reproducibility.


# load test data: 1999:2023 
# NOTE: data created in /home/jae/bio/aegis.temperature/inst/scripts/

using RData  

#fndat = "https://github.com/jae0/model_covariance/data/example_bottom_temp.rdz"

#fn = Downloads.download(fndat)  # save rdz locally
fn = joinpath( project_directory, "data", "example_bottom_temp.rdz" )

bt = RData.load( fn, convert=true)

# W = nb_to_adjacency_matrix( bt["nb"] )

node1, node2, scaling_factor = nodes( bt["nb"] ) # pre-compute required vars from adjacency_matrix outside of modelling step

Y = bt["obs"] 

nob, nvar = size(Y)   
nz = 2  # no latent factors to use

# X = linear covars
G = Y[:,["z"]]
G.z = log.(G.z)
nG = size(G,2)

# inducing_points for GP (for prediction)
n_inducing = 10
Gp =  zeros(n_inducing, nG)
for i in 1:nG
  Gp[:,i] = quantile(vec(G[:,i]), LinRange(0.01, 0.99, n_inducing))
end


# log_offset (if any)
nAU = size( bt["nb"], 1 )  # no of au
auid = collect( 1:nAU )
nbeta = 0 # no of covars linear


n_samples = 10  # posterior sampling
sampler = Turing.NUTS()  

# carstm_temperature() # incomplete (see carstm_functions.jl)

Y 
nob=size(Y, 1)
nvar=size(Y, 2)
nz=2
nvh=Int(nvar*nz - nz * (nz-1) / 2)
noise=1e-9 

# Fixed (covariate) effects 
#f_beta ~ filldist( Normal(0.0, 1.0), nbeta);
#f_effect = X * f_beta + log_offset

# icar (spatial effects)
beta_s ~ filldist( Normal(0.0, 1.0), nbeta); 
s_theta ~ filldist( Normal(0.0, 1.0), nAU)  # unstructured (heterogeneous effect)
s_phi ~ filldist( Normal(0.0, 1.0), nAU) # spatial effects: stan goes from -Inf to Inf .. 
dphi_s = s_phi[node1] - s_phi[node2]
Turing.@addlogprob! (-0.5 * dot( dphi_s, dphi_s ))
sum_phi_s = sum(s_phi) 
sum_phi_s ~ Normal(0, 0.001 * nAU);      # soft sum-to-zero constraint on s_phi)
s_sigma ~ truncated( Normal(0.0, 1.0), 0, Inf) ; 
s_rho ~ Beta(0.5, 0.5);

# spatial effects:  nAU
convolved_re_s = s_sigma .*( sqrt.(1 .- s_rho) .* s_theta .+ sqrt.(s_rho ./ scaling_factor) .* s_phi )
mp_icar =  X * beta_s +  convolved_re_s[auid]  # mean process for bym2 / icar

# GP (higher order terms)
# kernel_var ~ filldist(LogNormal(0.0, 0.5), nG)
# kernel_scale ~ filldist(LogNormal(0.0, 0.5), nG)

# k = ( kernel_var[1] * SqExponentialKernel() ) ∘ ScaleTransform(kernel_scale[1])

# variance process  
# gp = atomic( Stheno.GP(k), Stheno.GPC())
# gpo = gp(Xo, I2reg )
# gpp = gp(Xp, eps() )
# sfgp = SparseFiniteGP(gpp, gpp)
# vcv = cov(sfgp.fobs)

#    --- add more .. but kind of slow 
#    --- ... looking at AbstractGPs as a possible solution

# gps = rand( MvNormal( mean_process, Symmetric(kmat) ) ) # faster
# mp_gp = sum(gps, dims=1)  # mean process



# Fourier process (global, main effect)
t_period ~ filldist( LogNormal(0.0, 0.5), ncf ) 
t_beta ~ Normal(0, 1)  # linear trend in time
t_amp ~ MvNormal(Zeros(ncf), I) #  coefficients of harmonic components
t_phase ~ MvNormal(Zeros(ncf), I) #  coefficients of harmonic components
# t_error ~ LogNormal(0, 1)

 # fourier effects
mu_fp = t_beta .* ti + 
    t_amp[1] .* cos.(t_phase[1]) .* sin.( (2pi / t_period[1]) .* ti )   + 
    t_amp[1] .* sin.(t_phase[1]) .* cos.( (2pi / t_period[1]) .* ti )   +
    t_amp[2] .* cos.(t_phase[2]) .* sin.( (2pi / t_period[2]) .* ti )   + 
    t_amp[2] .* sin.(t_phase[2]) .* cos.( (2pi / t_period[2]) .* ti ) 

# mp_fp = rand( MvNormal( mu_fp, t_error^2 * I ) )  

# space X time


Y ~ MvNormal( mu_fp .+ mp_icar, Symmetric(vcv) )   # add mvn noise


```

#### Model

#### Results




## Example 2: Species Composition

See the [INLA-based (Laplace-Approximation) implementation.](https://github.com/jae0/aegis.speciescomposition/blob/master/inst/scripts/01_speciescomposition_carstm_1999_to_present.R)

Here we re-implement this as a fully Bayesian process with Julia, Turing
and the [supporting functions in this repository](https://github.com/jae0/model_covariance/)


Similar to Example 1, the main idea is to model spatial variability via a [Conditional
Autoregressive Process or CAR](./spatial_processes.md) and [temporal variability via Fourier terms](./temporal_processes.md). We begin with the same model:


$$y \sim N( \mathbf{\beta} \mathbf{X} + \textbf{GP}(z) + \textbf{ICAR}(s) + \textit{F}(t) + \textbf{ICAR}(s,t) \otimes \textit{F}(s,t)   , \: \sigma^2)$$

but note that $y^{n \times k}$ are mean centered observations of n data points and k-species which is represented as a multivariate latent process $Z^{n \times p}$ with p latent factors and latent-eigenvectors $W^{k \times p}$ and variance $\sigma^2 I$ (k latent-eigenvalues):

$$\mathbf{y} \sim \text{N} (\mathbf{Z} \mathbf{W}^T  + \mathbf{\beta} \mathbf{X} + \textbf{GP}(z, bt) + \textbf{ICAR}(s) + \textit{F}(t) + \textbf{ICAR}(s,t) \otimes \textit{F}(s,t),  \sigma^2 \mathbf{I})$$

The computation of each component is relatively simple, however, to improve parameter estimation and sampling efficiency, we use a [Householder transformation to ensure rotationally invariant solutions](./pca.md). 

But first prepare the data. This uses the [aegis.speciescomposition R library](https://github.com/jae0/aegis.speciescomposition/) to prepare the data and format it. As the purpose of this is to run a complex model in Julia, we do data manipulations outside of Julia unless there is a specific advantage to do so. 




$$y(t) = A \sin( \frac{2 \pi} {\tau}  t + B ) ) + C$$
 
$$A \sin(\frac{2 \pi} {k} t + B) = A \cos(B)  \sin(\frac{2 \pi} {\tau} t) + A \sin(B)  \cos(\frac{2 \pi} {\tau} t).$$

amplitude = sqrt.(b[:,1].^2 .+ b[:,2].^2)
phaseshift = atan.( abs.(b[:,1] ./ b[:,2]) )


Make data in R:

```r

  year.assessment = 2023

  yrs = 1999:year.assessment
  
  carstm_model_label="default"
  require(aegis)
  require(aegis.speciescomposition)
  require(vegan)

  p = speciescomposition_parameters( yrs=yrs, carstm_model_label=carstm_model_label )


  variabletomodel = "pca1"  # dummy for now

  p0 = speciescomposition_parameters(
    project_class="carstm",
    data_root = project.datadirectory( "aegis", "speciescomposition" ),
    variabletomodel = "",  # will b eover-ridden .. this brings in all pca's and ca's
    carstm_model_label = carstm_model_label,
    carstm_model_label = carstm_model_label,
    inputdata_spatial_discretization_planar_km = 0.5,  # km controls resolution of data prior to modelling to reduce data set and speed up modelling
    inputdata_temporal_discretization_yr = 1/52,  # ie., every 1 weeks .. controls resolution of data prior to modelling to reduce data set and speed up modelling
    year.assessment = max(yrs),
    yrs = yrs, 
    spatial_domain = "SSE",  # defines spatial area, currenty: "snowcrab" or "SSE"
    areal_units_proj4string_planar_km = aegis::projection_proj4string("utm20"),  # coord system to use for areal estimation and gridding for carstm
    areal_units_type = "tesselation",     
    areal_units_constraint="none",
    #areal_units_resolution_km = 1, # km dim of lattice ~ 1 hr
    # areal_units_overlay = "none",
    # spbuffer=5, lenprob=0.95,   # these are domain boundary options for areal_units
    # n_iter_drop=0, sa_threshold_km2=4, 
    # areal_units_constraint_ntarget=10, areal_units_constraint_nmin=1,  # granularity options for areal_units
    carstm_prediction_surface_parameters = list( 
      bathymetry = aegis.bathymetry::bathymetry_parameters( project_class="stmv" ),
      substrate = aegis.substrate::substrate_parameters(   project_class="stmv" ),
      temperature = aegis.temperature::temperature_parameters( project_class="carstm", spatial_domain="canada.east", yrs=1999:year.assessment, carstm_model_label="default" ) 
    ), 
   
  )

 
   
  # construct basic parameter list defining the main characteristics of the study
  p0$formula = NULL  # MUST reset to force a new formulae to be created on the fly below 
  p = speciescomposition_parameters( 
    p=p0, 
    project_class="carstm", 
    variabletomodel = variabletomodel, 
    yrs=p0$yrs, 
    # required
    carstm_model_label=carstm_model_label
  )  

  # update data files for external programs (e.g., carstm_julia)
  sppoly = areal_units( p=p0)
  nb = attributes(sppoly)$nb$nbs
  M = speciescomposition_db( p=p0, DS="carstm_inputs", sppoly=sppoly)
  
  M_preds = M[ M$tag=="predictions", ]
  M_obs   = M[ M$tag=="observations", ]

  outputfile = file.path(p$project_data_directory, "sps_comp.rdz")  # alter this to suite your needs

  redo_data = FALSE
  if (redo_data) {

    survey_data = survey_data_prepare(p=p, cthreshold = 0.005)
    set = survey_data$set
    
    m = data.table(survey_data$m)   # order needs to change to that of M_obs
    m$id = rownames(survey_data$m)
    m$m_order=1:nrow(m)

    set = set[  M_obs, on="id" ] 
    set$oorder = 1:nrow(set)

    m = set[,.(id, oorder)][m, on="id" ] 
    m = m[ is.finite(oorder), ]
    m = m[ order(oorder), ]
    ids = m$id

    m$m_order = NULL
    m$oorder = NULL
    m$id = NULL
 
    taxa = colnames(m)
  
    read_write_fast( data=list( set=set, m=m, nb=nb, obs=obs, preds=preds, taxa=taxa, ids=ids), fn=outputfile )

    # devtools::install_github("wesm/feather/R")
    # require(feather)
    
    #  rootdir = file.path("/home", "jae", "projects", "model_covariance", "data" )
    #  rootdir = p$project_data_directory

    #  py_save_object(set, file.path(rootdir, "set.pickle") )
    #  py_save_object(m, file.path(rootdir, "m.pickle") )
    #  py_save_object(obs, file.path(rootdir, "obs.pickle") )
    #  py_save_object(preds, file.path(rootdir, "preds.pickle") )
    #  py_save_object(taxa, file.path(rootdir, "taxa.pickle") )
    #  py_save_object(ids, file.path(rootdir, "ids.pickle") )
    #  py_save_object(nb, file.path(rootdir, "nb.pickle") )
  
  }

  data = read_write_fast(outputfile) 
  attach(data)
  
```

Now bring data into julia for analysis

```julia

    # y ∼ N(ZW^T +βX+GP(z,bt)+ICAR(s)+F(t)+ICAR(s,t)⊗F(s,t), σ^2 I)

    project_directory = joinpath( homedir(), "projects", "model_covariance"  )

    funcs = ( "startup.jl", "pca_functions.jl",  "regression_functions.jl", "car_functions.jl", "carstm_functions.jl" )

    for f in funcs
      include( joinpath( project_directory, f) )
    end

    # using Downloads
    # project_url = "https://raw.githubusercontent.com/jae0/model_covariance/master/"
    for f in funcs
      # include( download( string(project_url, f) ))
    end
 
    # second passs sometimes required ..not sure why
    for f in funcs
      include( joinpath( project_directory, f) )
      # include( download( string(project_url, f) ))
    end
 
    Random.seed!(1); # Set a seed for reproducibility.

    # include( joinpath( project_directory, "bijectors_override.jl") )


    # load test data: 1999:2023 
    # NOTE: data created in /home/jae/bio/aegis.speciescomposition/inst/scripts/01_speciescomposition_carstm_1999_to_present.R

    # fn = "https://github.com/jae0/model_covariance/raw/master/data/sps_comp.rdz"
    # fndat = joinpath( tempdir(), "sps.rdz" )
    # Downloads.download(fn, fndat )  # save rdz locally

    fndat = "/archive/bio.data/aegis/speciescomposition/data/sps_comp.rdz" 
    sps = RData.load( fndat, convert=true)

    # M, set, m, nb
#    Y = Matrix(sps["m"]) 
    Y = Matrix(sps["m"]) .- 0.5  # Y ranges from 0 to 1 .. make it symetrical around 0
   #  Y = Y .- mean(Y) # qscore abundance of species by each set (0,1)  center to mean
    id = 1:size(Y,1)
    grps = 1:size(Y,1)
    vn = sps["taxa"]

    # basic pca ..
    evecs, evals, pcloadings, variancepct, C, pcscores = pca_standard(Y; model="cor_pairwise", obs="rows", scale=false, center=false )  # sigma is std dev, not variance.

    biplot(pcscores=pcscores, pcloadings=pcloadings,  evecs=evecs, evals=evals, vn=vn, variancepct=variancepct, type="unstandardized"  )   
    #  plot!(xlim=(-2.5, 2.5))
    

    using RCall
    # NOTE: <$>  activates R REPL <backspace> to return to Julia
    
    @rput evals evecs pcloadings pcscores

    R"""
    read_write_fast( data=list( evals=evals, evecs=evecs, pcloadings=pcloadings, pcscores=pcscores), fn='/archive/bio.data/aegis/speciescomposition/data/carstm_pca_simple.rdz' )
    """

    # W = nb_to_adjacency_matrix( sps["nb"] )

    node1, node2, scaling_factor = nodes( sps["nb"] ) # pre-compute required vars from adjacency_matrix outside of modelling step
    nnodes = length(node1)

    # M, set, m, nb

    Y = Matrix(sps["m"]) .- 0.5  # Y ranges from 0 to 1 .. make it symetrical around 0
    otime = sps["obs"][:,"year"] + sps["obs"][:,"dyear"]

    nob, nvar = size(Y)   
    nz = 2  # no latent factors to use
    
    ncf = 1  # 1 for seasonal 1 for interannual ..

    # X = linear covars
    G = sps["obs"]
    G = G[:,["z", "t"]]
    G.z = log.(G.z)
    nG = size(G,2)

    # inducing_points for GP (for prediction)
    n_inducing = 10
    Gp =  zeros(n_inducing, nG)
    for i in 1:nG
      Gp[:,i] = quantile(vec(G[:,i]), LinRange(0.01, 0.99, n_inducing))
    end


    # log_offset (if any)
    nb = sps["nb"]
    nAU = size( nb, 1 )    # no of au
    nAU_float = convert(Float64, nAU)
    auid = parse.(Int, sps["obs"][:,"AUID"])
    X = 1.0
    nbeta = size(X, 2) # no of covars linear

    n_samples = 500  # posterior sampling
    turing_sampler = Turing.NUTS()  
    
    nvh, hindex, iz, ltri = PCA_BH_indexes( nvar, nz )  # indices reused .. to avoid recalc ...
    ti = otime .- mean(otime)
    ti2pi = ti .* 2.0 * pi

    noise=1.0e-9 

    t_period_prior = log.([1 5; 1 5])[:, 1:ncf]


    v_prior = eigenvector_to_householder(evecs, nvar, nz, ltri )  
    # householder_to_eigenvector( lower_triangle( v_prior, nvar, nz ) ) .- evecs[:,1:nz] # inverse transform
     
    # param sequence = sigma_noise, sigma(nz), v, r=norm(v)~ 1.0 (scaled)
    sigma_prior = log.(sqrt.(evals)[1:nz])

    # direct ppca
    M0 = ppca_basic( Y' )  # pca first  
    rand(M0)  
    res0 = sample(M0, Prior(), 10 ) #; init_params=init_params, init_ϵ=init_ϵ, 
    init_params0 = init_params_extract(res0)
    res0 = sample(M0, Turing.SMC(), 1000,  init_params=init_params0) # cannot be larger than 1000 , so iteratively restart

Summary Statistics
   parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
       Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

    pca_sd[1]    1.4183    0.4638     0.1467    0.1311    12.3302    0.9197        1.0520
    pca_sd[2]    4.7365    2.0230     0.6397    0.9972    13.8004    1.0099        1.1774
  pca_pdef_sd    1.0386    0.5007     0.1583    0.1841    54.8457    0.9124        4.6793
         v[1]   -0.0652    0.8898     0.2814    0.2139   178.4505    0.9027       15.2249
         v[2]    0.0513    1.0359     0.3276    0.0858   -10.9106    0.8945       -0.9309
         v[3]   -0.0290    0.8240     0.2606    0.2028     8.6221    0.9354        0.7356


    # ppca and carstm
    M = pca_carstm2( Y, ti)  # pca first and then carstm ... like species comp analysis
    rand(M)  
    res = sample(M, Prior(), 10 ) #; init_params=init_params, init_ϵ=init_ϵ, 
    init_params = init_params_copy(res, res0)
    rand(M)

    #  
    # carstm_pca() # incomplete (see car_functions.jl) ... carstm first and then pca ... like msmi 
    
    # init_params = init_params_extract(res, load_from_file=true) 
    init_params = init_params_extract(res)

    # res = optimize(M, MLE(), Optim.Options(iterations=100) )

    # res = sample(M, Turing.NUTS(), 100) # ; init_params=init_params, init_ϵ=0.01)
  

    Turing.setadbackend(:enzyme)

    Turing.setadbackend(:forwarddiff) 

    res = optimize(M, MAP())

    res = optimize(M, MLE())
    
    res = optimize(M, MLE(), LBFGS(), Optim.Options(iterations=100))
    res = optimize(M, MLE(), NelderMead())
    res = optimize(M, MLE(), SimulatedAnnealing())
    res = optimize(M, MLE(), ParticleSwarm())
    res = optimize(M, MLE(), Newton())
    res = optimize(M, MLE(), AcceleratedGradientDescent(), Optim.Options(iterations=100) )
    res = optimize(M, MLE(), Newton(), Optim.Options(iterations=100, allow_f_increases=true))
 
 
    # to do Variational Inference  
    samples_per_step, max_iters = 5, 100  # Number of samples used to estimate the ELBO in each optimization step.
    res_vi =  vi(M, Turing.ADVI( samples_per_step, max_iters)  ); 
    res_vi_samples = rand( res_vi, 1000)  # sample via simulation


    # turing_sampler = Turing.PG(2)    

    turing_sampler = Turing.SMC()   #   
    
    # turing_sampler = Turing.SGLD()   # Stochastic Gradient Langevin Dynamics (SGLD); slow, mixes poorly
    # turing_sampler = Turing.NUTS( 0.65 ) # , init_ϵ=0.001

    res = sample(M, turing_sampler, 1000,  init_params=init_params) # cannot be larger than 1000 , so iteratively restart

    arviz_plots = false
    if arviz_plots
        begin
            plot_autocorr(res; var_names=(:pca_sd, :eta))
           
        end

        idata_turing_post = from_mcmcchains(
            res;
            coords=(; school=schools),
            dims=NamedTuple(k => (:school,) for k in (:y, :σ, :θ)),
            library="Turing",
        )
        begin
            plot_trace(idata_turing_post)
           
        end

        begin
            prior = Turing.sample(rng2, M, Prior(), n_samples);
            # Instantiate the predictive model
            param_mod_predict = model_turing(similar(y, Missing), σ)
            # and then sample!
            prior_predictive = Turing.predict(rng2, param_mod_predict, prior)
            posterior_predictive = Turing.predict(rng2, param_mod_predict, res)
        end;

    # And to extract the pointwise log-likelihoods, which is useful if you want to compute metrics such as loo,

        log_likelihood = let
            log_likelihood = let
            log_likelihood = Turing.pointwise_loglikelihoods(
                param_mod_turing, MCMCChains.get_sections(res, :parameters)
            )
            # Ensure the ordering of the loglikelihoods matches the ordering of `posterior_predictive`
            ynames = string.(keys(posterior_predictive))
            log_likelihood_y = getindex.(Ref(log_likelihood), ynames)
            (; y=cat(log_likelihood_y...; dims=3))
        end;

        idata_turing = from_mcmcchains(
            res;
            posterior_predictive,
            log_likelihood,
            prior,
            prior_predictive,
            observed_data=(; y),
            coords=(; school=schools),
            dims=NamedTuple(k => (:school,) for k in (:y, :σ, :θ)),
            library=Turing,
        )
        # etc: https://julia.arviz.org/ArviZ/stable/quickstart/

        loo(idata_turing) # higher ELPD is better
        begin
            plot_loo_pit(idata_turing; y=:y, ecdf=true)
           
        end

    end


 ###############

    for _ in 1:5
      init_params = init_params_extract(res, override_means=true)  # updates a file each time 
      res = sample(M, turing_sampler, 1000,  init_params=init_params) # cannot be larger than 1000 , so iteratively restart
    end

    for _ in 1:5
      init_params = init_params_extract(res, override_means=false)  # updates a file each time 
      res = sample(M, turing_sampler, 1000, drop_warmup=true,  init_params=init_params)  # RAM is a problem ... and sequential ..keep nsamples  ~ 1000  -> 52G
    end
    
    
    turing_sampler = Turing.NUTS( 0.65 ) # , init_ϵ=0.001
    res = sample(M, turing_sampler, 10,  init_params=init_params) # cannot be larger than 1000 , so iteratively restart

    # f = DynamicPPL.LogDensityFunction(M);
    # DynamicPPL.link!!(f.varinfo, f.model);
    # res = sample(f, AdvancedHMC.NUTS(0.65), 10; init_params=init_params) # RAM is a problem (1chain=52 GB)
    # ; init_ϵ=0.01) #; init_params=init_params, init_ϵ=init_ϵ, drop_warmup=true, progress=true);

    summarystats(res)

    # posterior_summary(res, sym=:pca_sd, stat=:mean, dims=(1, nz))

    # sqrt(eigenvalues) 
    #    note no sort order from chains 
    # .. must access through PCA_posterior_samples to get the order properly
    
    pca_sd, evals, evecs, pcloadings, pcscores = 
        PCA_posterior_samples( res, Y, nz=nz, model_type="householder" )
 
    evecs_mean = DataFrame( convert(Array{Float64}, mean(evecs, dims=1)[1,:,:]), :auto)
    pcloadings_mean = DataFrame( convert(Array{Float64}, mean(pcloadings, dims=1)[1,:,:]), :auto)
    pcscores_mean = DataFrame( convert(Array{Float64}, mean(pcscores, dims=1)[1,:,:]), :auto)
    # pcscores_mean = reshape(mapslices( mean, pcscores, dims=1 ), (nob, nz))
     
    pl = plot( pcscores_mean[:,1], pcscores_mean[:,2], label=:none, seriestype=:scatter )

    j = 2  # observation index
    # variability of a single solution     
        plot!(
            pcscores[:, j, 1], pcscores[:, j, 2];
            # xlim=(-6., 6.), ylim=(-6., 6.),
            # group=["Setosa", "Versicolor", "Virginica"][id],
            # markercolor=["orange", "green", "grey"][id[j]], markerstrokewidth=0,
            seriesalpha=0.1, label=:none, title="Ordination",
            seriestype=:scatter
        )
     
    display(pl)
   
    
    for i in 1:n_samples
        plot!(
            pcscores[i, :, 1], pcscores[i, :, 2]; markerstrokewidth=0,
            seriesalpha=0.1, label=:none, title="Ordination",
            seriestype=:scatter
        )
    end
    display(pl)
   
    

    f_intercept = DataFrame(group(res, "f_intercept"))
    eta = DataFrame(group(res, "eta"))

    pca_sd = DataFrame(group(res, "pca_sd"))

    t_amp = DataFrame(group(res, "t_amp"))
    t_period = DataFrame(group(res, "t_period"))
    t_phase = DataFrame(group(res, "t_phase"))

    # icar (spatial effects)
    s_theta = DataFrame(group(res, "s_theta"))
    s_phi = DataFrame(group(res, "s_phi"))
    s_sigma = DataFrame(group(res, "s_sigma"))
    s_rho = DataFrame(group(res, "s_rho"))
    

    nchains = size(res)[3]
    nsims = size(res)[1]
    n_sample = nchains * nsims
    convolved_re_s = zeros(nAU, n_sample, nz)   
    for sp in 1:nz
    f = 0
    for l in 1:nchains
    for j in 1:nsims
        f += 1
        s_sigma =  res[j, Symbol("s_sigma[$sp]"), l]
        s_rho   =  res[j, Symbol("s_rho[$sp]"), l] 
        s_theta = [res[j, Symbol("s_theta[$k,$sp]"), l] for k in 1:nAU] 
        s_phi   = [res[j, Symbol("s_phi[$k,$sp]"), l] for k in 1:nAU]  
        convolved_re_s[:, f, sp] =  s_sigma .* ( 
          sqrt.(1.0 .- s_rho) .* s_theta .+ 
          sqrt.( s_rho ./ scaling_factor) .* s_phi
        )  # spatial effects nAU
    end  
    end
    end

    # auid = parse.(Int, sps["obs"][:,"AUID"])
    

    using RCall
    # NOTE: <$>  activates R REPL <backspace> to return to Julia
    
    @rput f_intercept eta t_amp t_period t_phase 
     @rput  s_theta s_phi s_sigma s_rho convolved_re_s #copy data to R
    @rput pca_sd evals evecs pcloadings pcscores

    R"""
    read_write_fast( 
      data=list(
         pca_sd=pca_sd, evals=evals, evecs=evecs, pcloadings=pcloadings, pcscores=pcscores, f_intercept=f_intercept, eta=eta, t_amp=t_amp, t_period=t_period, t_phase=t_phase, 
         s_theta=s_theta, s_phi=s_phi, s_sigma=s_sigma, s_rho=s_rho, convolved_re_s=convolved_re_s),
      fn='/archive/bio.data/aegis/speciescomposition/data/carstm_pca.rdz' )
    """

    # save a few data files for use outside Julia to hdf5
    # using HDF5

    # # more option: https://juliaio.github.io/HDF5.jl/stable/
  
    # fn = "/archive/bio.data/aegis/speciescomposition/data/carstm_pca.h5"
    # fid = h5open(fn, "w")

    # fid["pca_sd"] = Array(pca_sd )
    # attrs(fid["pca_sd"])["dimnames"] = String.( names(t_amp) )

    # close(fid)

    # h5write(fn, "evals", evals )
    # h5write(fn, "evecs", evecs )
    # h5write(fn, "pcloadings", pcloadings )
    # h5write(fn, "pcscores", pcscores )
    # # add moreas required:

    # t_amp = group(res, "t_amp")
    # t_period = group(res, "t_period")
    # t_phase = group(res, "t_phase") 
    # f_intercept = group(res, "f_intercept"); 

    # h5write(fn, "t_amp",  Array(t_amp) )
    # h5write(fn, "t_period", Array(t_period) )
    # h5write(fn, "t_phase", Array(t_phase) )
    # h5write(fn, "f_intercept", Array(f_intercept) ) 
 
    # pcscores = h5read(fn, "pcscores" )  #eg


```
Import the data back to R and map it (could do it in julia -- todo -- but infrastructure already in R)

```r

    install_libs = FALSE
    if (install_libs) {
      install.packages("BiocManager")
      BiocManager::install("rhdf5")
    }

    library(rhdf5)

    run_examples_hdf = false
    if (run_examples_hdf) {
        fn = file.path( "~/tmp", "test.h5" )
        h5createFile(fn)
        # heirarchies 
        h5createGroup(fn, "foo")
        h5createGroup(fn, "foo/foobaa")
        h5ls(fn)  # list objects
        A = matrix(1:10,nr=5,nc=2)
        h5write(A, fn, "foo/foobaa")
        H = list(e=2, f=c(1,2), g=matrix(0, 2,3))
        h5write(H, fn, "H")
        h5ls(fn)
        F = h5read(fn, "foo/foobaa")
        k = h5read(fn, "H/e")
    }
  

  fn = "/archive/bio.data/aegis/speciescomposition/data/carstm_pca.h5"
  convolved_re_s = h5read(fn, "convolved_re_s" ) 
   
   
  # bbox = c(-71.5, 41, -52.5,  50.5 )
  additional_features = features_to_add( 
      p=p0, 
      isobaths=c( 100, 200, 300, 400, 500  ), 
      xlim=c(-80,-40), 
      ylim=c(38, 60) , redo=TRUE
  )


  res = carstm_model( p=p, DS="carstm_randomeffects"  ) # to load currently saved results

  # pure spatial effect
  
  outputdir = "~/tmp"

  fn_root = paste( "speciescomposition", variabletomodel, "spatial_effect", sep="_" )
  outfilename = file.path( outputdir, paste(fn_root, "png", sep=".") )


  # carstm_julia results:
  if (soln =="turing")
    # PPCA solution of persistent spatial effects
    res = read_write_fast("/archive/bio.data/aegis/speciescomposition/data/carstm_pca.rdz")
    vn = "toplot"
    res$toplot = toplot = rowMeans(convolved_re_s[,,1])
    # toplot  = convolved_re_s[,,2]

  } else if (soln =="direct_simple_julia") {
    # direct pca in julia
    res = read_write_fast("/archive/bio.data/aegis/speciescomposition/data/carstm_pca_simple.rdz")
    set$pc1 = pcscores[,1]
    set$pc2 = pcscores[,2]
    set$AUID = obs$AUID
    oo = set[,.(pc1=mean(pc1), pc2=mean(pc2)), by="AUID" ]
    oo = oo[ sppoly, on="AUID" ]
    
    vn = "toplot"
    res$toplot = toplot = oo$pc1
    # res$toplot = toplot = oo$pc2

  } else if (soln=="carstm") {
    vn=c( "random", "space", "re_total" )
  
    toplot = carstm_results_unpack( res, vn )

  } else if (soln=="carstm_direct") {
    
    set$pc1 = obs$pca1  
    set$pc2 = obs$pca2
    set$AUID = obs$AUID
    oo = set[,.(pc1=mean(pc1), pc2=mean(pc2)), by="AUID" ]
    oo = oo[ sppoly, on="AUID" ]
    
    vn = "toplot"
    res$toplot = toplot = oo$pc1
    # toplot[,"mean"] = oo$pc2

  }

  

  brks = pretty(  quantile(toplot, probs=c(0.025, 0.975), na.rm=TRUE )  )

  plt = carstm_map(  res=res, vn=vn, 
    sppoly = sppoly, 
    colors= (RColorBrewer::brewer.pal(5, "RdYlBu")),
    breaks = brks,
    annotation=paste("Species composition: ", variabletomodel, "persistent spatial effect" ), 
    legend.position.inside=c( 0.1, 0.9 ),
    additional_features=additional_features,
    outfilename=outfilename
  )
 
 
```


```

Bottom line: the model is too slow... might be usable with GPU based solution but for now it is just a proof of concept


Next trying to implement the same thing but using [PYMC/numpyro (jax)](./carstm_python.md)



```julia




```


## Example 3: Snow crab habitat and abundance (Hurdle)

See the INLA-based (Laplace-Approximation) implementation here:
<https://github.com/jae0/bio.snowcrab/blob/master/inst/markdown/03.biomass_index_carstm.md>

Here we re-implement this as a fully Bayesian process with Julia, Turing
and the [supporting functions in this repository](https://github.com/jae0/model_covariance/)
