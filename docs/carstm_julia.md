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

This document shows how to build a CARSTM regression, optionally with a
pPCA, using bottom temperature (Gaussian, space-time, with time as
Fourier harmonics; (temporal processes)\[./temporal_processes.md\] ) and
species composition (Multivariate Normal, latent with Householder
transform, space-time; [spatial processes](./spatial_processes.md)) and snow crab (Hurdle,
binomial and Poisson, space-time) in the Maritimes Region of Canada.



## Areal units: partitioning of space

Areal Unit Modelling (CARSTM), the choice of spatial partition directly impacts the identifiability of temporal and spatial latent effects. A well-constructed partition must balance geometric compactness with statistical information density.

Multiple methods exist for partitioning spatial data. A few of the simpler ones are examined.

Specific criteria:
  - **Geometric Regularity**: How compact and convex the tiles are.
  - **Information Balance**: The variance of points per tile (lower is better for CAR stability).
  - **Temporal Coverage**: The minimum number of unique time points present in any single tile.



### Start environment

```{julia}

# WARNING: if this is the first run, this can take up to 1 hour to install and precompile libraries and their dependencies


using Pkg


### For Areal Units

pkgs_au = ["Random", "Statistics", "LinearAlgebra", "DataFrames",
       "StatsBase", "SparseArrays", "Plots", "StatsPlots", 
        "JLD2", "LibGEOS", "Graphs", "DelaunayTriangulation" ]

Pkg.add(pkgs_au)
Pkg.precompile()

for pk in pkgs_au
  @eval using $(Symbol(pk))
end



### For CARSTM 

pkgs = ["Random",   "Distributions", "Statistics", "MCMCChains", "DataFrames",
        "LinearAlgebra", "Clustering", "StatsBase", "HypothesisTests",
        "JLD2", "FFTW",  "SparseArrays", "StaticArrays", "FillArrays",
         "Bijectors", "DynamicPPL", "AdvancedVI", "Optimisers", "PosteriorStats",  "Turing" ]
 
Pkg.add(pkgs)
Pkg.precompile()

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



The issues we are looking for:

Identifiability issues** in hierarchical models (like ICAR or CARSTM) are important .
  - regions with no data make estimating local parameters like temporal persistence ($\rho$) a challenge
  - Poisson Weighted CVT: "shrink" tiles where data is abundant and "stretch" them where data is scarce. Every areal unit is informative.


1. **Information Balance**: `avt` achieved the highest stability (lowest variance). This is critical for CARSTM models to prevent 'starving' certain spatial units of data, which can cause MCMC convergence issues.
2. **Temporal Identifiability**: All methods maintained full temporal coverage (15 slices), meaning the longitudinal component of the model is identifiable in every unit.
3. **Geometric vs. Statistical Adaptation**: Standard `triangulation` performs poorly because it doesn't adapt to point density, while the **CVT** and **Merge** variants successfully cluster units where data is most abundant, leading to more robust precision matrices for the GMRF components.


Other observations:

1. **Geometric vs. Statistical Adaptation**: 
   - **`cvt`**  prioritize geometric regularlity and compactness. This is excellent for simple spatial models but can lead to 'sparse' units in low-density areas.
   - **`avt`** adaptively shrink tiles in high-density regions. This ensures that even the smallest units have enough points for stable estimation of local effects.

2. **Information Balance & Convergence**: 
   - The **`avt`** method achieved the lowest variance in point counts. This 'information balance' is a critical prerequisite for the CARSTM model's precision matrix to be well-conditioned.
   - The asymptotic convergence at 19 units for the density-aware methods suggests that 19 is the 'natural' granularity for this specific point process dataset.

3. **Temporal Identifiability**: 
   - All methods successfully maintained full temporal coverage (15 slices) across all units. This confirms that the partitioning logic, combined with the point-count constraints, ensures that the longitudinal (AR1) component of the CARSTM model is identifiable in every areal unit.

4. **Edge Handling**: 
   - The automated `boundary_hull` calculation (with dynamic buffering) correctly prevents 'infinite' Voronoi cells at the domain edges, which previously caused artifacts in the adjacency graph (W matrix).


### Basis: Delaunay Triangulation and Voronoi Tesselation

Given a set of centroids $S = \{s_1, s_2, ..., s_n\}$, a Voronoi cell $V_i$ is defined as the set of all points $x$ in the domain such that the distance to $s_i$ is less than or equal to any other centroid $s_j$:
$V_i = \{x \in \mathcal{D} \mid \|x - s_i\| \leq \|x - s_j\|, \forall j \neq i\}$  

**Algorithm:** Uses `DelaunayTriangulation.jl` to compute the dual graph, followed by clipping against the domain boundary using `LibGEOS.jl`.


See information in https://en.wikipedia.org/wiki/Delaunay_triangulation

- Place $k$ seeds across the domain.
- Define boundaries using Delaunay Triangulation and Voronoi tesselation.
- No adaptation to the underlying point process intensity. Seed Placement: needs to be defined.
- High variance of points per unit, meaning some units might be very dense while others are sparse.
- Geometric Regularity: While it produces compact, convex cells, their distribution can be uneven due to the random seed placement.


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

  

## CARSTM

## Example 0: Simulated data

```{julia}

# Recreate Data 
n_pts = 100
n_time = 15

(pts, y_sim, y_binary, time_idx, weights, trials, cov_indices) =
  generate_sim_data(n_pts, n_time; rndseed=42)


# Ensure cov_indices is correctly shaped as an N_obs x 4 matrix
cov_indices_mat = hcat(cov_indices, cov_indices, cov_indices, cov_indices)

trials_sim = ones(Int, length(y_binary)); # For binary outcome, 1 trial per observation
class1_sim = rand(1:13, length(y_binary)); # A categorical variable with 13 levels
class2_sim = rand(1:2, length(y_binary)) ; # A categorical variable with 2 levels
weights_sim = ones(Float64, length(y_binary)); # Assign equal weight to all observations

# Have a look at an example time slice

target_ts = 5 
filtered_pts_for_plot = pts[findall(==(target_ts), time_idx)]
plt = plot_kde_simple( filtered_pts_for_plot; grid_res=100, sd_extension_factor=1.0 )
display( plt )


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

area_idx = spatial_res.assignments ;
W_sym = spatial_res.adjacency_edges ;


n_areas = maximum(area_idx)
n_times = length( unique(time_idx ))
n_groups = N_groups13

plt = plot_spatial_graph(
  pts, area_assignments, centroids, W_sym; 
  title="Density-Prioritized Agglomerative Voronoi Tesselation (Inverse Quadtree)", 
  show_boundaries=true, 
  boundary_hull=hull_pts
)
  
display(plt)


# kde  view data density in space
x_kde, y_kde, intensity_kde = estimate_intensity_kde_optimized(pts)
plt_kde_intensity = heatmap(x_kde, y_kde, intensity_kde', title="KDE Estimated Spatial Density", color=:viridis, aspect_ratio=:equal, colorbar=true, xlabel="X-coordinate", ylabel="Y-coordinate")
display(plt_kde_intensity)


# Generate synthetic data for other models:  
# trials_sim represents the number of trials for each observation.
# For binary data, it's often 1 trial per observation.
# For proportional data, it could be a larger integer.
# class1_sim and class2_sim are categorical fixed effects.
# cov_indices is a 2D matrix (N_obs x 4) as expected by the model.

# cov_indices = rand(n_pts, 4)
cov_indices = hcat( cov_indices, cov_indices, cov_indices, cov_indices )

trials_sim = ones(Int, length(y_binary)) # For binary outcome, 1 trial per observation
class1_sim = rand(1:13, length(y_binary)) # A categorical variable with 13 levels
class2_sim = rand(1:2, length(y_binary))  # A categorical variable with 2 levels
weights_sim = ones(Float64, length(y_binary)) # Assign equal weight to all observations

# Ensure cov_indices is correctly shaped as an N_obs x 4 matrix
cov_indices_mat = hcat(cov_indices, cov_indices, cov_indices, cov_indices)

time_slice = 5  # for plotting
 

# ----------
# basic grmf
# 1. Convert the spatial results into a numerical adjacency matrix

g_v1 = get_spatial_graph(spatial_res)
adj_matrix_numeric = Graphs.adjacency_matrix(g_v1)

mod_v1 = model_v1_gaussian(y_sim, time_idx, area_idx, cov_indices_mat, adj_matrix_numeric)
chain_v1 = sample(mod_v1, NUTS(), 100)


plot_model_fit(chain_v1, y_sim, time_idx, area_idx)

stats_v1 = reconstruct_posteriors(mod_v1, chain_v1, pts, area_idx, time_idx, W_sym; cov_indices=cov_indices_mat)

summarystats( chain_v1 )
calculate_waic(mod_v1, chain_v1)
 
  plt_test = plot_posterior_vs_prior(
      mod_v1,
      chain_v1,
      :sigma_sp,
      title="Model V1: Automated Prior Sensitivity (sigma_sp)"
  )
  
  display(plt_test)
  plot_model_fit(chain_v1, y_sim, time_idx, area_idx)
  
  # PPC Verification for Model V1 - Updated signature
  ppc_v1 = posterior_predictive_check(mod_v1, stats_v1, y_sim)
  display(ppc_v1.plot_density)
  display(ppc_v1.plot_scatter)
  
  plt_spatial = plot_posterior_results(stats_v1, pts, centroids, W_sym; effect=:spatial)
  display(plt_spatial)
  
  plt_pred_t5 = plot_posterior_results(stats_v1, pts, centroids, W_sym; time_slice=5, effect=:st_predictions_denoised)
  display(plt_pred_t5)
  
  plt_prob = plot_posterior_results(stats_v1, pts, centroids, W_sym; time_slice=5, effect=:st_predictions_noisy)
  display(plt_prob)
  
  

 
# ----------
# RFF model with corrected predictor alignment
mod_v2 = model_v2_carstm_rff(y_sim, time_idx, area_idx, cov_indices_mat, W_sym)
chain_v2 = sample(mod_v2, MH(), 500)


stats_v2 = reconstruct_posteriors(mod_v2, chain_v2, pts, area_idx, time_idx, W_sym; cov_indices=cov_indices_mat)
ppc_v2 = posterior_predictive_check(mod_v2, stats_v2, y_sim)

display(ppc_v2.plot_density)
display(ppc_v2.plot_scatter)

plt_prob = plot_posterior_results(stats_v2, pts, centroids, W_sym; time_slice=5, effect=:st_predictions_noisy)
display(plt_prob)


# ----------
# lognormal model
# Ensure strictly positive data for LogNormal
y_lognorm = exp.(y_sim .- mean(y_sim))

mod_v3_ln = model_v3_lognormal(y_lognorm, time_idx, area_idx, cov_indices_mat, W_sym)
println("Sampling Model V3 (LogNormal)...")
chain_v3_ln = sample(mod_v3_ln, MH(), 500)

stats_v3_ln = reconstruct_posteriors(mod_v3_ln, chain_v3_ln, pts, area_idx, time_idx, W_sym; cov_indices=cov_indices_mat)
ppc_v3_ln = posterior_predictive_check(mod_v3_ln, stats_v3_ln, y_lognorm)

display(ppc_v3_ln.plot_density)
display(ppc_v3_ln.plot_scatter)
plt_prob = plot_posterior_results(stats_v3_ln, pts, centroids, W_sym; time_slice=5, effect=:st_predictions_noisy)
display(plt_prob)



# ----------
# binary model
mod_v4 = model_v4_binomial(
    y_binary, trials_sim, weights_sim,
    time_idx, area_idx, cov_indices,
    W_sym, class1_sim, class2_sim
)
chain_v4 = sample(mod_v4, MH(), 500)


stats_v4 = reconstruct_posteriors(mod_v4, chain_v4, pts, area_idx, time_idx, W_sym; cov_indices=cov_indices_mat)
ppc_v4 = posterior_predictive_check(mod_v4, stats_v4, y_binary)

display(ppc_v4.plot_density)
display(ppc_v4.plot_scatter)
plt_prob = plot_posterior_results(stats_v4, pts, centroids, W_sym; time_slice=5, effect=:st_predictions_noisy)
display(plt_prob)



# ------------
#  Deep GP (binomial)
mod_v5 = model_v5_deep_gp_carstm(
  y_binary, trials_sim, weights_sim, time_idx, area_idx, pts,
  cov_indices, W_sym, class1_sim, class2_sim
)
chain_v5 = sample(mod_v5, NUTS(), 100)


stats_v5 = reconstruct_posteriors(mod_v5, chain_v5, pts, area_idx, time_idx, W_sym; cov_indices=cov_indices_mat)
ppc_v5 = posterior_predictive_check(mod_v5, stats_v5, y_binary)

display(ppc_v5.plot_density)
display(ppc_v5.plot_scatter)
plt_prob = plot_posterior_results(stats_v5, pts, centroids, W_sym; time_slice=5, effect=:st_predictions_noisy)
display(plt_prob)


 




# ------------
# Generate 2 continuous covariates
N_obs_total = length(y_sim)
cont_covs = randn(N_obs_total, 2)

mod_v9 = model_v9_continuous_carstm(y_sim, time_idx, area_idx, cont_covs, W_sym; m_feat=10)
println("Sampling Model V9 (Continuous RFF)... ")
chain_v9 = sample(mod_v9, MH(), 500)

stats_v9 = reconstruct_posteriors(mod_v9, chain_v9, pts, area_idx, time_idx, W_sym)
ppc_v9 = posterior_predictive_check(mod_v9, stats_v9, y_sim)

display(ppc_v9.plot_density)
display(ppc_v9.plot_scatter)
plt_prob = plot_posterior_results(stats_v1, pts, centroids, W_sym; time_slice=5, effect=:st_predictions_noisy)
display(plt_prob)



# ------------
# Instantiate and run Model V6 test using the same simulation context
mod_v6 = model_v6_weighted_nb(
    y_counts,
    weights_sim,
    time_idx,
    area_idx,
    cov_indices_mat,
    W_sym,
    class1_sim,
    class2_sim
)
chain_v6 = sample(mod_v6, MH(), 500)

stats_v6 = reconstruct_posteriors(mod_v6, chain_v6, pts, area_idx, time_idx, W_sym; cov_indices=cov_indices_mat)
ppc_v6 = posterior_predictive_check(mod_v6, stats_v6, y_counts)
display(ppc_v6.plot_density)
display(ppc_v6.plot_scatter)
plt_prob = plot_posterior_results(stats_v6, pts, centroids, W_sym; time_slice=5, effect=:st_predictions_noisy)
display(plt_prob)

# ----------
# Poisson

mod_v7 = model_v7_weighted_pois(
    y_counts,
    weights_sim,
    time_idx,
    area_idx,
    cov_indices_mat,
    W_sym,
    class1_sim,
    class2_sim
)

println("Sampling Model V7 (Weighted ZIP)... ")
chain_v7 = sample(mod_v7, MH(), 500)

stats_v7 = reconstruct_posteriors(mod_v7, chain_v7, pts, area_idx, time_idx, W_sym; cov_indices=cov_indices_mat)
ppc_v7 = posterior_predictive_check(mod_v7, stats_v7, y_counts)

display(ppc_v7.plot_density)
display(ppc_v7.plot_scatter)

plt_prob_v7 = plot_posterior_results(stats_v7, pts, centroids, W_sym; time_slice=5, effect=:st_predictions_noisy)
display(plt_prob_v7)


# ----- 
# Variational Inference

n_samples_posterior = 1000 # Number of samples to draw from the approximated posterior
n_iters = 1000 # Number of iterations for VI optimization

results = []

# 1. Model V1: Basic CARSTM (GMRF)
println("Benchmarking Model V1...")
mod_v1 = model_v1_carstm_basic(y_sim, time_idx, area_idx, cov_indices, W_sym)
vi_v1, vi_v1_info, vi_v1_state = vi(mod_v1, q_meanfield_gaussian(mod_v1), n_iters; show_progress=true);
samples_v1 = rand(vi_v1, n_samples_posterior)
waic_v1 = calculate_waic(mod_v1, samples_v1)
push!(results, (model="V1_GMRF", waic=waic_v1))

# 2. Model V2: RFF CARSTM
println("Benchmarking Model V2...")
mod_v2 = model_v2_carstm_rff(y_sim, time_idx, area_idx, cov_indices, W_sym)
vi_v2, vi_v2_info, vi_v2_state = vi(mod_v2, q_meanfield_gaussian(mod_v2), n_iters; show_progress=true);
samples_v2 = rand(vi_v2, n_samples_posterior)
waic_v2 = calculate_waic(mod_v2, samples_v2)
push!(results, (model="V2_RFF", waic=waic_v2))

# 3. Model V3: Binomial
println("Benchmarking Model V3...")
mod_v3 = model_v3_binomial(y_binary, trials_sim, time_idx, area_idx, cov_indices, W_sym, class1_sim, class2_sim)
vi_v3, vi_v3_info, vi_v3_state = vi(mod_v3, q_meanfield_gaussian(mod_v3), n_iters; show_progress=true);
samples_v3 = rand(vi_v3, n_samples_posterior)
waic_v3 = calculate_waic(mod_v3, samples_v3)
push!(results, (model="V3_Binomial", waic=waic_v3))

# 4. Model V4: Weighted Binomial
println("Benchmarking Model V4...")
mod_v4 = model_v4_weighted_binary(
    y_binary, trials_sim, weights_sim,
    time_idx, area_idx, cov_indices,
    W_sym, class1_sim, class2_sim
)
vi_v4, vi_v4_info, vi_v4_state = vi(mod_v4, q_meanfield_gaussian(mod_v4), n_iters; show_progress=true);
samples_v4 = rand(vi_v4, n_samples_posterior)
waic_v4 = calculate_waic(mod_v4, samples_v4)
push!(results, (model="V4_Weighted", waic=waic_v4))

# 5. Model V5: Deep GP
println("Benchmarking Model V5...")
mod_v5 = model_v5_deep_gp_carstm(
  y_binary, trials_sim, weights_sim, time_idx, area_idx, pts,
  cov_indices, W_sym, class1_sim, class2_sim
)
vi_v5, vi_v5_info, vi_v5_state = vi(mod_v5, q_meanfield_gaussian(mod_v5), n_iters; show_progress=true);
samples_v5 = rand(vi_v5, n_samples_posterior)
waic_v5 = calculate_waic(mod_v5, samples_v5)

push!(results, (model="V5_DeepGP", waic=waic_v5))

# Display Results
df_results = DataFrame(results)
sort!(df_results, :waic)
println("\n--- Model Comparison (Sorted by WAIC) ---")
display(df_results)



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
