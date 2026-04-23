---
title: "CARSTM (Bayesian Conditional AutoRegressive Spatiotemporal) in Julia/Turing"
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
 

CARSTM (Bayesian Conditional AutoRegressive SpaceTime Models in Julia) is a Julia project that combines elements of the Spatial partionning methods together with some basic didactic spatiotemporal models.

It replicates most of the functionality of:

- the core functionality of the R-packages [aegis](https://github.com/jae0/aegis) and [aegis.polygons](https://github.com/jae0/aegis.polygons) for creating and handling areal unit information, and

- the core functionality of R-package CARSTM (https://github.com/jae0/carstm) which was an INLA wrapper for simple GRMF spatiotemporal models. 

This Julia approach, by leveraging the power and flexibility of the language (especially the Bayesian Turing.jl framework), far exceeds their abilities with a compact set of functions.  

Navigating Space and Time: A Conceptual Primer on the CARSTM Framework

## Introduction: The Spatio-Temporal Challenge in the Maritimes

In the rugged, high-stakes ecosystems of the Maritimes Region of Canada, ecological monitoring is a pursuit of moving targets. To accurately model variables like bottom temperature, species composition, and the population dynamics of the snow crab—often utilizing a Hurdle model (combining Binomial and Poisson components)—we require a framework that can handle extreme dimensionality and dependency. This is where the CARSTM (Conditional Autoregressive Spacetime Model) becomes indispensable.

CARSTM is a high-dimensional Bayesian hierarchical framework designed to decompose complex spatio-temporal data into interpretable latent components. Standard models typically fail in the Maritimes because they assume independence; however, ecological data is inherently dependent. To address the "Spatio-Temporal Challenge," CARSTM utilizes three primary components:

1. Spatial Clustering: Implemented via the BYM2 (Besag-York-Mollié) specification to account for geographical neighborhoods.
2. Temporal Autocorrelation: Utilizing AR1 or Random Fourier Features (RFF) to capture evolving trends.
3. Non-linear Interactions: Modeling Type IV Interactions where the relationship between space and time is non-stationary and dynamic.

Teacher’s Note: The "So What?" Why must we separate these latent fields? In ecological precision modeling, failing to distinguish between a permanent habitat feature (captured by the BYM2 spatial component) and a transient environmental anomaly (captured by the Type IV interaction) results in biased forecasts. By isolating these effects, we ensure that our management decisions for species like snow crab are based on the true underlying drivers rather than statistical noise.

This decomposition is made possible by the rigorous mathematical axioms that transform a computationally prohibitive problem into a tractable one.


--------------------------------------------------------------------------------


## The Core Axioms: Rules of the CARSTM Universe

For the CARSTM framework to produce valid inference, it operates under four core axioms. These rules allow us to move from O(N^3) Gaussian Process complexity to O(N) or O(N \log N) tractability.

Core Axiom	The Concept (Simplified)	The Benefit to the Learner
Markov Property	A spatial unit is independent of all non-neighbors given its immediate neighbors (\mathcal{N}(i)).	Creates Sparse Precision Matrices (Q); makes high-dimensional GMRF problems computationally solvable.
Additivity	The predictor \eta is a sum of separable parts: \alpha + \text{Space} + \text{Time} + \text{Interaction} + \text{Covariates}.	Allows independent study of geographic and temporal drivers while still permitting Type IV interactions.
Stationarity	Processes assume constant mean/variance over a standardized [0, 1] interval.	Provides structural stability; ensures the "rules" of time-series (AR1) or kernels (RFF) are consistent.
Rank-Deficiency	Intrinsic priors (ICAR and RW2) measure differences between units, not absolute levels.	Provides the mathematical basis for smoothing, though it requires constraints to achieve identifiability.

While these axioms provide the structure, the inherent rank-deficiency of these priors requires specific constraints to produce unique results.


--------------------------------------------------------------------------------


##  Solving the Identifiability Puzzle: The Sum-to-Zero Constraint

When we employ intrinsic priors like the ICAR (spatial) or RW2 (temporal), we encounter a singular precision matrix. Because these priors define the distribution of differences between points, they possess a "null space." Mathematically, adding any constant c to the vector \mathbf{u} (i.e., \mathbf{u} + c\mathbf{1}) results in the same prior log-density.

Without intervention, the model suffers from the Rank-Deficiency Problem: it cannot distinguish between the global intercept (\alpha) and the mean level of the spatial field. This leads to an improper posterior where the intercept may drift toward infinity while the spatial field drifts toward negative infinity.

The Three Impacts of the Sum-to-Zero Constraint (\sum u_i = 0):

1. Unique Estimates: By "pinning" the latent field to a mean of zero, the global intercept is preserved as the true overall mean of the response.
2. Geographic Interpretation: The spatial vector effectively captures only the deviations from the mean, highlighting which areas are geographically anomalous.
3. Sampler Stability: This prevents MCMC chains from wandering an infinite "ridge" of equally likely values, ensuring convergence.

Implementation Note: While "Soft Constraints" (penalty methods) exist, Explicit Re-centering (subtracting the empirical mean during each iteration) is the preferred method for maintaining stability within the NUTS sampler.


--------------------------------------------------------------------------------


##  Partitioning the Map: Areal Units and Information Balance

To run the model, we must discretize the spatial domain into "Areal Units." A well-constructed partition balances geometric compactness with statistical information density to avoid "Data Starvation."

* Centroidal Voronoi Tessellation (CVT): The "gold standard" for uniform statistical power. It uses Lloyd's algorithm to create a highly regular, "honeycomb" mesh.
* Binary Vector Tree (BVT): A recursive splitting method along the axis of maximum variance. It is the fastest approach, ideal for datasets with millions of points.
* Quadrant Voronoi Tessellation (QVT): A quadtree-like decomposition that excels at capturing multi-scale spatial clusters and density transitions.
* Agglomerative Voronoi Tessellation (AVT): An iterative merging approach that provides a hard guarantee on a minimum point threshold (min_pts). This is the most robust method for small sample sizes.

The Poisson Weighted (Adaptive) CVT In the Maritimes, data density is rarely uniform. Adaptive CVT uses Kernel Density Estimation (KDE) to migrate seeds toward density modes (peaks). This shrinks tiles in high-activity areas and stretches them in sparse areas, ensuring every unit is informative and minimizing Boundary Artifacts that occur when standard tiles split high-density clusters.


--------------------------------------------------------------------------------


##  Advanced Mechanics: RFF, Deep GPs, and Scaling

To handle non-stationary surfaces and large-scale seasonality, CARSTM employs sophisticated approximations and scaling techniques.

Random Fourier Features (RFF) Using Bochner’s Theorem, we approximate a stationary kernel k(\mathbf{x}, \mathbf{x}') by sampling from a non-negative spectral density. By transforming the problem into a linear Bayesian regression, the inner product \phi(\mathbf{x})^T \phi(\mathbf{x}') converges to the kernel k as m \to \infty, offering O(nm^2) efficiency compared to O(n^3) for traditional Gaussian Processes.

The Role of Sørbye & Rue Scaling and PC Priors As Senior Ecologists, we must ensure our priors are interpretable. We use Sørbye & Rue scaling to ensure the structured spatial component has a unit marginal variance. This allows the \phi parameter in a BYM2 model to represent the actual proportion of variance explained by space. Furthermore, we utilize Penalized Complexity (PC) Priors to provide principled shrinkage, pulling the model toward a simpler "base" state (zero variance) unless the data provides strong evidence otherwise.

Expanded Model Taxonomy | Model | Likelihood Family | Key Feature | Best Use Case | | :--- | :--- | :--- | :--- | | v1 | Gaussian | AR1 + BYM2 | Standard continuous data (e.g., temperature). | | v2 | Gaussian | RFF + BYM2 | Continuous data with multi-scale seasonality. | | v3 | LogNormal | AR1 + BYM2 | Right-skewed positive data (e.g., biomass). | | v4 | Binomial | AR1 + BYM2 | Proportions or prevalence data. | | v5 | Poisson | AR1 + BYM2 | Count data (optional Zero-Inflation). | | v6 | Neg-Binomial | AR1 + BYM2 | Over-dispersed count data (Variance > Mean). | | v7 | Binomial | Deep GP (RFF) | Non-stationary proportions. | | v8 | Gaussian | Deep GP (RFF) | Non-stationary continuous phenomena. | | v9 | Gaussian | Continuous RFF | Non-linear effects for continuous covariates. | | v10| Gaussian | 3-Layer Deep GP | Maximum flexibility for extreme non-stationarity. |


--------------------------------------------------------------------------------


##  The Mixed-Sampler Strategy: How the Model Learns

Inference in CARSTM requires a Mixed-Sampler Gibbs approach, as no single algorithm is optimal for the entire parameter space.

* ESS (Elliptical Slice Sampling):
  * Application: Latent Gaussian Fields (u_icar, u_iid, st_int_raw).
  * Justification: Analytically exact for Gaussian priors; requires no tuning and is extremely stable in high dimensions.
* NUTS (No-U-Turn Sampler):
  * Application: Fixed Effects / Regression Weights.
  * Justification: Navigates complex posterior geometries by finding optimal path lengths, avoiding the inefficiencies of a random walk.
* Particle Gibbs (PG):
  * Application: Zero-Inflation indicators (\phi_{zi}) and discrete latent states.
  * Justification: Uses a sequential Monte Carlo particle filter to navigate non-differentiable or jagged posterior paths.

For production-grade point estimates, we may utilize ADVI (Automatic Differentiation Variational Inference). In these cases, increasing the n_samples for the ELBO gradient estimation is critical to stabilize convergence against the noise of complex spatial interactions.


--------------------------------------------------------------------------------


## References

* Besag, J. (1974): Spatial interaction and the statistical analysis of lattice systems. Journal of the Royal Statistical Society.
* Rue, H., & Held, L. (2005): Gaussian Markov Random Fields: Theory and Applications. CRC Press.
* Sørbye, S. H., & Rue, H. (2014): Scaling intrinsic Gaussian Markov random field priors in spatial statistics.
* Rahimi, A., & Recht, B. (2007): Random features for large-scale kernel machines. NIPS.
* Hoffman, M. D., & Gelman, A. (2014): The No-U-Turn Sampler. Journal of Machine Learning Research.



## Computation examples

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
    local areal_units
    areal_units = assign_spatial_units(pts, m;
        max_total_arealunits=max_u,
        buffer_dist=0.8,
        min_pts=15)

    push!(results_expanded, (method=m, requested_max=max_u, actual_units=length(areal_units.centroids)))

    p = plot_spatial_graph(pts, areal_units; title="Method: $m", domain_boundary=areal_units.hull_coords)
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
areal_units = assign_spatial_units(pts, area_method;
        max_total=max_total_units,
        buffer_dist=0.8,  # fraction of mean distances
        min_pts=min_pts)

actual_units=length(areal_units.centroids)

plt = plot_spatial_graph(pts, areal_units; title="Method: $area_method", domain_boundary=areal_units.hull_coords)

display(plt)
   

# classify locations and adjacency
g_v1 = get_spatial_graph(areal_units)
W_sym = Float64.( Graphs.adjacency_matrix(g_v1) )

area_idx = areal_units.assignments ; 

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
modinputs = prepare_model_inputs(y_sim, pts, area_idx, time_idx, W_sym, n_categories)
cont_covs_dummy = randn(length(y_sim), 2)

# Ensure count data is strictly non-negative integers for count models
y_counts = abs.(Int.(round.(modinputs.y)))
precomp_counts = merge(modinputs, (y = y_counts,))
using BenchmarkTools

# Configuration
n_bench_iters = 50
bench_results = Dict{String, Float64}()

# Ensure count data is strictly non-negative integers for count models
y_counts = abs.(Int.(round.(modinputs.y)))
precomp_counts = merge(modinputs, (y = y_counts,))

println("Starting Suite Benchmark (MH, $n_bench_iters iterations)...\n")

# Define the full model set with corrected data inputs for count families
models_to_bench = Dict(
    "v1_gaussian"         => () -> model_v1_gaussian(modinputs),
    "v2_rff_gaussian"     => () -> model_v2_rff_gaussian(modinputs),
    "v3_lognormal"        => () -> model_v3_lognormal(modinputs),
    "v4_binomial"         => () -> model_v4_binomial(modinputs; trials=ones(Int, length(y_sim))),
    "v5_poisson"          => () -> model_v5_poisson(precomp_counts),
    "v6_negativebinomial" => () -> model_v6_negativebinomial(precomp_counts),
    "v7_deep_gp_binomial" => () -> model_v7_deep_gp_binomial(modinputs; trials=ones(Int, length(y_sim))),
    "v8_deep_gp_gaussian" => () -> model_v8_deep_gp_gaussian(modinputs),
    "v9_continuous_gaussian" => () -> model_v9_continuous_gaussian(cont_covs_dummy, modinputs),
    "v10_deep_gp_3layer"  => () -> model_v10_deep_gp_3layer_gaussian(modinputs)
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
mod_v2 = model_v2_rff_gaussian(modinputs)

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
m_v1 = model_v1_gaussian(modinputs)

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
