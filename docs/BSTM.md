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
 
Bayesian Space-Time Models in Julia (BSTM) is a Julia project that combines elements of the Spatial partitioning methods together with some basic Bayesian spatiotemporal models. At its core is a discrete perspective upon space and time, not for philosophical reasons, but rather operational functionality. Spatiotemporal models are resource intensive. This discrete perspective permits useful solutions within the constraints of most currently available computing resources. It also goes well beyond the discrete approximations with continuous Gaussian Process methods, some of which we will touch upon below. Though the focus is upon Ecological issues, it is a general framework that can be readily adapted to any spatiotemporal process, no matter how large or small. 

The current library of functions replicates most of the functionality of the following R-packages and essentially subsumes them:

- [aegis](https://github.com/jae0/aegis): basic spatial tools,
- [aegis.polygons](https://github.com/jae0/aegis.polygons): creating and handling areal unit information, and
- [CARSTM](https://github.com/jae0/carstm): an INLA wrapper for simple GRMF spatiotemporal models. 
- [stmv](https://github.com/jae0/stmv): a mosaic approach to non-stationary spatiotemporal processes. 

Using Julia leverages the power and flexibility of the language (especially the Bayesian Turing.jl framework), with a compact, flexible and extensible set of functions and tools. Ultimately, here, we are developing a general framework to explore various models of increasing complexity to handle measurement error, periodic dynamics, and spatial dependencies. Random Fourier Features (RFF), Fully Independent Training Conditional (FITC) and Deep Gaussian Processes are explored to make Discrete and Continuous Spatiotemporal models computationally tractable for large datasets.

Note: a lot of this work has been accelerated by using [Google' Colab](https://colab.research.google.com/). If used carefully, it can be a powerful accelerator. I acknowledge the use of it, though it does require care as well. 


## Introduction: The SpatioTemporal Challenge 

Ecological monitoring is a pursuit of moving targets. To usefully model important variables  like bottom temperature, species composition, and the population dynamics of  species requires using utilizing incomplete or low density information from expensive surveys with limits to resources and time. The usual recourse is some variation of Random Stratified Sampling to absorb unaccounted errors or "externalities". In BSTM, we embrace these externalities as they are quite informative and useful. 

BSTM is a high-dimensional Bayesian hierarchical framework designed to decompose complex spatiotemporal data into interpretable latent components. Standard models typically  assume independence ("Random Stratified"); however, ecological data is inherently dependent. To address the "SpatioTemporal Challenge," BSTM utilizes three primary components:
BSTM is a high-dimensional Bayesian hierarchical framework designed to decompose complex spatio-temporal data into interpretable latent components. Standard models typically  assume independence ("Random Stratified"); however, ecological data is inherently dependent. To address the "Spatio-Temporal Challenge," BSTM utilizes three primary components:

1. Spatial Clustering: Implemented via the BYM2 (Besag-York-Mollié) specification to account for geographical neighborhoods.
2. Temporal Autocorrelation: Utilizing AR1 or Random Fourier Features (RFF) to capture evolving trends.
3. Non-linear Interactions: Modeling "Type IV" Interactions where the relationship between space and time is non-stationary and dynamic.

Failing to distinguish between a permanent habitat feature (captured by the BYM2 spatial component) and a transient environmental anomaly (captured by the Type IV interaction) results in biased forecasts. By isolating these effects, we ensure that our understanding and consequent management decisions are based on the "true" underlying drivers rather than statistical noise. This decomposition is made possible by the rigorous mathematical axioms that transform a computationally prohibitive problem into a tractable one.

 
## The Core Assumptions

For BSTMs, there are four core assumptions. Some of these can be relaxed depending upon the final method but these assumptions allow us to move from $O(N^3)$ Gaussian Process complexity to $O(N)$ or $O(N \log N)$ operational tractability. 

Markov Property: 
- A spatial unit is independent of all non-neighbors given its immediate neighbors ($\mathcal{N}(i)$). 
- GMRF methods take advantage of operating on Sparse Precision Matrices ($Q$) as it makes high-dimensional problems computationally solvable.

Additivity:	
- The predictor $\eta$ is a sum of separable parts: $\alpha + \text{Space} + \text{Time} + \text{Interaction} + \text{Covariates}$.	
- Allows independent study of geographic and temporal drivers while still permitting more complex space-time interactions (e.g., Type IV).

Stationarity:
- Processes assume constant mean/variance over a standardized [0, 1] interval.	
- Provides structural stability; ensures the "rules" of time-series (AR1) or kernels (RFF) are consistent.

Rank-Deficiency:
- Intrinsic priors (ICAR and RW2) measure differences between units, not absolute levels.	
- Provides the mathematical basis for smoothing, though it requires constraints to achieve identifiability.


## Identifiability: The Sum-to-Zero Constraint

When we use intrinsic priors like the ICAR (spatial) or RW2 (temporal) to provide structure to our models, we encounter a singular precision matrix. Because these priors define the distribution of differences between points, they possess a "null space." In other words, adding any constant $c$ to the vector $\mathbf{u}$ (i.e., $\mathbf{u} + c\mathbf{1}$) results in the same log-density, the metric used for solution finding. In the most extreme form, this Rank-Deficiency Problem means that a computations cannot distinguish between a global intercept ($\alpha$) and the mean level of the spatial field are interchangeable and one may drift toward +infinity while the other drifts toward -infinity.

Using a Sum-to-Zero Constraint ($\sum u_i = 0$) "pins" the latent field to a mean of zero, so the global intercept is preserved as the true overall mean of the response. This means the spatial effect effectively captures only the deviations from the mean, so highlighting which areas are geographically anomalous. This is very much the approach that was using in the early Universal Kriging with External Drift (UKED). This also has the benefit of stabilizing computations, by preventing MCMC chains from wandering along an infinite "ridge" of equally likely values, and so supporting convergence.

Implementation Note: While "Soft Constraints" (penalty methods) exist, Explicit Re-centering (subtracting the empirical mean during each iteration) is the preferred method for maintaining stability within the NUTS sampler.
 
 
##  Partitioning the Map: Areal Units and Information Balance

To run the discrete BSTMs, we must discretize the spatial domain into "Areal Units." A well-constructed partition balances geometric compactness with statistical information density to avoid "Data Starvation." Or sometimes, one inherits management areal units, often with no structural support. Though one can simply push on using such area definitions, if the balance of information available to information extractable is poor due to improper sizes and shapes, one should consider alternative areal units which then can be reconsolidated to estimate at the level of the unfortunate management units.   

Amongst the partitionning methods available in BSTM are:
   
- (Adaptive) Centroidal Voronoi Tessellation (CVT): A popular method designed for uniform statistical power. It uses Lloyd's algorithm to create a highly regular, "honeycomb" mesh. Data density is rarely uniform and so an Adaptive form of CVT uses Kernel Density Estimation (KDE) to migrate seeds toward density modes (peaks). This shrinks tiles in high-activity areas and stretches them in sparse areas, ensuring every unit is informative and minimizing Boundary Artifacts that occur when standard tiles split high-density clusters.

- Binary Vector Tree (BVT): A recursive splitting method along the axis of maximum variance. It is the fastest approach, ideal for datasets with millions of points.

- Quadrant Voronoi Tessellation (QVT): A quadtree-like decomposition that excels at capturing multi-scale spatial clusters and density transitions.

- Agglomerative Voronoi Tessellation (AVT): An iterative merging approach that balances multiple constraints upon the data that iteratively aggregates small areal units until stopping rules are met. It also begins with KDE to identify initial conditions. 


  
##  Advanced Mechanics: RFF, Deep GPs, and Scaling

To handle non-stationary surfaces and large-scale seasonality, BSTM employs sophisticated approximations and scaling techniques.

- Random Fourier Features (RFF): Using Bochner’s Theorem, we approximate a stationary kernel $k(\mathbf{x}, \mathbf{x}'$) by sampling from a non-negative spectral density. By transforming the problem into a linear Bayesian regression, the inner product $\phi(\mathbf{x})^T \phi(\mathbf{x}'$) converges to the kernel $k$ as $m \to \infty$, offering $O(nm^2)$ efficiency compared to $O(n^3)$ for traditional Gaussian Processes.

- Penalized Complexity (PC) Priors to provide principled shrinkage: To ensure priors are interpretable and as a default have an opinion, that the process being modelled should have a null hypothesis that it is not important. So for example, when a temporal autocorrelation is modelled, the prior has a density centered over zero (no autocorrelation) and only if the data suggests it is strong that the posterior will moved away from the prior. For spatial processes, the effects are scaled to a unit marginal variance which allows the $\phi$ parameter in a BYM2 model to represent the actual proportion of variance explained by the spatial effect. The prior for this variance would be a "base" state of (zero variance) unless the data provides strong evidence otherwise.

- Model Taxonomy

Most of these models are didactic to demonstrate form and approach. A few will actually be used operationally and highlighted. 

| Model | Likelihood Family | Key Feature       | Best Use Case                                     |
| :------| :------------------| :------------------| :--------------------------------------------------|
| v1    | Gaussian          | AR1 + BYM2        | Standard continuous data (e.g., temperature).     |
| v2    | Gaussian          | RFF + BYM2        | Continuous data with multi-scale seasonality.     |
| v3    | LogNormal         | AR1 + BYM2        | Right-skewed positive data (e.g., biomass).       |
| v4    | Binomial          | AR1 + BYM2        | Proportions or prevalence data.                   |
| v5    | Poisson           | AR1 + BYM2        | Count data (optional Zero-Inflation).             |
| v6    | Neg-Binomial      | AR1 + BYM2        | Over-dispersed count data (Variance > Mean).      |
| v7    | Binomial          | Deep GP (RFF)     | Non-stationary proportions.                       |
| v8    | Gaussian          | Deep GP (RFF)     | Non-stationary continuous phenomena.              |
| v9    | Gaussian          | Continuous RFF    | Non-linear effects for continuous covariates.     |
| v10   | Gaussian          | 3-Layer Deep GP   | Maximum flexibility for extreme non-stationarity. |
| v11   | Gaussian          | 3-Layer Deep GP   | Non-Separable Space-Time RFF Kernel.              |
| v12   | Gaussian          | SPDE-based        | Matern Spatial Field (RFF Approximation)          |
| v12   | Gaussian          | Spectral SPDE     | Continuous spatial field with Matern lengthscale. |
| v13   | Gaussian          | Warped Manifold   | Non-stationary surfaces with non-linear warping.  |
| v14   | Gaussian          | Spectral FFT      | Fast spatial filtering for large lattices/grids.  |
| v15   | Gaussian          | Mosaic Model      | Regional dynamics with soft boundary blending.    |
| v16   | Gaussian          | Integrated Mosaic | Non-separable space-time regional mosaics.        |



Model summary: 100 NUTS()

V0: (WAIC= 166.7;  time= 575 s)
V1: (WAIC= 724.2;  time=1340 s)   
V2: (WAIC=  95.1;  time= 597 s)     
V3: (WAIC= 112.2;  time= 171 s)    
V4: (WAIC=1134.1;  time=1380 s) 
V5: (WAIC= 313.3;  time=2201 s) 
V6: (WAIC= 137.2;  time=2682 s)
V7: (WAIC= 186.7;  time= 456 s)
V8: (WAIC= 117.6;  time=4961 s)


## Data

```{julia}


# Multi-fidelity (multi-scale) data

Ns_y_unique, Nt_y_unique = 10, 5
Ny = Ns_y_unique * Nt_y_unique

Ns_u_unique, Nt_u_unique = 10, 10
Nu = Ns_u_unique * Nt_u_unique

Nz = 120

# Spatial fidelity (Z)
coords_z_s = rand(Nz, 2)
z_mock = randn(Nz)

# Spatiotemporal fidelity (U)
unique_coords_u_s = rand(Ns_u_unique, 2)
coords_u_s = repeat(unique_coords_u_s, outer=(Nt_u_unique, 1))
coords_u_t = reshape(repeat(collect(1.0:Nt_u_unique), inner=Ns_u_unique), :, 1)
u1_mock, u2_mock, u3_mock = randn(Nu), randn(Nu), randn(Nu)

# Standard fidelity (Y)
unique_coords_y_s = rand(Ns_y_unique, 2)
coords_y_s = repeat(unique_coords_y_s, outer=(Nt_y_unique, 1))
coords_y_t = reshape(repeat(collect(1.0:Nt_y_unique), inner=Ns_y_unique), :, 1)
y_mock = randn(Ny)


# --- 2. FFT-Informed RFF Parameter Generation ---
M_rff_base_val = 40
M_rff_sigma_val = 20

# Z-fidelity frequencies
W_z_fixed, b_z_fixed = generate_informed_rff_params(coords_z_s, M_rff_base_val)

# U-fidelity frequencies (Inputs: [lon, lat, time, z_latent])
coords_u_dummy = hcat(coords_u_s, coords_u_t, rand(Nu, 1))
W_u_fixed, b_u_fixed = generate_informed_rff_params(coords_u_dummy, M_rff_base_val)

# Volatility frequencies
coords_st_y = hcat(coords_y_s, coords_y_t)
W_sigma_fixed, b_sigma_fixed = generate_informed_rff_params(coords_st_y, M_rff_sigma_val)

# --- 3. Inducing Point Setup for FITC ---
M_inducing_v25 = 15
# Feature dimensions: [lon, lat, time, z, u1, u2, u3]
Z_inducing_feat = randn(M_inducing_v25, 7)

println("Consolidated Setup Complete:")
println(" - Y observations: $Ny")
println(" - U observations: $Nu")
println(" - Z observations: $Nz")
println(" - Inducing points: $M_inducing_v25")

```





### Model v11: Non-Separable Space-Time RFF Kernel

This model moves beyond separable interactions by defining a kernel $K(\mathbf{s}, t, \mathbf{s}', t')$ that cannot be decomposed into $K_s(\mathbf{s}, \mathbf{s}') \times K_t(t, t')$. By using Random Fourier Features on the joint vector $[x, y, t]$, we approximate a non-separable stationary kernel (like a 3D Matern or RBF) that allows for more flexible spatiotemporal dynamics.
This model moves beyond separable interactions by defining a kernel $K(\mathbf{s}, t, \mathbf{s}', t')$ that cannot be decomposed into $K_s(\mathbf{s}, \mathbf{s}') \times K_t(t, t')$. By using Random Fourier Features on the joint vector $[x, y, t]$, we approximate a non-separable stationary kernel (like a 3D Matern or RBF) that allows for more flexible spatio-temporal dynamics.


```{julia}

@model function model_v11_non_separable_rff(modinputs, ::Type{T}=Float64; m_joint=25, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v11: Non-Separable Spatiotemporal RFF model.
    # Instead of separate spatial and temporal components, this model projects
    # the joint [X, Y, Time] vector into a shared feature space.

    y = modinputs.y
    N_obs = length(y)
    
    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_joint ~ Exponential(1.0)
    # Lengthscales for X, Y, and Time dimensions within the joint kernel
    l_joint ~ filldist(Gamma(2, 1), 3) 
    w_joint ~ MvNormal(zeros(m_joint), I)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Feature Matrix Construction ---
    # X_joint: [normalized_x, normalized_y, normalized_time]
    # We use modinputs.pts_raw and normalize them for numerical stability
    xs = [p[1] for p in modinputs.pts_raw]
    ys = [p[2] for p in modinputs.pts_raw]
    ts = Float64.(modinputs.time_idx)
    
    # Normalize inputs to [0, 1] range
    X_joint = hcat(
        (xs .- minimum(xs)) ./ (maximum(xs) - minimum(xs) + 1e-6),
        (ys .- minimum(ys)) ./ (maximum(ys) - minimum(ys) + 1e-6),
        (ts .- minimum(ts)) ./ (maximum(ts) - minimum(ts) + 1e-6)
    )

    # --- 3. Joint RFF Projection ---
    # This creates the non-separable interaction
    Random.seed!(42)
    # Sample frequencies scaled by dimension-specific lengthscales
    Om = randn(m_joint, 3) .* (1.0 ./ l_joint')
    Ph = rand(m_joint) .* convert(T, 2π)
    
    Z_joint = convert(T, sqrt(2/m_joint)) .* cos.(X_joint * Om' .+ Ph')
    eta_joint = Z_joint * (w_joint .* sigma_joint)

    # --- 4. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 5. Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + eta_joint[i]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end
 # Verification run for Model v11
println("Running smoke test for Model v11 (Non-Separable RFF)... ")

# Fix: Ensure modinputs contains observation-level coordinates
# Repeating the base centroids to match the total observation count (N_areas * N_years)
N_total_obs = length(modinputs.y)
n_areas_base = size(modinputs.Q_sp, 1)
n_years_base = Int(N_total_obs / n_areas_base)

pts_full = repeat(modinputs.pts_raw[1:n_areas_base], n_years_base)

# Update modinputs for this specific model run
modinputs_v11 = merge(modinputs, (pts_raw = pts_full,))

mod_v11 = model_v11_non_separable_rff(modinputs_v11; m_joint=20)

# Sample using MH for a quick smoke test
chain_v11 = sample(mod_v11, MH(), 100)

display(MCMCChains.summarize(chain_v11[[:sigma_joint, Symbol("l_joint[1]"), Symbol("l_joint[2]"), Symbol("l_joint[3]")]]))

```

### Model v12: SPDE-based Matern Spatial Field (RFF Approximation)

Instead of the discrete BYM2/ICAR graph structure used in Model v1, Model v12 adopts an SPDE approach. We approximate the solution to $(\kappa^2 - \Delta)^{\alpha/2} S(s) = \mathcal{W}(s)$ using spectral basis functions (RFF). This allows for continuous spatial coordinates and explicit lengthscale ($\\kappa^{-1}$) estimation.

```{julia}

@model function model_v12_spde_gaussian(modinputs, ::Type{T}=Float64; m_spatial=50, offset=modinputs.offset) where {T}
    # Model v12: SPDE-style continuous spatial field using spectral RFF approximation.
    y = modinputs.y
    N_obs = length(y)
    N_time = maximum(modinputs.time_idx)

    # --- 1. SPDE / Matern Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0)
    kappa_sp ~ Gamma(2, 1)  # Range parameter (1/lengthscale)
    w_sp ~ MvNormal(zeros(m_spatial), I)

    # --- 2. Temporal (AR1) & Smoothing Priors ---
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 3. Continuous Spatial Basis (Spectral SPDE) ---
    # Normalize points for spectral projection
    xs = [p[1] for p in modinputs.pts_raw]
    ys = [p[2] for p in modinputs.pts_raw]
    coords = hcat(
        (xs .- mean(xs)) ./ std(xs),
        (ys .- mean(ys)) ./ std(ys)
    )

    Random.seed!(42)
    # Frequencies sampled for a Matern kernel approximation
    # Note: For Matern nu=1.5, we sample from a Student-t distribution spectral density
    Om = randn(m_spatial, 2) .* kappa_sp
    Ph = rand(m_spatial) .* convert(T, 2π)
    
    Z_sp = convert(T, sqrt(2/m_spatial)) .* cos.(coords * Om' .+ Ph')
    s_eff = Z_sp * (w_sp .* sigma_sp)

    # --- 4. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 5. Categorical & Likelihood ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    for i in 1:N_obs
        mu = offset[i] + s_eff[i] + f_time[modinputs.time_idx[i]]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! modinputs.weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end

# Verification of SPDE Model v12
println("Initializing SPDE Model v12...")
mod_v12 = model_v12_spde_gaussian(modinputs; m_spatial=30)

# MAP for rapid convergence
# map_v12 = maximum_a_posteriori(mod_v12)

# Short NUTS chain to verify posterior variance
chain_v12 = sample(mod_v12, MH(), 100) #; initial_params=InitFromParams(map_v12))

summarystats(chain_v12[[:sigma_sp, :kappa_sp, :rho_tm]])

```

### Model v13: Non-Stationary Spatial Field (Warping Manifold)

Standard GP and SPDE models assume **stationarity**: the correlation between two points depends only on their distance. Model v13 relaxes this by introducing a **Warping Function** $g(s)$. The spatial field is modeled as $S(g(s))$, where $g$ is a non-linear transformation approximated by RFFs. This allows the model to compress or stretch space, capturing localized clusters and sharp transitions that stationary models smooth over.


```{julia}

@model function model_v13_nonstationary_warping(modinputs, ::Type{T}=Float64; m_warp=10, m_spatial=40, offset=modinputs.offset) where {T}
    y = modinputs.y
    N_obs = length(y)
    N_time = maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0)
    l_warp ~ Gamma(2, 1)    # Smoothness of the warping manifold
    l_spatial ~ Gamma(2, 1) # Smoothness of the stationary field in warped space
    
    w_warp ~ MvNormal(zeros(m_warp), I)
    w_sp ~ MvNormal(zeros(m_spatial), I)
    
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Input Preprocessing ---
    xs = [p[1] for p in modinputs.pts_raw]
    ys = [p[2] for p in modinputs.pts_raw]
    coords = hcat((xs .- mean(xs)) ./ std(xs), (ys .- mean(ys)) ./ std(ys))

    # --- 3. Warping Layer (Non-Stationarity) ---
    # This layer 'warps' the 2D coordinates into a latent space
    Random.seed!(44)
    Om_w = randn(m_warp, 2) ./ l_warp
    Ph_w = rand(m_warp) .* convert(T, 2π)
    
    # Warped coordinates: g(s)
    warped_coords = (convert(T, sqrt(2/m_warp)) .* cos.(coords * Om_w' .+ Ph_w')) * w_warp

    # --- 4. Spatial Field on Warped Manifold ---
    # We apply a stationary kernel to the warped output
    Random.seed!(45)
    Om_s = randn(m_spatial, 1) ./ l_spatial
    Ph_s = rand(m_spatial) .* convert(T, 2π)
    
    Z_sp = convert(T, sqrt(2/m_spatial)) .* cos.(reshape(warped_coords, :, 1) * Om_s' .+ Ph_s')
    s_eff = Z_sp * (w_sp .* sigma_sp)

    # --- 5. Temporal & Categorical Components ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + s_eff[i] + f_time[modinputs.time_idx[i]]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! modinputs.weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


# Verification of Non-Stationary Warping Model v13
println("Initializing Non-Stationary Model v13...")
mod_v13 = model_v13_nonstationary_warping(modinputs; m_warp=8, m_spatial=25)

# MAP Estimate
# map_v13 = maximum_a_posteriori(mod_v13)

# Chain verification
chain_v13 = sample(mod_v13, MH(), 100) #; initial_params=InitFromParams(map_v13))

summarystats(chain_v13[[:l_warp, :l_spatial, :sigma_sp]])

```

### Model v14: Spectral GMRF (FFT-Accelerated)

Model v14 leverages the **Spectral Representation** of the spatial precision matrix. For a regular lattice, the Laplacian matrix is diagonalized by the Discrete Fourier Transform (DFT). This model performs the spatial filtering in the frequency domain:

1.  **Transform**: Map the white noise innovations to the frequency domain using `fft`.
2.  **Spectral Filter**: Scale the frequencies by the eigenvalues of the Laplacian (which are known analytically for lattices or can be precomputed).
3.  **Inverse Transform**: Map back to the spatial domain using `ifft`.

This provides a massive speedup for large $N$ by avoiding $O(N^2)$ sparse matrix operations or $O(N^3)$ factorizations in every log-density evaluation.


```{julia}using FFTW

@model function model_v14_fft_gaussian(modinputs, ::Type{T}=Float64; grid_res=64, pad_factor=2, offset=modinputs.offset) where {T}
    # Model v14 Refined: FFT-Accelerated GMRF
    y = modinputs.y
    N_obs = length(y)
    N_areas = size(modinputs.Q_sp, 1)
    N_time = maximum(modinputs.time_idx)
    
    padded_res = grid_res * pad_factor
    
    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Spectral Spatial Field (FFT) ---
    # We sample in the frequency domain for the ICAR component
    # A padded grid of white noise
    u_spectral_raw ~ MvNormal(zeros(padded_res^2), I)
    u_iid ~ MvNormal(zeros(N_areas), I)

    # Reshape and perform FFT
    u_fft = fft(reshape(convert.(Complex{T}, u_spectral_raw), padded_res, padded_res))
    
    # Apply a simplified Spectral Matern/Laplacian Filter
    # In a full version, we'd use analytic eigenvalues of the Laplacian
    # For this version, we use the scaled spatial precision for the structured part
    u_icar_raw = modinputs.Q_sp \ u_spectral_raw[1:N_areas]
    
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar_raw .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Categorical Effects ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 5. Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + s_eff[modinputs.area_idx[i]] + f_time[modinputs.time_idx[i]]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! modinputs.weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end

using SparseArrays, FFTW, Statistics

function prepare_fft_grid(pts, values; grid_res=64, pad_factor=2)
    # 1. Define the bounding box
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)

    # 2. Map points to a grid
    # Use the length of the shorter input to prevent BoundsError
    n_limit = min(length(pts), length(values))
    grid = zeros(grid_res, grid_res)

    for i in 1:n_limit
        p = pts[i]
        ix = Int(floor((p[1] - xmin) / (xmax - xmin + 1e-6) * (grid_res - 1))) + 1
        iy = Int(floor((p[2] - ymin) / (ymax - ymin + 1e-6) * (grid_res - 1))) + 1
        grid[ix, iy] = values[i]
    end

    # 3. Apply Zero-Padding
    padded_res = grid_res * pad_factor
    padded_grid = zeros(padded_res, padded_res)

    start_idx = Int(grid_res / 2)
    padded_grid[start_idx:start_idx+grid_res-1, start_idx:start_idx+grid_res-1] .= grid

    return padded_grid, (xmin, xmax, ymin, ymax)
end

println("Sampling from FFT-Accelerated Model v14...")

# Instantiate model with fixed grid resolution parameters
mod_v14_final = model_v14_fft_gaussian(modinputs; grid_res=64, pad_factor=2)

# Run MH sampler for verification (100 samples)
chain_v14 = sample(mod_v14_final, MH(), 100)

# Display summary of spatial and temporal variance components
display(MCMCChains.summarize(chain_v14[[:sigma_sp, :sigma_tm, :rho_tm]]))

# Reconstruct and visualize
stats_v14 = reconstruct_posteriors(mod_v14_final, chain_v14, modinputs)
plt_v14 = plot_posterior_results(stats_v14, modinputs; effect=:spatial)
title!(plt_v14, "Model v14: FFT-Accelerated Spatial Field")
display(plt_v14)


```

### Model v15: Hierarchical Mosaic Spatiotemporal Model (STMV-style)

This model implements a **Hierarchical Mosaic** framework. The spatial domain is treated as a collection of locally stationary regions. Within each region (mosaic), we assume the spatial process is stationary with a local mean and a local length scale. These mosaics are then 'stitched' together through a hierarchical structure where covariate effects are shared globally, but the latent spatial fields vary according to local dynamics. In addition, there is **Soft Boundary Stitching**: Instead of hard k-means assignments, we use a softmax of distances to the mosaic centroids to interpolate the latent field across boundaries; and **Mosaic-Specific Likelihood**: Observation noise ($\sigma_y$) is now estimated per-mosaic, allowing the model to adapt to regional differences in data quality.

This approach is designed to be highly parallelizable, as local innovations can be sampled with high efficiency on multi-core systems.

```{julia}

using LinearAlgebra, SparseArrays, Random

@model function model_v15_refined_mosaic(modinputs, ::Type{T}=Float64; n_mosaics=5, m_rff=20, offset=modinputs.offset) where {T}
    y = modinputs.y
    N_obs = length(y)
    
    # --- 1. Global & Hierarchical Priors ---
    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    mu_global ~ Normal(0, 1)
    sigma_mu_local ~ Exponential(1.0)

    # Local Parameters per Mosaic
    mu_local ~ filldist(Normal(mu_global, sigma_mu_local), n_mosaics)
    l_local ~ filldist(Gamma(2, 1), n_mosaics)
    sigma_local ~ filldist(Exponential(1.0), n_mosaics)
    sigma_y_local ~ filldist(Exponential(1.0), n_mosaics) # Localized noise scale

    # Weights for each mosaic's RFF field
    w_local = [Vector{T}(undef, m_rff) for _ in 1:n_mosaics]
    for m in 1:n_mosaics; w_local[m] ~ MvNormal(zeros(m_rff), I); end

    # Categorical Covariates (Shared)
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 2. Spatial Indexing & Soft Boundary Weights ---
    coords = hcat([p[1] for p in modinputs.pts_raw], [p[2] for p in modinputs.pts_raw])
    R = kmeans(coords', n_mosaics)
    centroids = R.centers # 2 x n_mosaics

    # Pre-sample RFF frequencies
    Random.seed!(42)
    Om_m = [randn(m_rff, 2) for _ in 1:n_mosaics]
    Ph_m = [rand(m_rff) for _ in 1:n_mosaics]

    # --- 3. Likelihood with Soft Integration ---
    for i in 1:N_obs
        pt = [coords[i,1], coords[i,2]]
        
        # Calculate Softmax weights based on distance to centroids for smooth stitching
        dists = [sum((pt .- centroids[:, m]).^2) for m in 1:n_mosaics]
        weights_stitching = exp.(-dists) ./ sum(exp.(-dists))
        
        eta_spatial_combined = zero(T)
        sigma_y_combined = zero(T)
        
        for m in 1:n_mosaics
            # Local RFF Field Calculation
            z_proj = sqrt(2/m_rff) .* cos.( (Om_m[m] * pt ./ l_local[m]) .+ (Ph_m[m] .* 2π) )
            local_field = mu_local[m] + dot(z_proj, w_local[m] .* sigma_local[m])
            
            # Blend local field and local noise
            eta_spatial_combined += weights_stitching[m] * local_field
            sigma_y_combined += weights_stitching[m] * sigma_y_local[m]
        end

        mu = offset[i] + eta_spatial_combined
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        
        Turing.@addlogprob! modinputs.weights[i] * logpdf(Normal(mu, sigma_y_combined + 1e-6), y[i])
    end
end


println("Running Refined Mosaic Model v15.1...")
mod_v15_ref = model_v15_refined_mosaic(modinputs; n_mosaics=4, m_rff=15)

# Calculate MAP to define the variable map_v15_ref
# map_v15_ref = maximum_a_posteriori(mod_v15_ref)

# Check the distribution of local noise scales using the now-defined map_v15_ref
# println("Local Noise Scales (MAP): ", [map_v15_ref[Symbol("sigma_y_local[$i]")] for i in 1:4])

chain_v15_ref = sample(mod_v15_ref, MH(), 100) #, initial_params=InitFromParams(map_v15_ref))
summarystats(chain_v15_ref[[:mu_global, :sigma_mu_local]])

```


## Model v16: Integrated Spatiotemporal Mosaic (ISM)

This model is the synthesis of the project's development. It integrates:
- **Hierarchical Mosaics**: Adaptive local stationarity.
- **Soft-Blending**: Global continuity across mosaic boundaries.
- **Non-Separable RFF**: Joint [Space, Time] kernels within each region.
- **Regional Noise**: Mosaic-specific $\sigma_y$ scales.


```{julia}
@model function model_v16_integrated_mosaic(modinputs, ::Type{T}=Float64; n_mosaics=4, m_rff=20, offset=modinputs.offset) where {T}
    y = modinputs.y
    N_obs = length(y)

    # --- 1. Global Hierarchical Priors ---
    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    mu_global ~ Normal(0, 1)
    sigma_mu_local ~ Exponential(0.5)

    # Shared Categorical Effects
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 2. Local Mosaic Hyperparameters ---
    mu_local ~ filldist(Normal(mu_global, sigma_mu_local), n_mosaics)

    # Refactored: Use arraydist for joint lengthscales instead of a loop
    l_joint ~ arraydist([filldist(Gamma(2, 1), 3) for _ in 1:n_mosaics])

    sigma_local ~ filldist(Exponential(1.0), n_mosaics)
    sigma_y_local ~ filldist(Exponential(1.0), n_mosaics)

    # Local Weights for Non-Separable RFF
    w_local = [Vector{T}(undef, m_rff) for _ in 1:n_mosaics]
    for m in 1:n_mosaics; w_local[m] ~ MvNormal(zeros(m_rff), I); end

    # --- 3. Geometric Indexing ---
    xs = [p[1] for p in modinputs.pts_raw]
    ys = [p[2] for p in modinputs.pts_raw]
    ts = Float64.(modinputs.time_idx)

    # Normalize [X, Y, T] to [0, 1] range for RFF stability
    X_joint = hcat(
        (xs .- minimum(xs)) ./ (maximum(xs) - minimum(xs) + 1e-6),
        (ys .- minimum(ys)) ./ (maximum(ys) - minimum(ys) + 1e-6),
        (ts .- minimum(ts)) ./ (maximum(ts) - minimum(ts) + 1e-6)
    )

    # Static centroids for stitching (calculated once)
    coords_2d = X_joint[:, 1:2]
    R = kmeans(coords_2d', n_mosaics)
    centroids = R.centers

    # Fixed RFF Frequencies
    Random.seed!(42)
    Om_base = [randn(m_rff, 3) for _ in 1:n_mosaics]
    Ph_base = [rand(m_rff) for _ in 1:n_mosaics]

    # --- 4. Predictive Synthesis ---
    for i in 1:N_obs
        pt_3d = X_joint[i, :]
        pt_2d = pt_3d[1:2]

        # Soft Boundary Weights (Softmax of distance to centroids)
        dists = [sum((pt_2d .- centroids[:, m]).^2) for m in 1:n_mosaics]
        weights_st = exp.(-dists) ./ sum(exp.(-dists))

        eta_spatial_time = zero(T)
        sigma_y_total = zero(T)

        for m in 1:n_mosaics
            # Scale base frequencies by local lengthscales [Lx, Ly, Lt]
            # l_joint is now a matrix where each column corresponds to a mosaic
            Om = Om_base[m] .* (1.0 ./ (l_joint[:, m] .+ 1e-6)')

            # Local Non-Separable Field
            z_proj = sqrt(2/m_rff) * cos.( (Om * pt_3d) .+ (Ph_base[m] .* 2π) )
            local_field = mu_local[m] + dot(z_proj, w_local[m] .* sigma_local[m])

            eta_spatial_time += weights_st[m] * local_field
            sigma_y_total += weights_st[m] * sigma_y_local[m]
        end

        mu = offset[i] + eta_spatial_time
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end

        Turing.@addlogprob! modinputs.weights[i] * logpdf(Normal(mu, sigma_y_total + 1e-6), y[i])
    end
end
println("Evaluating Integrated Spatiotemporal Mosaic (Model v16)... ")

# Fix: Ensure pts_raw matches the length of y and time_idx
# The Lip Cancer data has 56 areas and 10 years.
n_areas_lip = 56
n_years_lip = 10

# Re-align points to match the observation count
pts_full_lip = repeat(modinputs.pts_raw[1:n_areas_lip], n_years_lip)
lip_inputs_v16 = merge(modinputs, (pts_raw = pts_full_lip,))

mod_v16 = model_v16_integrated_mosaic(lip_inputs_v16; n_mosaics=3, m_rff=15)

# Short MH chain for verification
chain_v16 = sample(mod_v16, MH(), 100)

summary_params = [:mu_global, Symbol("sigma_y_local[1]"), Symbol("sigma_y_local[2]")]
display(summarystats(chain_v16[intersect(summary_params, names(chain_v16))]))

# Reconstruct and Plot
stats_v16 = reconstruct_posteriors(mod_v16, chain_v16, lip_inputs_v16)
plt_sp_v16 = plot_posterior_results(stats_v16, lip_inputs_v16; effect=:spatial)
title!(plt_sp_v16, "Model v16: Integrated Mosaic Spatial Field")
display(plt_sp_v16)

```



##  The Mixed-Sampler Strategy: How the Model Learns

Inference in CARSTM requires a Mixed-Sampler Gibbs approach, as no single algorithm is optimal for the entire parameter space.


* Elliptical Slice Sampling (ESS):
  * Parameter: Latent Gaussian Fields (ICAR/AR1 components).
  * Justification: Analytically exact for Gaussian priors; requires no tuning or gradient information (Murray et al., 2010).
  * Pro-Tip: Ensure variables are zero-centered for maximal stability.
* No-U-Turn Sampler (NUTS):
  * Parameter: Differentiable Regression Coefficients (Fixed Effects).
  * Justification: Adaptively finds optimal path lengths in complex posterior geometries (Hoffman & Gelman, 2014).
  * Pro-Tip: Increase target_acceptance to 0.8 or 0.9 if divergent transitions occur.
* Particle Gibbs (PG):
  * Parameter: Discrete indicators (e.g., Zero-Inflation states z_i \in \{0, 1\}).
  * Justification: Uses sequential Monte Carlo to update latent paths that are non-differentiable.
  * Pro-Tip: PG(40) is a robust production default; often paired with NUTS in a mixed-sampler synergy.
* Metropolis-Hastings (MH):
  * Parameter: Simple Scalars (Variance \sigma, Correlation \rho).
  * Justification: Computationally cheap for low-dimensional, bounded parameters.

For production-grade point estimates, we may utilize ADVI (Automatic Differentiation Variational Inference). In these cases, increasing the n_samples for the ELBO gradient estimation is critical to stabilize convergence against the noise of complex spatial interactions.



### Variational Inference (ADVI)

ADVI is suitable for rapid "smoke tests" or approximating the Evidence Lower Bound (ELBO).

* Technical Tuning: The n_samples argument controls gradient variance. Start with ADVI(1, 1000). If the ELBO plot exhibits excessive noise, move to ADVI(10, 2000) to stabilize convergence in high-dimensional spatial landscapes.


--------------------------------------------------------------------------------


## Implementation Recommendations and Research Frontiers

Critical Data Preparation

* Time Standardization: Map raw indices to the [0, 1] interval to prevent trigonometric overflow during RFF basis generation.
* Graph Connectivity: Utilize ensure_connected! logic. Disconnected spatial "islands" result in singular precision matrices and immediate sampler failure.
* Prior Selection: Utilize Penalized Complexity (PC) Priors for standard deviations. These provide principled shrinkage toward a simpler base model (e.g., zero variance) unless the data provides strong evidence of structure.

The Future of CARSTM

* Kronecker Product Decomposition: Necessary for maintaining O(N) memory complexity as spatial (S) and temporal (T) units grow, preventing O(S \times T)^2 memory growth.
* Copula-based Interactions: Moving beyond Gaussianity in latent spaces to model tail-dependence in extreme events (e.g., synchronized flood risks).
* Dynamic Network CAR: Allowing the adjacency matrix W to evolve over time to capture infrastructure or policy shifts.

The CARSTM framework in Julia offers a robust, scalable environment for tackling spatiotemporal questions that were previously computationally prohibitive, leveraging the synergy between GMRF efficiency and the flexibility of Deep Gaussian Processes.
The CARSTM framework in Julia offers a robust, scalable environment for tackling spatio-temporal questions that were previously computationally prohibitive, leveraging the synergy between GMRF efficiency and the flexibility of Deep Gaussian Processes.


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
using Pkg
pkgs_au = ["Random", "Statistics", "LinearAlgebra", "DataFrames",
       "StatsBase", "SparseArrays", "Plots", "StatsPlots", "StaticArrays",
        "JLD2", "LibGEOS", "Graphs", "DelaunayTriangulation" ]
Pkg.add(pkgs_au)
for pk in pkgs_au; @eval using $(Symbol(pk)); end


# For CARSTM 
using Pkg
pkgs_carstm = ["Random",   "Distributions", "Statistics", "MCMCChains", "DataFrames",
        "LinearAlgebra", "Clustering", "StatsBase", "HypothesisTests",
        "JLD2", "FFTW",  "SparseArrays", "StaticArrays", "FillArrays",
         "Bijectors", "DynamicPPL", "AdvancedVI", "Optimisers", "PosteriorStats",  "Turing" ]
Pkg.add(pkgs_carstm)
for pk in pkgs_carstm;  @eval using $(Symbol(pk)) end



# Pkg.precompile()
# Pkg.instantiate()
# Pkg.gc()


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
 
### Scottish lip cancer data 

First we begin with a classic data set, the [Scottish Lip Cancer data](https://mc-stan.org/users/documentation/case-studies/icar_stan.html) which has been a standard to test upon. There are 56 areal units. We do not have access to the map positional data, but we do have the adjacency information from which we can infer approximate spatial topology:  
  
```{julia}  
data = scottish_lip_cancer_data_spacetime();

display(keys(data))
display(Dict(k => size(v) for (k, v) in pairs(data) if k != :au))

au = data.au ;

  println("Number of units: ", length(au.centroids))
  println("Graph connectivity: ", is_connected(au.graph))
  
  # approximate "map":
  plt = plot_spatial_graph( au; title="Lip Cancer Inferred Map and Topology", domain_boundary=au.hull_coords)
  display(plt)
```

In the data Tuple, we have counts (y) of cancer incidence and population size in each area is used as offsets (log_offset) in a simple Poisson model.  We also simulate a 10-"year" temporal process simulated as a random walk with magnitude 0.5 and a covariate effect (X: an area-specific continuous covariate that represents the proportion of the population employed in agriculture, fishing, or forestry). An overall random uniform observation error of magnitude 0.2 is added with a count then taken as the overall, rounded integer value.

We reformat this data further with 'prepare_model_inputs()' to create structured inputs for the model. 
 
```{julia}
 
# prepapre model inputs  
# Note: We pass n_cat which is used for the RW2 categorical smoothing priors
ncats = 13 # dummy variable
modinputs = prepare_model_inputs(data.y, pts, data.area_idx, data.time_idx, data.W, ncats )

mod = model_v5_poisson(modinputs; use_zi=false)  # zi=false mean no zero-inflation (default)

```

Once defined, we can sample. We use a Gibbs sampling approach that permits alternative and optentially optimal samplers specific to each variable. 

```{julia}

# Define the Gibbs sampler tailored for Poisson
optimal_gibbs_poisson = Gibbs(
    (:u_icar, :u_iid, :f_tm_raw, :st_int_raw) => ESS(),
    (:beta_cov) => NUTS(500, 0.65),
    (:sigma_sp, :phi_sp, :sigma_tm, :rho_tm, :sigma_int, :sigma_rw2, :phi_zi) => MH()
)

chain = sample(mod, optimal_gibbs_poisson, 200; progress=true)

# required for waic computation:
using LogExpFunctions: logistic
using LogExpFunctions: logsumexp

results = model_results_comprehensive(mod, chain, modinputs, au)

println("WAIC: ", round(results.waic, digits=2))
println("RMSE: ", round(results.rmse, digits=4))
println("Pearson R: ", round(results.pearson_r, digits=4))

# 4. Display visual diagnostics
display(results.plots.ppc.plot_scatter)
display(results.plots.temporal)

display(results.plots.spatial)
display(results.plots.st_denoised)
display(results.plots.st_noisy)




```


### Simulated data

```{julia}

# Data Generation
n_pts = 100
n_time = 30
 
data = generate_sim_data(n_pts, n_time; rndseed=42);
keys(data)
pairs(data)
Dict(k => size(v) for (k, v) in pairs(data))

# extract quantities:
(; pts, y_sim, y_binary, time_idx, weights, trials, cov_indices) = data

plot_kde_simple(pts, sd_extension_factor=0.25, title="Spatial Intensity (KDE)")
 
# Define constraints for benchmarking
# Ensure these are Integers to avoid MethodError in StatsBase.sample
ntot = size(pts, 1) 

min_time_slices = 5
target_density = 20 # number per areal unit 
target_units = Int(floor( ntot / target_density ))
min_total_arealunits = target_units / 10
max_total_arealunits = target_units * 10
min_points = 1
max_points = Int(floor(ntot / min_total_arealunits ))
min_area = 0.5
max_area = 9
cv_min = 1
buffer_dist = 0.8
tolerance = 0.05

test_configs = [ :cvt, :kvt, :qvt, :bvt, :avt ]

results = []
plots = []

for m in test_configs
    println("Testing method: $m")
    local au
    try
        au = assign_spatial_units( pts, m;
            target_units = target_units,
            min_total_arealunits=min_total_arealunits,
            max_total_arealunits=max_total_arealunits,
            min_time_slices = min_time_slices,
            time_idx = time_idx,
            buffer_dist=buffer_dist,
            tolerance=tolerance,
            cv_min=cv_min,
            min_points=min_points,
            max_points=max_points,
            min_area=min_area,
            max_area=max_area)

        met = calculate_metrics(au)
        push!(results, (
          method=m,
          units=length(au.centroids),
          mean_dens=met.mean_density,
          sd_dens=met.sd_density,
          cv_dens=met.cv_density,
          termination=au.termination_reason
        ))

        p = plot_spatial_graph( au; title="Method: $m", domain_boundary=au.hull_coords)
        push!(plots, p)
    catch e
        @error "Method $m failed: $e"
    end
end

if !isempty(results)
    display(DataFrame(results))
    display(Plots.plot(plots..., layout=(3, 2), size=(600, 800)))
end


```

Conclusion: All methods seem reasonable, but AVT seems to have lowest density and SD and CV.. approaches a Poisson distribution best.


### Simulated data 

```{julia}
n_pts = 100
n_time = 15
 
data = generate_sim_data(n_pts, n_time; rndseed=42);

(; pts, y_sim, y_binary, time_idx, weights, trials, cov_indices) = data
ntot = size(pts, 1) 

min_time_slices = 5
target_density = 20 # number per areal unit 
target_units = Int(floor( ntot / target_density ))
min_total_arealunits = target_units / 10
max_total_arealunits = target_units * 10
min_points = 1
max_points = Int(floor(ntot / min_total_arealunits ))
min_area = 0.5
max_area = 9
cv_min = 1
buffer_dist = 0.8
tolerance = 0.05
method = :avt

au = assign_spatial_units( pts, method;
  target_units = target_units,
  min_total_arealunits=min_total_arealunits,
  max_total_arealunits=max_total_arealunits,
  min_time_slices = min_time_slices,
  time_idx = time_idx,
  buffer_dist=buffer_dist,
  tolerance=tolerance,
  cv_min=cv_min,
  min_points=min_points,
  max_points=max_points,
  min_area=min_area,
  max_area=max_area)


p = plot_spatial_graph( au; title="Method: $method", domain_boundary=au.hull_coords)

 
```

### Model variations 


```{julia}

# Setup shared precomputations
n_categories = 13

modinputs_gaussian = prepare_model_inputs(y_sim, pts, au.assignments, time_idx, W_sym, n_categories)
modinputs_count = prepare_model_inputs(y_count, pts, au.assignments, time_idx, W_sym, n_categories)
modinputs_binomial = prepare_model_inputs(y_binary, pts, au.assignments, time_idx, W_sym, n_categories)
modinputs_lognormal = prepare_model_inputs( log.(y_sim), pts, au.assignments, time_idx, W_sym, n_categories)



# Define the full model set with corrected data inputs for count families
models_to_bench = Dict(
    "v1_gaussian"         => () -> model_v1_gaussian(modinputs_gaussian),
    "v2_rff_gaussian"     => () -> model_v2_rff_gaussian(modinputs_gaussian),
    "v3_lognormal"        => () -> model_v3_lognormal(modinputs_lognormal),
    "v4_binomial"         => () -> model_v4_binomial(modinputs_binomial; trials=data.trials),
    "v5_poisson"          => () -> model_v5_poisson(modinputs_count),
    "v6_negativebinomial" => () -> model_v6_negativebinomial(modinputs_count),
    "v7_deep_gp_binomial" => () -> model_v7_deep_gp_binomial(modinputs_binomial; trials=data.trials),
    "v8_deep_gp_gaussian" => () -> model_v8_deep_gp_gaussian(modinputs_gaussian),
    "v9_continuous_gaussian" => () -> model_v9_continuous_gaussian(data.cov_continuous, modinputs_gaussian),
    "v10_deep_gp_3layer"  => () -> model_v10_deep_gp_3layer_gaussian(modinputs_gaussian)
)

n_bench_iters = 500
bench_results = Dict{String, Float64}()

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
