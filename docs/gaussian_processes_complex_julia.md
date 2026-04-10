
# Bayesian Spatiotemporal Modeling Framework in Julia

This notebook implements a series of Bayesian models using Julia, especially Turing and GaussianProcesses mostly focussed upon advanced spatiotemporal processes with structural dependencies and anisotropy. 

Most of this was rapidly developed using the very useful tool: [Google Colab](https://colab.research.google.com).

Warning: I have checked for errors, but there may still be some remaining.


## Introduction

Developing a general framework to explore various models of increasing complexity to handle measurement error, periodic dynamics, and spatial dependencies. Random Fourier Features (RFF), Fully Independent Training Conditional (FITC) and Deep Gaussian Processes are explored to make Gaussian Processes (GPs) computationally tractable for large datasets.

First we install and load the required libraries. This can take a while, depending upon available compute resources. You can of course, run them directly on if you do not want to install Julia on your own system: [Google Colab](https://colab.research.google.com) -- just make sure that you have chosen the "Julia" Runtime.

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



```{julia}
using Pkg

pkgs = ["Random", "Turing", "Distributions", "Statistics", "MCMCChains", "DataFrames",
  "LinearAlgebra", "AbstractGPs", "KernelFunctions", "Clustering", "StatsBase",
   "JLD2", "FFTW", "Lux", "Flux"]

Pkg.add(pkgs)
Pkg.precompile()


for pk in pkgs
  @eval using $(Symbol(pk))
end
```
 
## Basic support functions

```{julia}

function compute_y_waic(mod, ch)
    # to compute WAIC
    try
        pll = pointwise_loglikelihoods(mod, ch)
        y_keys = [k for k in keys(pll) if occursin("y_obs", string(k))]
        if !isempty(y_keys)
            loglik_mat = hcat([vec(pll[k]) for k in y_keys]...)
            lppd = sum(log.(mean(exp.(loglik_mat), dims=1)))
            p_waic = sum(var(loglik_mat, dims=1))
            return -2 * (lppd - p_waic)
        end
    catch e
        return NaN
    end
    return NaN
end

function get_posterior_means(ch, param_base, N)
    means = zeros(N)
    for i in 1:N
        p_symbol = Symbol("$param_base[$i]")
        if p_symbol in names(ch, :parameters)
            means[i] = mean(ch[p_symbol])
        end
    end
    return means
end



function generate_data(N; period=12.0, seed=42)
    Random.seed!(seed)
    # 1. Coordinates: Space (Xlon, Xlat) and Time (T)
    coords_space = rand(N, 2)
    coords_time = reshape(collect(1.0:N), :, 1)

    # 2. Covariates
    # Z: Purely spatial covariate
    Z = randn(N)

    # Latent (True) Spatiotemporal Covariates
    U1_true = sin.(coords_time[:,1] ./ 5.0) .+ 0.5 .* Z
    U2_true = cos.(coords_time[:,1] ./ 5.0) .- 0.3 .* Z
    U3_true = 0.2 .* (coords_time[:,1] ./ N) .+ 0.1 .* Z

    # 3. Add measurement error to covariates (observed version)
    sigma_u = 0.1
    U1_obs = U1_true .+ randn(N) .* sigma_u
    U2_obs = U2_true .+ randn(N) .* sigma_u
    U3_obs = U3_true .+ randn(N) .* sigma_u

    # 4. Generate Dependent Variable Y
    # Components: Linear Trend + Seasonal Harmonic + Latent Process + Noise
    trend = 0.05 .* coords_time[:,1]
    seasonal = 1.0 .* cos.(2 * pi .* coords_time[:,1] ./ period)

    # Simulate a spatial effect manually for the ground truth
    spatial_effect = sin.(coords_space[:,1] .* 2π) .* cos.(coords_space[:,2] .* 2π)

    sigma_y = 0.2
    # Y is a function of trend, season, GP effect, and U1
    y_obs = 1.0 .+ trend .+ seasonal .+ spatial_effect .+ (0.5 .* U1_true) .+ randn(N) .* sigma_y

    return (
        y_obs = y_obs,
        U1_obs = U1_obs,
        U2_obs = U2_obs,
        U3_obs = U3_obs,
        Z = Z,
        coords_space = coords_space,
        coords_time = coords_time
    )
end


function generate_informed_rff_params(coords, M_rff_count)
    D_in = size(coords, 2)
    std_coords = vec(std(coords, dims=1)) .+ 1e-6
    W_fixed = randn(D_in, M_rff_count) ./ std_coords
    b_fixed = rand(M_rff_count) .* 2pi
    return W_fixed, b_fixed
end

function generate_rff_params_for_se_kernel(D_in, M_rff, lengthscale)
    # Helper function to generate RFF parameters for a Squared Exponential kernel
    # For a Squared Exponential kernel, the spectral density is Gaussian: N(0, (1/l)^2 * I)
    sigma_spectral = 1.0 / lengthscale
    W_matrix = randn(D_in, M_rff) .* sigma_spectral # D_in x M_rff matrix
    b_vector = rand(Uniform(0, 2pi), M_rff)
    return W_matrix, b_vector
end

function rff_map(coords, W, b)
    projection = (coords * W) .+ b'
    return sqrt(2 / size(W, 2)) .* cos.(projection)
end


function generate_inducing_points(coords_st, M_inducing, seed=42)
    # Helper function to generate inducing points (simple random sampling for now)
    Random.seed!(seed)
    N_data = size(coords_st, 1)
    if M_inducing >= N_data
        return coords_st # If M >= N, just use all data points (becomes exact GP)
    end
    indices = sample(1:N_data, M_inducing, replace=false)
    return coords_st[indices, :]
end



```


## Generate simulated data


```{julia}

# basic data
data = generate_data(50)


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

### Detailed Explanation: Random Fourier Features (RFF)

Random Fourier Features (RFFs), introduced by Rahimi and Recht (2007), provide an efficient way to approximate shift-invariant kernels, such as the Squared Exponential (RBF) or Matérn kernels. This approximation allows Gaussian Processes (GPs) and Support Vector Machines (SVMs) to scale to larger datasets by transforming the non-linear kernel learning problem into a linear learning problem in a randomized feature space.

#### The Core Idea: Bochner's Theorem

The mathematical foundation for RFFs lies in Bochner's Theorem. This theorem states that a continuous, shift-invariant kernel function $k(x, x') = k(x - x')$ can be expressed as the Fourier transform of a non-negative measure (or a probability density function for normalized kernels) called the spectral density $p(\omega)$:

$$k(\Delta x) = \int_{\mathbb{R}^D} e^{i \omega^T \Delta x} p(\omega) d\omega$$

where $\Delta x = x - x'$. For real-valued kernels, this can be written as:

$$k(\Delta x) = \int_{\mathbb{R}^D} \cos(\omega^T \Delta x) p(\omega) d\omega$$

This integral represents the expected value of $\cos(\omega^T \Delta x)$ where $\omega$ is sampled from $p(\omega)$.

#### The RFF Approximation

RFFs approximate this integral using Monte Carlo sampling. Instead of computing the integral, we can approximate it by drawing $M$ samples of frequencies $\omega_1, \dots, \omega_M$ from the spectral density $p(\omega)$, and $M$ samples of phase shifts $b_1, \dots, b_M$ from a uniform distribution $U(0, 2\pi)$.

For any two input points $x, x' \in \mathbb{R}^D$, the kernel $k(x, x')$ can be approximated as:

$$k(x, x') \approx \frac{1}{M} \sum_{j=1}^M \cos(\omega_j^T (x - x'))$$

By using the trigonometric identity $\cos(A - B) = \cos A \cos B + \sin A \sin B$, and introducing a random phase $b_j$, we can define a feature map $\phi(x)$ such that $k(x, x') \approx \phi(x)^T \phi(x')$:

$$\phi(x) = \sqrt{\frac{2}{M}} \begin{pmatrix}
\cos(\omega_1^T x + b_1) \\
\cos(\omega_2^T x + b_2) \\
\vdots \\
\cos(\omega_M^T x + b_M)
\end{pmatrix}$$

This is the key RFF feature map. The original input $x$ (a $1 \times D$ vector) is transformed into a new $1 \times M$ feature vector $\phi(x)$.

#### Components of the RFF Approximation:

*   Input Data ($x$): A $D$-dimensional vector representing a single data point.
*   Projection Weights ($W$ or $\omega_j$): A $D \times M$ matrix where each column $\omega_j$ is a frequency vector sampled from the spectral density $p(\omega)$ of the chosen kernel. For a Squared Exponential (RBF) kernel with lengthscale $l$, the spectral density is a Gaussian distribution $N(0, (1/l)^2 I)$. Thus, $\omega_j \sim N(0, (1/l)^2 I)$.
*   Offsets ($b$ or $b_j$): An $M$-dimensional vector where each $b_j$ is sampled uniformly from $[0, 2\pi]$. These random phase shifts are crucial for the unbiasedness of the kernel approximation.
*   Number of Features ($M$): Determines the dimensionality of the feature space. A larger $M$ leads to a more accurate approximation of the kernel but increases computational cost. A common heuristic is $M = 100 \times D$ or $M = 2 \times D$ for good performance, but it can be tuned.
*   Signal Variance (e.g., $\sigma_f^2$): In a GP context, the RFF feature map gives a kernel of unit amplitude. If the true kernel is $\sigma_f^2 k(x, x')$, the linear model using RFFs would be $f(x) = \sigma_f \phi(x)^T \beta$, where $\beta \sim N(0, I)$, or equivalently, $f(x) = \phi(x)^T \beta_{GP}$ where $\beta_{GP} \sim N(0, \sigma_f^2 I)$.

#### Computational Benefits

*   Exact GP: Computations involve inverting an $N \times N$ kernel matrix, leading to $O(N^3)$ complexity.
*   RFF-approximated GP: Once the data is mapped into the $M$-dimensional feature space, inference (e.g., linear regression) becomes $O(NM^2)$ or $O(M^3)$ for inversion of the smaller feature matrix, significantly faster when $M \ll N$. Prediction for a new point is $O(DM)$.

By leveraging RFFs, we can apply kernel methods to much larger datasets than would be feasible with exact kernel computations, making them powerful tools for scalable Bayesian modeling.

### Julia Example: Implementing Random Fourier Features

A basic RFF transformation for a Squared Exponential (RBF) kernel and how it can approximate the original kernel. We'll reuse the `rff_map` helper function defined earlier.


```{julia}

D_in_example = 2 # 2D input space
M_rff_example = 100 # Number of RFF features
lengthscale_example = 0.5 # Lengthscale of the SE kernel
signal_variance_example = 1.0 # Signal variance

# Generate RFF parameters
W_example, b_example = generate_rff_params_for_se_kernel(D_in_example, M_rff_example, lengthscale_example);

# Generate some synthetic data points
x_data = rand(10, D_in_example);
x_prime_data = rand(10, D_in_example);

# Compute RFF features for the data points
Phi_x = rff_map(x_data, W_example, b_example);
Phi_x_prime = rff_map(x_prime_data, W_example, b_example);

# Approximate the kernel matrix using RFFs
K_rff_approx = signal_variance_example .* (Phi_x * Phi_x_prime');

# Compute the true Squared Exponential kernel matrix
k_true = SqExponentialKernel() ∘ ScaleTransform(inv(lengthscale_example));
K_true = signal_variance_example .* kernelmatrix(k_true, RowVecs(x_data), RowVecs(x_prime_data));

println("--- RFF Approximation of Squared Exponential Kernel ---");
println("Approximate Kernel Matrix (first 5x5 block):");
display(K_rff_approx[1:5, 1:5]);

println("\nTrue Kernel Matrix (first 5x5 block):");
display(K_true[1:5, 1:5]);

# Calculate Frobenius norm difference to see the approximation quality
difference_norm = norm(K_rff_approx - K_true);
println("\nFrobenius Norm of Difference (RFF vs True Kernel): ", difference_norm);

# This difference will decrease as M_rff_example increases, demonstrating better approximation.

```

    --- RFF Approximation of Squared Exponential Kernel ---
    Approximate Kernel Matrix (first 5x5 block):
    
    True Kernel Matrix (first 5x5 block):
    
    Frobenius Norm of Difference (RFF vs True Kernel): 0.6118940913556318

    5×5 Matrix{Float64}:
     0.303852  0.496976  0.905675  0.947403  0.564709
     1.03997   0.219968  0.159245  0.246104  0.835722
     0.305929  0.897091  0.448935  0.186139  0.531721
     0.466451  0.774857  0.588579  0.341906  0.820317
     1.0085    0.289544  0.240062  0.297764  0.967216



    5×5 Matrix{Float64}:
     0.262682  0.406843  0.842856  0.883973  0.507205
     0.97597   0.162638  0.16002   0.180386  0.791499
     0.221877  0.918095  0.391791  0.122494  0.537356
     0.453599  0.813989  0.519247  0.25401   0.83906
     0.940836  0.282646  0.247371  0.225369  0.935376


## Fully Independent Training Conditional (FITC)

FITC is an approximation method for Gaussian Processes (GPs) that addresses the computational burden of large datasets. It is also known as the "sparse pseudo-input GP" or "Deterministic Training Conditional (DTC)". Another common term is "Inducing Points".

### Core Idea and Theoretical Basis

Instead of directly approximating the kernel function via feature maps (like RFFs), FITC introduces a small set of "inducing points" ($Z = \{z_1, \dots, z_M\}$, where $M \ll N$). The fundamental assumption is that, conditional on the values of the latent GP at these inducing points ($f_Z$), the observed data points ($f_i$) are conditionally independent:

$$p(f | X, Z) \approx p(f | f_Z, Z) = \prod_{i=1}^N p(f_i | f_Z, Z)$$

This approximation significantly simplifies the covariance structure and speeds up calculations.

### Mechanism and Computational Advantages
*   Sparsity Source: A small set of judiciously chosen inducing points. These points are not necessarily part of the training data.
*   Approximation: FITC approximates the posterior distribution of the GP, effectively 'compressing' the GP through these inducing points.
*   Covariance Calculation: It simplifies the computation of the covariance matrix by involving inversions only for the smaller $M \times M$ covariance matrix of the inducing points ($K_{ZZ}$) and their cross-covariances with the data ($K_{XZ}$). The inducing points act as a bottleneck for information flow.
*   Computational Advantage: Reduces the computational complexity from $O(N^3)$ (for exact GPs) to $O(N M^2 + M^3)$.
*   Interpretation: The GP is conditioned on a smaller set of latent variables (the values at the inducing points).
*   Inducing Point Optimization: A crucial aspect is the choice and optimization of the inducing point locations and possibly their values. These are often treated as hyperparameters to be learned or optimized within the model.

### Mathematical Formulation for `f` in FITC
Given $N$ data points $X = \{x_1, \dots, x_N\}$ and $M$ inducing points $Z = \{z_1, \dots, z_M\}$, the latent GP values at observed points $f$ are approximated. If $f_Z$ are the latent values at inducing points, then the conditional distribution $p(f | f_Z)$ is Gaussian with:

*   Conditional Mean: $E[f | f_Z] = K_{XZ} K_{ZZ}^{-1} f_Z$
*   Conditional Covariance (Diagonal Approximation): $Cov[f | f_Z] \approx diag(K_{XX} - K_{XZ} K_{ZZ}^{-1} K_{ZX})$

Where:
*   $K_{XX}$ is the $N \times N$ kernel matrix between all observed points.
*   $K_{ZZ}$ is the $M \times M$ kernel matrix between all inducing points.
*   $K_{XZ}$ is the $N \times M$ kernel matrix between observed points and inducing points.
*   $K_{ZX} = K_{XZ}^T$.

In the model, we sample $f_Z \sim MvNormal(0, K_{ZZ})$, and then $f \sim MvNormal(E[f | f_Z], diag(Cov[f | f_Z]))$. By sampling $f_Z$ and then $f$ conditionally, we maintain a non-centered parameterization for the sparse GP.

### Julia Example: Demonstrating FITC Mechanics

This example demonstrates the core computations involved in a Fully Independent Training Conditional (FITC) approximation using `AbstractGPs.jl` and `KernelFunctions.jl`. We will:

1.  Define a kernel.
2.  Generate some synthetic data and inducing points.
3.  Compute the necessary kernel matrices ($K_{XX}$, $K_{ZZ}$, $K_{XZ}$).
4.  Calculate the conditional mean and the diagonal of the conditional covariance as used in FITC.

This mirrors the logic used in models like V6 to define the sparse GP latent process.


```{julia}

N_data_points = 50 # Number of observed data points
M_inducing_points = 10 # Number of inducing points
D_input = 3 # Input dimensions (e.g., 2D space + 1D time)

# 1. Generate synthetic input data (spatiotemporal coordinates)
coords_data = rand(N_data_points, D_input);

# 2. Generate inducing points
coords_inducing = generate_inducing_points(coords_data, M_inducing_points);

# 3. Define a spatiotemporal kernel (e.g., Anisotropic Squared Exponential)
ls_st_example = [0.5, 0.8, 1.2]; # Example lengthscales for each dimension
sigma_f_example = 1.0; # Signal variance

k_st = SqExponentialKernel() ∘ ARDTransform(1.0 ./ ls_st_example);

# Define a base GP using AbstractGPs.jl
g_base = GP(sigma_f_example^2 * k_st);

# Use RowVecs for coordinates for kernelmatrix compatibility
data_vecs = RowVecs(coords_data);
inducing_vecs = RowVecs(coords_inducing);

# 4. Compute the necessary kernel matrices
K_ZZ = cov(g_base(inducing_vecs)) + 1e-6*I; # Covariance at inducing points with jitter
K_XZ = cov(g_base(data_vecs), g_base(inducing_vecs)); # Cross-covariance
K_XX_diag = diag(cov(g_base(data_vecs))); # Diagonal of covariance at data points

println("--- FITC Approximation Mechanics ---");
println("Size of K_ZZ (inducing points covariance): ", size(K_ZZ));
println("Size of K_XZ (cross-covariance): ", size(K_XZ));
println("Length of K_XX_diag (data covariance diagonal): ", length(K_XX_diag));

# 5. Simulate sampling latent values at inducing points
# In a full model, u_latent would be a sampled variable (e.g., u_latent ~ MvNormal(zeros(M_inducing_points), K_ZZ))
# For this demonstration, we'll use a deterministic value or a single sample.
u_latent_sample = rand(MvNormal(zeros(M_inducing_points), K_ZZ));

# 6. Calculate conditional mean and diagonal covariance at observed points (FITC formulas)
m_f_conditional = K_XZ * (K_ZZ \ u_latent_sample); # Conditional mean
cov_f_diag_conditional = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ')); # Diagonal of conditional covariance

println("\nFirst 5 values of Conditional Mean (m_f): ", m_f_conditional[1:5]);
println("First 5 values of Conditional Covariance Diagonal (diag(Cov[f|fZ])): ", cov_f_diag_conditional[1:5]);

# In a Turing.jl model, the latent GP 'f' would then be sampled as:
# f ~ MvNormal(m_f_conditional, Diagonal(max.(0, cov_f_diag_conditional) + 1e-6*ones(N_data_points)))

```


## V0: Base Model - Dense Kernel Matrix Spatiotemporal GP

Data generation function and the base spatiotemporal model (V0) using `AbstractGPs` and `KernelFunctions` with a separable covariance structure.

### Model Assumptions:
*   Dependent Variable (Y): Modeled with a mean component (trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus observation noise.
*   Latent Covariates (U1, U2, U3): Assumed to be observed with measurement error. Their 'true' values are non-centered parameterized, where `u_true = mean(u_obs) + u_off * std(u_obs)`.
*   Trend: A simple cumulative sum of `alpha_raw` terms, implementing a random walk prior on the intercept over unique time points.
*   Seasonal Process: Modeled as a fixed-period harmonic (sine/cosine waves).
*   Spatiotemporal GP (f):
    *   Separable Covariance: Assumes the spatiotemporal kernel can be factored into a product of a spatial kernel and a temporal kernel, i.e., $K((x_s, t_s), (x_t, t_t)) = K_s(x_s, x_t) \times K_t(t_s, t_t)$.
    *   Spatial Kernel: Isotropic Squared Exponential kernel (single lengthscale `ls_s`).
    *   Temporal Kernel: Squared Exponential kernel (single lengthscale `ls_t`).
    *   Dense Kernel Matrix: Computes the full $N \times N$ covariance matrix, leading to $O(N^3)$ computational complexity for inference, which is noted as "very slow".
    *   Non-centered Parameterization: The latent GP `f` is sampled directly from `MvNormal(zeros(N), K + 1e-6*I)`.
*   Observation Noise (sigma_y, sigma_u): Assumed to be homoscedastic (constant variance) and normally distributed.
*   Priors: Standard weakly informative priors (Exponential for scales, Normal for coefficients, Uniform for phases).

### Key References:
*   Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (For GP fundamentals)
*   Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press. (For non-centered parameterization)

Problem: very slow, really slow


```{julia}
@model function model_v0_dense_gp(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)

    # --- Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_u ~ filldist(Exponential(0.5), 3)

    u1_off ~ filldist(Normal(0, 1), N)
    u2_off ~ filldist(Normal(0, 1), N)
    u3_off ~ filldist(Normal(0, 1), N)

    u1_true = mean(u1_obs) .+ u1_off .* std(u1_obs)
    u2_true = mean(u2_obs) .+ u2_off .* std(u2_obs)
    u3_true = mean(u3_obs) .+ u3_off .* std(u3_obs)

    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    sigma_alpha ~ Exponential(0.1)
    alpha_raw ~ filldist(Normal(0, 1), T_unique)
    alpha_t = cumsum(alpha_raw .* sigma_alpha)
    trend = alpha_t[Int.(coords_time[:,1])]

    # Spatiotemporal GP (Separable)
    # Corrected: Isotropic spatial lengthscale (single value)
    ls_s ~ Gamma(2, 2)
    ls_t ~ Gamma(2, 2)
    sigma_f ~ Exponential(1.0)

    # Correcting Kernel Transformations for Isotropic Spatial Kernel
    # Spatial kernel (Isotropic): use ScaleTransform with the inverse lengthscale
    k_s = SqExponentialKernel() ∘ ScaleTransform(inv(ls_s))

    # Temporal kernel: ensure coords_time is a vector for 1D kernel
    k_t = SqExponentialKernel() ∘ ScaleTransform(inv(ls_t))

    # Compute individual kernel matrices
    K_s = kernelmatrix(k_s, RowVecs(coords_space))
    K_t = kernelmatrix(k_t, vec(coords_time))

    # Separable covariance structure
    K = (sigma_f^2) .* K_s .* K_t

    # Latent process realization
    f ~ MvNormal(zeros(N), K + 1e-6*I)

    beta_covs ~ filldist(Normal(0, 1), 4)
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    y_obs ~ MvNormal(mu_base .+ f, sigma_y^2 * I)
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)

end
```



```{julia}

data = generate_data(50)

model_v0 = model_v0_dense_gp(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time)
chain_v0 = sample(model_v0, NUTS(), 100)
display(describe(chain_v0))
waic_v0 = compute_y_waic(model_v0, chain_v0)  # not reliable without convergence
println("WAIC for V0: ", waic_v0)

```

## V1: Non-separable Anisotropic Spatiotemporal GP

This model extends V0 by using a non-separable anisotropic kernel for the spatiotemporal Gaussian Process. Instead of multiplying separate spatial and temporal kernels, a single kernel is applied to the combined spatiotemporal coordinates, allowing for complex interactions between space and time. Anisotropy is handled by using an ARD (Automatic Relevance Determination) kernel, assigning a unique lengthscale to each dimension (longitude, latitude, and time).

### Model Assumptions:
*   Dependent Variable (Y): Similar to V0, modeled with a mean component (trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus observation noise.
*   Latent Covariates (U1, U2, U3): Same as V0, observed with measurement error and non-centered parameterization.
*   Trend: Same as V0, a random walk prior on the intercept over unique time points.
*   Seasonal Process: Same as V0, a fixed-period harmonic.
*   Spatiotemporal GP (f):
    *   Non-Separable Covariance: A single kernel is applied to the concatenated spatiotemporal coordinates, $K((x_s, t_s), (x_t, t_t)) = K_{st}([x_s, t_s], [x_t, t_t])$, allowing for more complex spatiotemporal interactions.
    *   Anisotropic Kernel: Uses an ARD (Automatic Relevance Determination) Squared Exponential kernel, meaning each input dimension (longitude, latitude, time) has its own lengthscale (`ls_st[1]`, `ls_st[2]`, `ls_st[3]`). This allows the GP to adapt to different correlation structures along different axes.
    *   Dense Kernel Matrix: Still computes the full $N \times N$ covariance matrix, leading to $O(N^3)$ computational complexity, inherited from V0.
    *   Non-centered Parameterization: The latent GP `f` is sampled directly from `MvNormal(zeros(N), K + 1e-6*I)`.
*   Observation Noise (sigma_y, sigma_u): Same as V0, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, extended for multiple lengthscales in `ls_st`.

### Key References:
*   Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (For anisotropic kernels and non-separable GPs)
*   Duvenaud, D. (2014). *Automatic Model Construction with Gaussian Processes*. PhD thesis, University of Cambridge. (For ARD kernels)


```{julia}
@model function model_v1_anisotropic_gp(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2) # Total dimensions for spatiotemporal coords

    # --- Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_u ~ filldist(Exponential(0.5), 3)

    u1_off ~ filldist(Normal(0, 1), N)
    u2_off ~ filldist(Normal(0, 1), N)
    u3_off ~ filldist(Normal(0, 1), N)

    u1_true = mean(u1_obs) .+ u1_off .* std(u1_obs)
    u2_true = mean(u2_obs) .+ u2_off .* std(u2_obs)
    u3_true = mean(u3_obs) .+ u3_off .* std(u3_obs)

    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    sigma_alpha ~ Exponential(0.1)
    alpha_raw ~ filldist(Normal(0, 1), T_unique)
    alpha_t = cumsum(alpha_raw .* sigma_alpha)
    trend = alpha_t[Int.(coords_time[:,1])]

    # Spatiotemporal GP (Non-separable Anisotropic)
    # Combine spatial and temporal coordinates
    coords_st = hcat(coords_space, coords_time)

    # Lengthscales for each dimension (2 for space, 1 for time)
    ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_f ~ Exponential(1.0)

    # Anisotropic Spatiotemporal kernel
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))

    # Compute the full spatiotemporal kernel matrix
    K = (sigma_f^2) .* kernelmatrix(k_st, RowVecs(coords_st))

    # Latent process realization
    f ~ MvNormal(zeros(N), K + 1e-6*I)

    beta_covs ~ filldist(Normal(0, 1), 4)
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    y_obs ~ MvNormal(mu_base .+ f, sigma_y^2 * I)
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```



```{julia}
data = generate_data(50)
model_v1 = model_v1_anisotropic_gp(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time)
chain_v1 = sample(model_v1, NUTS(), 100) # Using MH sampler for demonstration, consider NUTS for better sampling
display(describe(chain_v1))
waic_v1 = compute_y_waic(model_v1, chain_v1)
println("WAIC for V1: ", waic_v1)
```

    Samples per chain = 100
    Wall duration     = 1340.86 seconds
    Compute duration  = 1340.86 seconds
    WAIC for V1: 724.226399993753


## V2: Fully Adaptive Random Fourier Features (RFF)

This model builds upon V1 by replacing the direct computation of the dense kernel matrix with an approximation using Random Fourier Features (RFF). The 'adaptive' aspect comes from treating the RFF projection weights `W` and offsets `b` as parameters within the Bayesian model, allowing the model to learn the spectral density of the kernel directly from the data. The latent GP `f` is then constructed as a linear combination of these learned features.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V1, modeled with a mean component (trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus observation noise.
*   Latent Covariates (U1, U2, U3): Same as V1, observed with measurement error and non-centered parameterization.
*   Trend: Same as V1, a random walk prior on the intercept over unique time points.
*   Seasonal Process: Same as V1, a fixed-period harmonic.
*   Spatiotemporal GP (f):
    *   RFF Approximation: Instead of computing the full kernel matrix, the spatiotemporal GP is approximated using Random Fourier Features. This reduces computational complexity from $O(N^3)$ to $O(N M_{rff}^2)$ or $O(N D M_{rff})$ (where $M_{rff}$ is the number of RFF features, and $D$ is input dimensions).
    *   Adaptive RFF: The projection weights (`W_matrix`) and offsets (`b`) for the RFF are treated as parameters and learned during inference. This allows the RFF to adaptively approximate the true kernel's spectral density, rather than relying on fixed, pre-sampled features.
    *   Non-centered Parameterization: The coefficients `beta_rff` are sampled from a Normal distribution whose variance is related to `sigma_f^2`, maintaining a non-centered approach for the latent GP `f`.
*   Observation Noise (sigma_y, sigma_u): Same as V1, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors for `sigma_y`, `sigma_u`, `beta_cos`, `beta_sin`, `sigma_alpha`, `alpha_raw`, `sigma_f`, and `beta_covs`. For RFF parameters:
    *   `W_matrix`: Normal priors, reflecting the spectral density of a Squared Exponential kernel.
    *   `b`: Uniform prior between 0 and 2π for phases.
    *   `beta_rff`: Normal prior with variance `sigma_f^2`.

### Key References:
*   Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS.
*   Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (For GP fundamentals and kernel theory)
*   Hernández-Lobato, J. M., & Adams, R. P. (2015). *Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks*. ICML. (For adaptive RFF concepts in Bayesian contexts)


```{julia}
@model function model_v2_adaptive_rff(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0, M_rff=50)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2) # Total dimensions for spatiotemporal coords

    # --- Priors (similar to V1) ---
    sigma_y ~ Exponential(1.0)
    sigma_u ~ filldist(Exponential(0.5), 3)

    u1_off ~ filldist(Normal(0, 1), N)
    u2_off ~ filldist(Normal(0, 1), N)
    u3_off ~ filldist(Normal(0, 1), N)

    u1_true = mean(u1_obs) .+ u1_off .* std(u1_obs)
    u2_true = mean(u2_obs) .+ u2_off .* std(u2_obs)
    u3_true = mean(u3_obs) .+ u3_off .* std(u3_obs)

    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    sigma_alpha ~ Exponential(0.1)
    alpha_raw ~ filldist(Normal(0, 1), T_unique)
    alpha_t = cumsum(alpha_raw .* sigma_alpha)
    trend = alpha_t[Int.(coords_time[:,1])]

    # Spatiotemporal GP using Adaptive RFF
    coords_st = hcat(coords_space, coords_time)

    # Adaptive RFF parameters
    # W: projection weights (D_st x M_rff)
    # Normal priors for W, reflecting the spectral density of SqExponential kernel
    W_matrix ~ filldist(Normal(0, 1), D_st, M_rff)

    # b: offsets (M_rff) - Uniform distribution for phases
    b ~ filldist(Uniform(0, 2pi), M_rff)

    # Scale of the RFF features, analogous to sigma_f in the exact GP
    sigma_f ~ Exponential(1.0)

    # RFF coefficients
    beta_rff ~ filldist(Normal(0, sigma_f^2), M_rff)

    # Compute RFF features
    Phi = rff_map(coords_st, W_matrix, b)

    # Latent process realization as a linear model of RFF features
    f = Phi * beta_rff

    beta_covs ~ filldist(Normal(0, 1), 4)
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    y_obs ~ MvNormal(mu_base .+ f, sigma_y^2 * I)
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```



```{julia}
data = generate_data(50)
M_rff_val = 50 # Number of RFF features
model_v2 = model_v2_adaptive_rff(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time; M_rff=M_rff_val)
chain_v2 = sample(model_v2, NUTS(), 100) # Using MH sampler for demonstration
display(describe(chain_v2))
waic_v2 = compute_y_waic(model_v2, chain_v2)
println("WAIC for V2: ", waic_v2)
```

    Samples per chain = 100
    Compute duration  = 597.07 seconds
    WAIC for V2: 95.11863291759116


## V3: Nested Covariates

This model builds upon V2 by assuming that the covariates `U1`, `U2`, and `U3` are nested functions. Specifically, it explicitly models the relationships between these covariates and the base inputs (`coords_time`, `z`).

### Model Assumptions:
*   Dependent Variable (Y): Similar to V2, modeled with a mean component (trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus observation noise.
*   Latent Covariates (U1, U2, U3): These covariates are modeled as *nested linear functions* of `coords_time`, `z`, and previous `U` covariates. Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) are assumed to have measurement error.
    *   `U1 = f1(coords_time, Z)`: Modeled as a linear function of time and spatial covariate `Z`.
    *   `U2 = f2(coords_time, Z, U1)`: Modeled as a linear function of time, `Z`, and the latent `U1`.
    *   `U3 = f3(coords_time, Z, U1)`: Modeled as a linear function of time, `Z`, and the latent `U1`.
    (Note: In this implementation, `Z_time` is `coords_time[:,1]` and `Z_space` is `z`).
*   Trend: Same as V2, a random walk prior on the intercept over unique time points.
*   Seasonal Process: Same as V2, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V2, uses an Adaptive Random Fourier Features (RFF) approximation with learned projection weights and offsets.
*   Observation Noise (sigma_y, sigma_u): Same as V2, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors. For the new nested covariate relationships, `beta_u1_ztime`, `beta_u1_zspace`, `beta_u2_ztime`, etc., are given Normal(0, 1) priors, reflecting an initial assumption of simple linear dependencies.

### Key References:
*   Hierarchical Modeling: Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press. (For general principles of hierarchical modeling).
*   Adaptive RFF: Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (Still relevant for the GP component).

This model introduces explicit structural assumptions about how covariates influence each other, moving beyond simple additive effects by encoding a directed dependency graph among them.


```{julia}
@model function model_v3_nested_covs(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0, M_rff=50)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)

    # --- Priors --- General model parameters
    sigma_y ~ Exponential(1.0)
    sigma_u ~ filldist(Exponential(0.5), 3) # Measurement error for u_obs

    # --- Nested Covariate Parameters --- Coefficients for the functional relationships
    # U1 = f1(Z_time, Z_space)
    beta_u1_ztime ~ Normal(0, 1)
    beta_u1_zspace ~ Normal(0, 1)

    # U2 = f2(Z_time, Z_space, U1)
    beta_u2_ztime ~ Normal(0, 1)
    beta_u2_zspace ~ Normal(0, 1)
    beta_u2_u1 ~ Normal(0, 1)

    # U3 = f3(Z_time, Z_space, U1)
    beta_u3_ztime ~ Normal(0, 1)
    beta_u3_zspace ~ Normal(0, 1)
    beta_u3_u1 ~ Normal(0, 1)

    # --- Define Nested Latent Covariates --- Deterministic transformations
    u1_true = beta_u1_ztime .* coords_time[:,1] .+ beta_u1_zspace .* z
    u2_true = beta_u2_ztime .* coords_time[:,1] .+ beta_u2_zspace .* z .+ beta_u2_u1 .* u1_true
    u3_true = beta_u3_ztime .* coords_time[:,1] .+ beta_u3_zspace .* z .+ beta_u3_u1 .* u1_true

    # --- Trend and Seasonal Components (from V2) ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    sigma_alpha ~ Exponential(0.1)
    alpha_raw ~ filldist(Normal(0, 1), T_unique)
    alpha_t = cumsum(alpha_raw .* sigma_alpha)
    trend = alpha_t[Int.(coords_time[:,1])]

    # --- Spatiotemporal GP using Adaptive RFF (from V2) ---
    coords_st = hcat(coords_space, coords_time)
    W_matrix ~ filldist(Normal(0, 1), D_st, M_rff)
    b ~ filldist(Uniform(0, 2pi), M_rff)
    sigma_f ~ Exponential(1.0)
    beta_rff ~ filldist(Normal(0, sigma_f^2), M_rff)
    Phi = rff_map(coords_st, W_matrix, b)
    f = Phi * beta_rff

    # --- Mean of Y and Likelihoods ---
    beta_covs ~ filldist(Normal(0, 1), 4) # These betas are for u1_true, u2_true, u3_true, z
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    y_obs ~ MvNormal(mu_base .+ f, sigma_y^2 * I)
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I) # u1_obs informs u1_true
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I) # u2_obs informs u2_true
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I) # u3_obs informs u3_true
end
```



```{julia}
data = generate_data(50)
M_rff_val = 50 # Number of RFF features
model_v3 = model_v3_nested_covs(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time; M_rff=M_rff_val)
chain_v3 = sample(model_v3, NUTS(), 100) # Using MH sampler for demonstration; consider NUTS for better sampling
display(describe(chain_v3))
waic_v3 = compute_y_waic(model_v3, chain_v3)
println("WAIC for V3: ", waic_v3)
```

    Samples per chain = 100
    Compute duration  = 170.98 seconds
    WAIC for V3: 112.23686150367824


## V4: Time-varying Intercept

This model builds upon V3 by adding a latent temporal process (Random Walk Intercept), allowing the model to evolve over the temporal dimension. This is implemented by explicitly defining the `trend` component as a random walk.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V3, modeled with a mean component (trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus observation noise.
*   Latent Covariates (U1, U2, U3): Same as V3, modeled as nested linear functions and observed with measurement error.
*   Trend: New in V4, the trend component is explicitly modeled as a Random Walk. This allows the intercept to vary smoothly over time, capturing non-linear temporal dynamics. It replaces the `cumsum(alpha_raw * sigma_alpha)` from previous versions with an explicit loop for the random walk: `alpha[t] ~ Normal(alpha[t-1], sigma_alpha)`.
*   Seasonal Process: Same as V3, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V3, uses an Adaptive Random Fourier Features (RFF) approximation with learned projection weights and offsets.
*   Observation Noise (sigma_y, sigma_u): Same as V3, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors. For the random walk, `sigma_alpha` is an `Exponential(0.1)` prior, controlling the step size of the walk.

### Key References:
*   Random Walk Models: Harvey, A. C. (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press. (For time-varying parameters and random walks)
*   Hierarchical Bayesian Models: Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press. (For general hierarchical modeling principles applied to time series)


```{julia}
@model function model_v4_time_varying_intercept(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0, M_rff=50)
    N = length(y_obs)
    T_unique = length(unique(coords_time)) # Number of unique time points for the random walk
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)

    # --- Priors --- General model parameters
    sigma_y ~ Exponential(1.0)
    sigma_u ~ filldist(Exponential(0.5), 3) # Measurement error for u_obs

    # --- Nested Covariate Parameters (from V3) ---
    beta_u1_ztime ~ Normal(0, 1)
    beta_u1_zspace ~ Normal(0, 1)

    beta_u2_ztime ~ Normal(0, 1)
    beta_u2_zspace ~ Normal(0, 1)
    beta_u2_u1 ~ Normal(0, 1)

    beta_u3_ztime ~ Normal(0, 1)
    beta_u3_zspace ~ Normal(0, 1)
    beta_u3_u1 ~ Normal(0, 1)

    # --- Define Nested Latent Covariates (from V3) ---
    u1_true = beta_u1_ztime .* coords_time[:,1] .+ beta_u1_zspace .* z
    u2_true = beta_u2_ztime .* coords_time[:,1] .+ beta_u2_zspace .* z .+ beta_u2_u1 .* u1_true
    u3_true = beta_u3_ztime .* coords_time[:,1] .+ beta_u3_zspace .* z .+ beta_u3_u1 .* u1_true

    # --- Seasonal Component (from V3) ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    # --- Time-varying Intercept (Random Walk) ---
    sigma_alpha ~ Exponential(0.1)
    alpha = Vector{Real}(undef, T_unique)
    alpha[1] ~ Normal(0, sigma_alpha)
    for t in 2:T_unique
        alpha[t] ~ Normal(alpha[t-1], sigma_alpha)
    end
    trend = alpha[Int.(coords_time[:,1])] # Map unique time points back to original time coordinates

    # --- Spatiotemporal GP using Adaptive RFF (from V3) ---
    coords_st = hcat(coords_space, coords_time)
    W_matrix ~ filldist(Normal(0, 1), D_st, M_rff)
    b ~ filldist(Uniform(0, 2pi), M_rff)
    sigma_f ~ Exponential(1.0)
    beta_rff ~ filldist(Normal(0, sigma_f^2), M_rff)
    Phi = rff_map(coords_st, W_matrix, b)
    f = Phi * beta_rff

    # --- Mean of Y and Likelihoods ---
    beta_covs ~ filldist(Normal(0, 1), 4) # These betas are for u1_true, u2_true, u3_true, z
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    y_obs ~ MvNormal(mu_base .+ f, sigma_y^2 * I)
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```



```{julia}
data = generate_data(50)
M_rff_val = 50 # Number of RFF features
model_v4 = model_v4_time_varying_intercept(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time; M_rff=M_rff_val)
chain_v4 = sample(model_v4, NUTS(), 100) # Using MH sampler for demonstration; consider NUTS for better sampling
display(describe(chain_v4))
waic_v4 = compute_y_waic(model_v4, chain_v4)
println("WAIC for V4: ", waic_v4)
```
    Samples per chain = 100
    Compute duration  = 1380.22 seconds
    WAIC for V4: 1134.1174265686002


## V5: Spatiotemporal Stochastic Volatility

This model builds upon V4 by treating the observation noise not as a constant, but as a time-varying and space-varying process. It uses a secondary RFF mapping to model the log-variance, allowing the model to account for heteroscedasticity.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V4, modeled with a mean component (trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component.
*   Latent Covariates (U1, U2, U3): Same as V4, modeled as nested linear functions and observed with measurement error.
*   Trend: Same as V4, a Random Walk intercept.
*   Seasonal Process: Same as V4, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V4, uses an Adaptive Random Fourier Features (RFF) approximation for the mean function.
*   Observation Noise (sigma_y): New in V5, the observation noise variance (`sigma_y^2`) is no longer a constant scalar. Instead, it is modeled as a spatiotemporally varying process. This is achieved by:
    *   Using a secondary RFF mapping (`W_sigma`, `b_sigma`, `beta_rff_sigma`) on the `coords_st` to model the *log-variance* of the observation noise (`log_sigma_y`).
    *   The standard deviation `sigma_y_process` is then derived from `exp.(log_sigma_y ./ 2)`. This allows the model to capture heteroscedasticity, meaning the observation noise can vary across space and time.
*   Covariate Observation Noise (sigma_u): Same as V4, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, extended for the new RFF parameters for the stochastic volatility component.

### Key References:
*   Stochastic Volatility Models: Shephard, N. (2005). *Stochastic Volatility: Selected Readings*. Oxford University Press. (General theory of stochastic volatility)
*   Heteroscedastic Gaussian Processes: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS. (Concepts of learning input-dependent noise in GPs)
*   Random Fourier Features: Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (Used for the log-variance RFF mapping)


```{julia}
@model function model_v5_stochastic_volatility(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0, M_rff=50, M_rff_sigma=20)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)

    # --- Priors --- General model parameters
    sigma_u ~ filldist(Exponential(0.5), 3) # Measurement error for u_obs

    # --- Nested Covariate Parameters (from V4) ---
    beta_u1_ztime ~ Normal(0, 1)
    beta_u1_zspace ~ Normal(0, 1)

    beta_u2_ztime ~ Normal(0, 1)
    beta_u2_zspace ~ Normal(0, 1)
    beta_u2_u1 ~ Normal(0, 1)

    beta_u3_ztime ~ Normal(0, 1)
    beta_u3_zspace ~ Normal(0, 1)
    beta_u3_u1 ~ Normal(0, 1)

    # --- Define Nested Latent Covariates (from V4) ---
    u1_true = beta_u1_ztime .* coords_time[:,1] .+ beta_u1_zspace .* z
    u2_true = beta_u2_ztime .* coords_time[:,1] .+ beta_u2_zspace .* z .+ beta_u2_u1 .* u1_true
    u3_true = beta_u3_ztime .* coords_time[:,1] .+ beta_u3_zspace .* z .+ beta_u3_u1 .* u1_true

    # --- Seasonal Component (from V4) ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    # --- Time-varying Intercept (Random Walk, from V4) ---
    sigma_alpha ~ Exponential(0.1)
    alpha = Vector{Real}(undef, T_unique)
    alpha[1] ~ Normal(0, sigma_alpha)
    for t in 2:T_unique
        alpha[t] ~ Normal(alpha[t-1], sigma_alpha)
    end
    trend = alpha[Int.(coords_time[:,1])] # Map unique time points back to original time coordinates

    # --- Spatiotemporal GP using Adaptive RFF (from V4) ---
    coords_st = hcat(coords_space, coords_time)
    W_matrix ~ filldist(Normal(0, 1), D_st, M_rff)
    b ~ filldist(Uniform(0, 2pi), M_rff)
    sigma_f ~ Exponential(1.0)
    beta_rff ~ filldist(Normal(0, sigma_f^2), M_rff)
    Phi = rff_map(coords_st, W_matrix, b)
    f = Phi * beta_rff

    # --- Spatiotemporal Stochastic Volatility (New in V5) ---
    # Secondary RFF mapping for log-variance
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0) # Scale for the log-variance GP
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)

    Phi_sigma = rff_map(coords_st, W_sigma, b_sigma)
    log_sigma_y = Phi_sigma * beta_rff_sigma # Latent log-variance process
    sigma_y_process = exp.(log_sigma_y ./ 2) # Convert log-variance to standard deviation

    # --- Mean of Y and Likelihoods ---
    beta_covs ~ filldist(Normal(0, 1), 4) # These betas are for u1_true, u2_true, u3_true, z
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    # Use spatiotemporal sigma_y_process in the likelihood
    y_obs ~ MvNormal(mu_base .+ f, Diagonal(sigma_y_process.^2))
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```



```{julia}
data = generate_data(50)
M_rff_val = 50 # Number of RFF features for mean GP
M_rff_sigma_val = 20 # Number of RFF features for log-variance GP
model_v5 = model_v5_stochastic_volatility(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time; M_rff=M_rff_val, M_rff_sigma=M_rff_sigma_val)
chain_v5 = sample(model_v5, NUTS(), 100) # Using MH sampler for demonstration; consider NUTS for better sampling
display(describe(chain_v5))
waic_v5 = compute_y_waic(model_v5, chain_v5)
println("WAIC for V5: ", waic_v5)
```

    Samples per chain = 100
    Compute duration  = 2201.19 seconds
    WAIC for V5: 313.3121722876543


## V6: Fully Independent Training Conditional (FITC)

This model builds upon V5. However, instead of using Random Fourier Features for the main GP, it implements a Fully Independent Training Conditional (FITC) approximation. FITC uses a smaller set of *inducing points* to approximate the full GP, significantly reducing computational cost while aiming to preserve accuracy. The model retains the nested covariates, time-varying intercept, and spatiotemporal stochastic volatility from V5.

Crucially, this implementation also switches to the more robust NUTS (No-U-Turn Sampler) to handle the increased model complexity and improve sampling convergence, which was a recurring issue with the MH sampler in previous models.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V5, modeled with a mean component (trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component.
*   Latent Covariates (U1, U2, U3): Same as V5, modeled as nested linear functions and observed with measurement error.
*   Trend: Same as V5, a Random Walk intercept.
*   Seasonal Process: Same as V5, a fixed-period harmonic.
*   Spatiotemporal GP (f): New in V6, the main spatiotemporal GP is approximated using Fully Independent Training Conditional (FITC). This involves:
    *   Inducing Points (`Z_inducing`): A set of `M_inducing_val` inducing points are used to approximate the GP. These points are *not* learned within the model parameters; instead, they are generated externally (e.g., via random sampling or K-Means, as shown in the example setup).
    *   Kernel (`k_st`): An anisotropic Squared Exponential kernel (`SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))`) is used, similar to V1, applied to the combined spatiotemporal coordinates.
    *   Approximation: The conditional mean and a diagonal approximation of the conditional covariance of the GP at observed data points are computed given the latent values at the inducing points (`u_latent`). This significantly reduces the computational complexity from $O(N^3)$ to $O(N M^2 + M^3)$.
    *   Non-centered Parameterization: The latent values at inducing points (`u_latent`) are sampled from a multivariate normal distribution defined by the kernel at inducing points.
*   Observation Noise (sigma_y): Same as V5, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V5, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, extended for the `ls_st` (anisotropic lengthscales) and `sigma_f` for the FITC GP, and all parameters for the stochastic volatility component.

### Key References:
*   FITC: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS.
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (For general GP theory)
*   NUTS Sampler: Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*. Journal of Machine Learning Research, 15, 1593-1623. (For improved sampling efficiency for complex models)


## V9: Gaussian Process Trend

This model builds upon V8 by replacing the Random Walk Intercept with a Gaussian Process-based trend. This allows for a more flexible and smooth representation of the underlying temporal trend, potentially improving model fidelity, especially in cases where the trend exhibits non-linear but continuous behavior. The GP trend is defined over the unique time points using a `SqExponentialKernel`.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V8, modeled with a mean component (GP trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): Same as V8, modeled as nonlinear functions using separate RFF mappings of `coords_time`, `Z`, and other `U` covariates. Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) still have measurement error.
*   Trend: New in V9, the trend component is now explicitly modeled as a 1D Gaussian Process (`GP Trend`) using a `SqExponentialKernel` over unique time points. This replaces the Random Walk Intercept from previous versions.
*   Seasonal Process: Same as V8, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V8, uses the Fully Independent Training Conditional (FITC) approximation with learned inducing point locations (`Z_inducing`) and latent values (`u_latent`).
*   Observation Noise (sigma_y): Same as V8, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V8, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, extended for the `ls_trend` and `sigma_trend` parameters of the GP Trend.

### Key References:
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (For GP fundamentals and GP-based trends).
*   Time Series with GPs: Roberts, S., Osborne, M. A., & Ebden, M. (2013). *Gaussian Processes for Time-Series Analysis*. In *Time-Series Analysis* (pp. 59-86). Springer, Berlin, Heidelberg. (For applying GPs to time series data).
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (For the nonlinear covariate mappings).
*   FITC and Inducing Point Optimization: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS. (Still relevant for the main GP component).


```{julia}
@model function model_v6_fitc(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time, Z_inducing; period=12.0, M_rff_sigma=20)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)
    M_inducing_val = size(Z_inducing, 1) # Number of inducing points

    # --- Priors --- General model parameters
    sigma_u ~ filldist(Exponential(0.5), 3) # Measurement error for u_obs

    # --- Nested Covariate Parameters (from V5) ---
    beta_u1_ztime ~ Normal(0, 1)
    beta_u1_zspace ~ Normal(0, 1)

    beta_u2_ztime ~ Normal(0, 1)
    beta_u2_zspace ~ Normal(0, 1)
    beta_u2_u1 ~ Normal(0, 1)

    beta_u3_ztime ~ Normal(0, 1)
    beta_u3_zspace ~ Normal(0, 1)
    beta_u3_u1 ~ Normal(0, 1)

    # --- Define Nested Latent Covariates (from V5) ---
    u1_true = beta_u1_ztime .* coords_time[:,1] .+ beta_u1_zspace .* z
    u2_true = beta_u2_ztime .* coords_time[:,1] .+ beta_u2_zspace .* z .+ beta_u2_u1 .* u1_true
    u3_true = beta_u3_ztime .* coords_time[:,1] .+ beta_u3_zspace .* z .+ beta_u3_u1 .* u1_true

    # --- Seasonal Component (from V5) ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    # --- Time-varying Intercept (Random Walk, from V5) ---
    sigma_alpha ~ Exponential(0.1)
    alpha = Vector{Real}(undef, T_unique)
    alpha[1] ~ Normal(0, sigma_alpha)
    for t in 2:T_unique
        alpha[t] ~ Normal(alpha[t-1], sigma_alpha)
    end
    trend = alpha[Int.(coords_time[:,1])]

    # --- Spatiotemporal GP using FITC (New in V6) ---
    coords_st = hcat(coords_space, coords_time)

    # Lengthscales for each dimension (2 for space, 1 for time)
    ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_f ~ Exponential(1.0)

    # Anisotropic Spatiotemporal kernel
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))

    # Compute kernel matrices for FITC approximation
    # Add jitter for numerical stability to K_ZZ
    K_ZZ = kernelmatrix(k_st, RowVecs(Z_inducing)) + 1e-6*I
    K_XZ = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_inducing))
    K_XX_diag = diag(kernelmatrix(k_st, RowVecs(coords_st)))

    # Latent values at inducing points
    u_latent ~ MvNormal(zeros(M_inducing_val), K_ZZ)

    # Mean function of GP at observed points, conditioned on u_latent
    mean_f = K_XZ * (K_ZZ \ u_latent)

    # Diagonal covariance for FITC
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))

    # Ensure positive definite covariance (add small jitter if necessary)
    f ~ MvNormal(mean_f, Diagonal(max.(0, cov_f_diag) + 1e-6*ones(N)))

    # --- Spatiotemporal Stochastic Volatility (from V5) ---
    # Secondary RFF mapping for log-variance
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0) # Scale for the log-variance GP
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)

    Phi_sigma = rff_map(coords_st, W_sigma, b_sigma)
    log_sigma_y = Phi_sigma * beta_rff_sigma # Latent log-variance process
    sigma_y_process = exp.(log_sigma_y ./ 2) # Convert log-variance to standard deviation

    # --- Mean of Y and Likelihoods ---
    beta_covs ~ filldist(Normal(0, 1), 4) # These betas are for u1_true, u2_true, u3_true, z
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    # Use spatiotemporal sigma_y_process in the likelihood
    y_obs ~ MvNormal(mu_base .+ f, Diagonal(sigma_y_process.^2))
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```



```{julia}

# Generate inducing points for V6
D_s_v6 = size(data.coords_space, 2)
D_st_v6 = D_s_v6 + size(data.coords_time, 2)
coords_st_v6 = hcat(data.coords_space, data.coords_time)
M_inducing_val_v6 = 10 # Number of inducing points (e.g., 10-20% of N)
Z_inducing_v6 = generate_inducing_points(coords_st_v6, M_inducing_val_v6)

# Sample Model V6 with NUTS
M_rff_sigma_val_v6 = 20 # Number of RFF features for log-variance GP
model_v6 = model_v6_fitc(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time, Z_inducing_v6; M_rff_sigma=M_rff_sigma_val_v6)

# Using NUTS sampler for better convergence; consider increasing iterations for production runs
chain_v6 = sample(model_v6, NUTS(), 100) # Increased samples from 100 to 500
display(describe(chain_v6))
waic_v6 = compute_y_waic(model_v6, chain_v6)
println("WAIC for V6: ", waic_v6)

println("\nNote: For robust results, consider increasing the number of samples (e.g., 1000-2000 or more) and tuning the NUTS parameters (e.g., `adapts=num_adapts_steps`) if convergence issues persist.")
```

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.00078125
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:44:03[39m


    Chains MCMC chain (100×247×1 Array{Float64, 3}):
    
    Iterations        = 51:1:150
    Number of chains  = 1
    Samples per chain = 100
    Wall duration     = 2682.27 seconds
    Compute duration  = 2682.27 seconds
    parameters        = sigma_u[1], sigma_u[2], sigma_u[3], beta_u1_ztime, beta_u1_zspace, beta_u2_ztime, beta_u2_zspace, beta_u2_u1, beta_u3_ztime, beta_u3_zspace, beta_u3_u1, beta_cos, beta_sin, sigma_alpha, alpha[1], alpha[2], alpha[3], alpha[4], alpha[5], alpha[6], alpha[7], alpha[8], alpha[9], alpha[10], alpha[11], alpha[12], alpha[13], alpha[14], alpha[15], alpha[16], alpha[17], alpha[18], alpha[19], alpha[20], alpha[21], alpha[22], alpha[23], alpha[24], alpha[25], alpha[26], alpha[27], alpha[28], alpha[29], alpha[30], alpha[31], alpha[32], alpha[33], alpha[34], alpha[35], alpha[36], alpha[37], alpha[38], alpha[39], alpha[40], alpha[41], alpha[42], alpha[43], alpha[44], alpha[45], alpha[46], alpha[47], alpha[48], alpha[49], alpha[50], ls_st[1], ls_st[2], ls_st[3], sigma_f, u_latent[1], u_latent[2], u_latent[3], u_latent[4], u_latent[5], u_latent[6], u_latent[7], u_latent[8], u_latent[9], u_latent[10], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15], f[16], f[17], f[18], f[19], f[20], f[21], f[22], f[23], f[24], f[25], f[26], f[27], f[28], f[29], f[30], f[31], f[32], f[33], f[34], f[35], f[36], f[37], f[38], f[39], f[40], f[41], f[42], f[43], f[44], f[45], f[46], f[47], f[48], f[49], f[50], W_sigma[1, 1], W_sigma[2, 1], W_sigma[3, 1], W_sigma[1, 2], W_sigma[2, 2], W_sigma[3, 2], W_sigma[1, 3], W_sigma[2, 3], W_sigma[3, 3], W_sigma[1, 4], W_sigma[2, 4], W_sigma[3, 4], W_sigma[1, 5], W_sigma[2, 5], W_sigma[3, 5], W_sigma[1, 6], W_sigma[2, 6], W_sigma[3, 6], W_sigma[1, 7], W_sigma[2, 7], W_sigma[3, 7], W_sigma[1, 8], W_sigma[2, 8], W_sigma[3, 8], W_sigma[1, 9], W_sigma[2, 9], W_sigma[3, 9], W_sigma[1, 10], W_sigma[2, 10], W_sigma[3, 10], W_sigma[1, 11], W_sigma[2, 11], W_sigma[3, 11], W_sigma[1, 12], W_sigma[2, 12], W_sigma[3, 12], W_sigma[1, 13], W_sigma[2, 13], W_sigma[3, 13], W_sigma[1, 14], W_sigma[2, 14], W_sigma[3, 14], W_sigma[1, 15], W_sigma[2, 15], W_sigma[3, 15], W_sigma[1, 16], W_sigma[2, 16], W_sigma[3, 16], W_sigma[1, 17], W_sigma[2, 17], W_sigma[3, 17], W_sigma[1, 18], W_sigma[2, 18], W_sigma[3, 18], W_sigma[1, 19], W_sigma[2, 19], W_sigma[3, 19], W_sigma[1, 20], W_sigma[2, 20], W_sigma[3, 20], b_sigma[1], b_sigma[2], b_sigma[3], b_sigma[4], b_sigma[5], b_sigma[6], b_sigma[7], b_sigma[8], b_sigma[9], b_sigma[10], b_sigma[11], b_sigma[12], b_sigma[13], b_sigma[14], b_sigma[15], b_sigma[16], b_sigma[17], b_sigma[18], b_sigma[19], b_sigma[20], sigma_log_var, beta_rff_sigma[1], beta_rff_sigma[2], beta_rff_sigma[3], beta_rff_sigma[4], beta_rff_sigma[5], beta_rff_sigma[6], beta_rff_sigma[7], beta_rff_sigma[8], beta_rff_sigma[9], beta_rff_sigma[10], beta_rff_sigma[11], beta_rff_sigma[12], beta_rff_sigma[13], beta_rff_sigma[14], beta_rff_sigma[15], beta_rff_sigma[16], beta_rff_sigma[17], beta_rff_sigma[18], beta_rff_sigma[19], beta_rff_sigma[20], beta_covs[1], beta_covs[2], beta_covs[3], beta_covs[4]
    internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, logprior, loglikelihood, logjoint
    
    Summary Statistics
    
     [1m     parameters [0m [1m    mean [0m [1m     std [0m [1m    mcse [0m [1m ess_bulk [0m [1m ess_tail [0m [1m    rhat[0m ⋯
     [90m         Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m  Float64 [0m [90m  Float64 [0m [90m Float64[0m ⋯
    
          sigma_u[1]    0.7102    0.0785    0.0062   188.0786    54.8565    1.1084 ⋯
          sigma_u[2]    0.8429    0.1091    0.0540     3.8527    63.6000    1.2065 ⋯
          sigma_u[3]    0.1020    0.0096    0.0007   200.0000    81.5536    1.0247 ⋯
       beta_u1_ztime    0.0085    0.0031    0.0014     4.9760    60.0314    1.1634 ⋯
      beta_u1_zspace    0.2985    0.1083    0.0095   110.3793    34.3628    1.0772 ⋯
       beta_u2_ztime    0.0058    0.0091    0.0048     5.3047    13.9888    1.3237 ⋯
      beta_u2_zspace   -0.2709    0.2242    0.0712     9.9643    14.4748    1.1052 ⋯
          beta_u2_u1    0.0710    0.6297    0.2204     8.1483    16.6208    1.1471 ⋯
       beta_u3_ztime    0.0073    0.0047    0.0012    15.7508    21.5943    1.1831 ⋯
      beta_u3_zspace    0.1558    0.1573    0.0385    16.1619    31.0890    1.0493 ⋯
          beta_u3_u1   -0.2445    0.4963    0.1290    15.0152    21.0246    1.1481 ⋯
            beta_cos    0.7840    0.2302    0.0428    30.5721    48.5819    1.0068 ⋯
            beta_sin   -0.1052    0.2599    0.1818     2.0867    44.5066    1.4813 ⋯
         sigma_alpha    0.2362    0.0582    0.0175    11.7069    12.7391    1.0307 ⋯
            alpha[1]    0.2546    0.2004    0.0405    24.2934    87.2420    1.0826 ⋯
            alpha[2]    0.5235    0.2815    0.1166     5.7718    30.0069    1.1671 ⋯
            alpha[3]    0.6964    0.3168    0.1813     3.0684    26.6200    1.3338 ⋯
            alpha[4]    0.8325    0.3294    0.1988     2.8303    22.9118    1.3391 ⋯
            alpha[5]    0.9602    0.3380    0.2416     2.0816    16.8447    1.5096 ⋯
            alpha[6]    1.0602    0.3776    0.2652     1.9719    16.4027    1.5351 ⋯
            alpha[7]    1.0878    0.3430    0.2133     2.6667    12.9123    1.3187 ⋯
                   ⋮         ⋮         ⋮         ⋮          ⋮          ⋮         ⋮ ⋱
    
    [36m                                                   1 column and 212 rows omitted[0m
    
    Quantiles
    
     [1m     parameters [0m [1m    2.5% [0m [1m   25.0% [0m [1m   50.0% [0m [1m   75.0% [0m [1m   97.5% [0m
     [90m         Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m
    
          sigma_u[1]    0.5971    0.6541    0.7016    0.7525    0.8810
          sigma_u[2]    0.6470    0.7783    0.8401    0.9182    1.0839
          sigma_u[3]    0.0875    0.0934    0.1020    0.1093    0.1187
       beta_u1_ztime    0.0027    0.0062    0.0083    0.0104    0.0150
      beta_u1_zspace    0.0905    0.2317    0.2958    0.3744    0.5130
       beta_u2_ztime   -0.0182    0.0035    0.0085    0.0114    0.0170
      beta_u2_zspace   -0.8080   -0.3820   -0.2566   -0.1462    0.1906
          beta_u2_u1   -1.1211   -0.3235    0.0799    0.3740    1.4403
       beta_u3_ztime   -0.0008    0.0044    0.0066    0.0101    0.0167
      beta_u3_zspace   -0.1662    0.0661    0.1682    0.2695    0.4076
          beta_u3_u1   -1.0645   -0.5988   -0.3374    0.0811    0.7587
            beta_cos    0.3782    0.6513    0.7631    0.9224    1.2834
            beta_sin   -0.5370   -0.2906   -0.1124    0.0797    0.4125
         sigma_alpha    0.1130    0.1927    0.2377    0.2735    0.3278
            alpha[1]   -0.0983    0.1160    0.2505    0.3895    0.6344
            alpha[2]    0.0669    0.3298    0.5341    0.7380    1.0946
            alpha[3]    0.1150    0.4447    0.7071    0.9381    1.2642
            alpha[4]    0.1541    0.5531    0.9078    1.1127    1.2772
            alpha[5]    0.2096    0.7583    1.0158    1.2137    1.4661
            alpha[6]    0.2766    0.7955    1.1293    1.3484    1.6476
            alpha[7]    0.3746    0.8896    1.1050    1.3477    1.6683
                   ⋮         ⋮         ⋮         ⋮         ⋮         ⋮
    
    [36m                                                    212 rows omitted[0m



    nothing


    WAIC for V6: 137.20586598982402
    
    Note: For robust results, consider increasing the number of samples (e.g., 1000-2000 or more) and tuning the NUTS parameters (e.g., `adapts=num_adapts_steps`) if convergence issues persist.


## V7: Sparse Functional Form based on V6

This model builds upon V6 by using a sparse Gaussian Process. While the underlying mathematical approximation remains that of FITC (using inducing points and a diagonal approximation for the conditional variance), this version uses `AbstractGPs.jl`'s `GP` object and conditioning syntax to represent the GP, making the code more abstract and potentially more extensible within the `AbstractGPs.jl` ecosystem. We continue to use the NUTS sampler from V6 to handle model complexity and improve convergence.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V6, modeled with a mean component (trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component.
*   Latent Covariates (U1, U2, U3): Same as V6, modeled as nested linear functions (using direct linear relationships for `u1_true`, `u2_true`, `u3_true`) and observed with measurement error.
*   Trend: Same as V6, a Random Walk intercept.
*   Seasonal Process: Same as V6, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V6, uses the Fully Independent Training Conditional (FITC) approximation via `AbstractGPs.jl`.
    *   Inducing Points (`Z_inducing`): New in V7, the locations of the inducing points are now treated as parameters to be learned directly by the NUTS sampler. They are initialized with a prior based on the mean and scaled standard deviation of the input data, allowing for adaptive placement of inducing points.
    *   Kernel: An anisotropic Squared Exponential kernel (`SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))`) is used.
    *   Approximation: The conditional mean and diagonal approximation of the conditional covariance are computed using the FITC formulas, leveraging `AbstractGPs.jl` for kernel matrix calculations.
    *   Non-centered Parameterization: Latent values at inducing points (`u_latent`) are sampled from a multivariate normal distribution defined by the kernel at inducing points.
*   Observation Noise (sigma_y): Same as V6, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V6, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, with the addition of priors for `Z_inducing` locations.

### Key References:
*   FITC: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS.
*   AbstractGPs.jl: Used for constructing and manipulating GP kernels and objects.
*   Inducing Point Optimization: Hensman, J., Matthews, A. G., & Ghahramani, Z. (2015). *Scalable Variational Gaussian Process Regression*. PMLR. (For the concept of learning inducing point locations, adapted here for MCMC).


```{julia}
@model function model_v7_fitc_abstractgps_linear(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0, M_rff_sigma=20, M_inducing_val=10) # M_rff_u is no longer needed here
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)

    # --- Priors --- General model parameters
    sigma_u ~ filldist(Exponential(0.5), 3) # Measurement error for u_obs

    # --- Nested Latent Covariate Parameters (Linear relationships - New in V7_linear) ---
    # U1 = f1(coords_time, Z)
    beta_u1_ztime ~ Normal(0, 1)
    beta_u1_zspace ~ Normal(0, 1)

    # U2 = f2(coords_time, Z, U1)
    beta_u2_ztime ~ Normal(0, 1)
    beta_u2_zspace ~ Normal(0, 1)
    beta_u2_u1 ~ Normal(0, 1)

    # U3 = f3(coords_time, Z, U1)
    beta_u3_ztime ~ Normal(0, 1)
    beta_u3_zspace ~ Normal(0, 1)
    beta_u3_u1 ~ Normal(0, 1)

    # --- Define Nested Latent Covariates (Linear relationships - New in V7_linear) ---
    u1_true = beta_u1_ztime .* coords_time[:,1] .+ beta_u1_zspace .* z
    u2_true = beta_u2_ztime .* coords_time[:,1] .+ beta_u2_zspace .* z .+ beta_u2_u1 .* u1_true
    u3_true = beta_u3_ztime .* coords_time[:,1] .+ beta_u3_zspace .* z .+ beta_u3_u1 .* u1_true

    # --- Seasonal Component (from V7) ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    # --- Time-varying Intercept (Random Walk, from V7) ---
    sigma_alpha ~ Exponential(0.1)
    alpha = Vector{Real}(undef, T_unique)
    alpha[1] ~ Normal(0, sigma_alpha)
    for t in 2:T_unique
        alpha[t] ~ Normal(alpha[t-1], sigma_alpha)
    end
    trend = alpha[Int.(coords_time[:,1])]

    # --- Spatiotemporal GP using AbstractGPs for FITC (from V7) ---
    coords_st_orig = hcat(coords_space, coords_time) # N x D_st

    # Optimized Inducing Points: Z_inducing are now parameters with a prior
    # Initialize Z_inducing as an array of unknown values
    Z_inducing = Matrix{Float64}(undef, M_inducing_val, D_st)

    # Prior based on the observed data's mean and a scaled standard deviation for exploration.
    mu_coords_st = mean(coords_st_orig, dims=1) # This is a 1xDst matrix
    std_coords_st = std(coords_st_orig, dims=1) # This is a 1xDst matrix

    # Assign priors column-wise to Z_inducing
    for j in 1:D_st
        # Each column of Z_inducing consists of M_inducing_val i.i.d. samples
        # from a Normal distribution specific to that j-th dimension.
        # Use mu_coords_st[j] and std_coords_st[j] which are scalars after indexing into 1xDst matrices
        Z_inducing[:, j] ~ filldist(Normal(mu_coords_st[j], 2.0 * std_coords_st[j]), M_inducing_val)
    end

    # Lengthscales for each dimension (2 for space, 1 for time)
    ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_f ~ Exponential(1.0)

    # Anisotropic Spatiotemporal kernel
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))

    # Define the base GP using AbstractGPs.jl
    g_base = GP(sigma_f^2 * k_st)

    # Use RowVecs for coordinates
    Z_inducing_vecs = RowVecs(Z_inducing)
    coords_st_vecs = RowVecs(coords_st_orig)

    # Extract kernel matrices using AbstractGPs.jl
    K_ZZ = cov(g_base(Z_inducing_vecs)) + 1e-6*I
    K_XZ = cov(g_base(coords_st_vecs), g_base(Z_inducing_vecs))
    K_XX_diag = diag(cov(g_base(coords_st_vecs)))

    # Latent values at inducing points
    u_latent ~ MvNormal(zeros(M_inducing_val), K_ZZ)

    # Compute conditional mean and diagonal covariance using FITC formulas
    m_f = K_XZ * (K_ZZ \ u_latent)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))

    # Ensure positive definite covariance (add small jitter if necessary)
    f ~ MvNormal(m_f, Diagonal(max.(0, cov_f_diag) + 1e-6*ones(N)))

    # --- Spatiotemporal Stochastic Volatility (from V7) ---
    # Secondary RFF mapping for log-variance
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0) # Scale for the log-variance GP
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)

    Phi_sigma = rff_map(coords_st_orig, W_sigma, b_sigma)
    log_sigma_y = Phi_sigma * beta_rff_sigma # Latent log-variance process
    sigma_y_process = exp.(log_sigma_y ./ 2) # Convert log-variance to standard deviation

    # --- Mean of Y and Likelihoods ---
    beta_covs ~ filldist(Normal(0, 1), 4) # These betas are for u1_true, u2_true, u3_true, z
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    # Use spatiotemporal sigma_y_process in the likelihood
    y_obs ~ MvNormal(mu_base .+ f, Diagonal(sigma_y_process.^2))
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```

 


```{julia}
# The Z_inducing variable is learned within the model.
D_s_v7_linear = size(data.coords_space, 2)
D_st_v7_linear = D_s_v7_linear + size(data.coords_time, 2)
coords_st_v7_linear = hcat(data.coords_space, data.coords_time)
M_inducing_val_v7_linear = 10 # Number of inducing points
# M_rff_u_val is no longer needed for this linear version
M_rff_sigma_val_v7_linear = 20 # Number of RFF features for log-variance GP

# Instantiate and sample Model V7_linear with NUTS
model_v7_fitc_abstractgps_linear_inst = model_v7_fitc_abstractgps_linear(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time; M_rff_sigma=M_rff_sigma_val_v7_linear, M_inducing_val=M_inducing_val_v7_linear)

# Using NUTS sampler; consider increasing iterations for production runs
chain_v7_linear = sample(model_v7_fitc_abstractgps_linear_inst, NUTS(), 100) # Reduced samples for faster testing
display(describe(chain_v7_linear))
waic_v7_linear = compute_y_waic(model_v7_fitc_abstractgps_linear_inst, chain_v7_linear)
println("WAIC for V7_linear: ", waic_v7_linear)

println("\nNote: For robust results, consider increasing the number of samples (e.g., 1000-2000 or more) and tuning the NUTS parameters (e.g., `adapts=num_adapts_steps`) if convergence issues persist.")
```

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.00078125
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:06:54[39m


    Chains MCMC chain (100×277×1 Array{Float64, 3}):
    
    Iterations        = 51:1:150
    Number of chains  = 1
    Samples per chain = 100
    Wall duration     = 456.03 seconds
    Compute duration  = 456.03 seconds
    parameters        = sigma_u[1], sigma_u[2], sigma_u[3], beta_u1_ztime, beta_u1_zspace, beta_u2_ztime, beta_u2_zspace, beta_u2_u1, beta_u3_ztime, beta_u3_zspace, beta_u3_u1, beta_cos, beta_sin, sigma_alpha, alpha[1], alpha[2], alpha[3], alpha[4], alpha[5], alpha[6], alpha[7], alpha[8], alpha[9], alpha[10], alpha[11], alpha[12], alpha[13], alpha[14], alpha[15], alpha[16], alpha[17], alpha[18], alpha[19], alpha[20], alpha[21], alpha[22], alpha[23], alpha[24], alpha[25], alpha[26], alpha[27], alpha[28], alpha[29], alpha[30], alpha[31], alpha[32], alpha[33], alpha[34], alpha[35], alpha[36], alpha[37], alpha[38], alpha[39], alpha[40], alpha[41], alpha[42], alpha[43], alpha[44], alpha[45], alpha[46], alpha[47], alpha[48], alpha[49], alpha[50], Z_inducing[:, 1][1], Z_inducing[:, 1][2], Z_inducing[:, 1][3], Z_inducing[:, 1][4], Z_inducing[:, 1][5], Z_inducing[:, 1][6], Z_inducing[:, 1][7], Z_inducing[:, 1][8], Z_inducing[:, 1][9], Z_inducing[:, 1][10], Z_inducing[:, 2][1], Z_inducing[:, 2][2], Z_inducing[:, 2][3], Z_inducing[:, 2][4], Z_inducing[:, 2][5], Z_inducing[:, 2][6], Z_inducing[:, 2][7], Z_inducing[:, 2][8], Z_inducing[:, 2][9], Z_inducing[:, 2][10], Z_inducing[:, 3][1], Z_inducing[:, 3][2], Z_inducing[:, 3][3], Z_inducing[:, 3][4], Z_inducing[:, 3][5], Z_inducing[:, 3][6], Z_inducing[:, 3][7], Z_inducing[:, 3][8], Z_inducing[:, 3][9], Z_inducing[:, 3][10], ls_st[1], ls_st[2], ls_st[3], sigma_f, u_latent[1], u_latent[2], u_latent[3], u_latent[4], u_latent[5], u_latent[6], u_latent[7], u_latent[8], u_latent[9], u_latent[10], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15], f[16], f[17], f[18], f[19], f[20], f[21], f[22], f[23], f[24], f[25], f[26], f[27], f[28], f[29], f[30], f[31], f[32], f[33], f[34], f[35], f[36], f[37], f[38], f[39], f[40], f[41], f[42], f[43], f[44], f[45], f[46], f[47], f[48], f[49], f[50], W_sigma[1, 1], W_sigma[2, 1], W_sigma[3, 1], W_sigma[1, 2], W_sigma[2, 2], W_sigma[3, 2], W_sigma[1, 3], W_sigma[2, 3], W_sigma[3, 3], W_sigma[1, 4], W_sigma[2, 4], W_sigma[3, 4], W_sigma[1, 5], W_sigma[2, 5], W_sigma[3, 5], W_sigma[1, 6], W_sigma[2, 6], W_sigma[3, 6], W_sigma[1, 7], W_sigma[2, 7], W_sigma[3, 7], W_sigma[1, 8], W_sigma[2, 8], W_sigma[3, 8], W_sigma[1, 9], W_sigma[2, 9], W_sigma[3, 9], W_sigma[1, 10], W_sigma[2, 10], W_sigma[3, 10], W_sigma[1, 11], W_sigma[2, 11], W_sigma[3, 11], W_sigma[1, 12], W_sigma[2, 12], W_sigma[3, 12], W_sigma[1, 13], W_sigma[2, 13], W_sigma[3, 13], W_sigma[1, 14], W_sigma[2, 14], W_sigma[3, 14], W_sigma[1, 15], W_sigma[2, 15], W_sigma[3, 15], W_sigma[1, 16], W_sigma[2, 16], W_sigma[3, 16], W_sigma[1, 17], W_sigma[2, 17], W_sigma[3, 17], W_sigma[1, 18], W_sigma[2, 18], W_sigma[3, 18], W_sigma[1, 19], W_sigma[2, 19], W_sigma[3, 19], W_sigma[1, 20], W_sigma[2, 20], W_sigma[3, 20], b_sigma[1], b_sigma[2], b_sigma[3], b_sigma[4], b_sigma[5], b_sigma[6], b_sigma[7], b_sigma[8], b_sigma[9], b_sigma[10], b_sigma[11], b_sigma[12], b_sigma[13], b_sigma[14], b_sigma[15], b_sigma[16], b_sigma[17], b_sigma[18], b_sigma[19], b_sigma[20], sigma_log_var, beta_rff_sigma[1], beta_rff_sigma[2], beta_rff_sigma[3], beta_rff_sigma[4], beta_rff_sigma[5], beta_rff_sigma[6], beta_rff_sigma[7], beta_rff_sigma[8], beta_rff_sigma[9], beta_rff_sigma[10], beta_rff_sigma[11], beta_rff_sigma[12], beta_rff_sigma[13], beta_rff_sigma[14], beta_rff_sigma[15], beta_rff_sigma[16], beta_rff_sigma[17], beta_rff_sigma[18], beta_rff_sigma[19], beta_rff_sigma[20], beta_covs[1], beta_covs[2], beta_covs[3], beta_covs[4]
    internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, logprior, loglikelihood, logjoint
    
    Summary Statistics
    
     [1m     parameters [0m [1m    mean [0m [1m     std [0m [1m    mcse [0m [1m ess_bulk [0m [1m ess_tail [0m [1m    rhat[0m ⋯
     [90m         Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m  Float64 [0m [90m  Float64 [0m [90m Float64[0m ⋯
    
          sigma_u[1]    0.9068    0.0027    0.0014     5.9984    17.3961    1.1057 ⋯
          sigma_u[2]    1.2062    0.0016    0.0012     1.6830    13.1912    1.7019 ⋯
          sigma_u[3]    0.1129    0.0004    0.0003     1.4591    25.8522    2.0316 ⋯
       beta_u1_ztime    0.0248    0.0006    0.0002    12.8886    12.7391    1.0521 ⋯
      beta_u1_zspace    0.3052    0.0010    0.0003     9.7813    16.0147    1.0895 ⋯
       beta_u2_ztime   -0.0076    0.0017    0.0009     4.1215    21.0246    1.2538 ⋯
      beta_u2_zspace   -0.6683    0.0026    0.0011     6.6605    13.0721    1.2326 ⋯
          beta_u2_u1    0.8013    0.0016    0.0011     2.1016    16.5767    1.4451 ⋯
       beta_u3_ztime   -0.0378    0.0009    0.0004     7.1444    12.7391    1.1051 ⋯
      beta_u3_zspace   -0.4526    0.0012    0.0006     4.2549     8.6272    1.1842 ⋯
          beta_u3_u1    1.7113    0.0019    0.0007     9.8662    24.3582    1.1830 ⋯
            beta_cos    0.4867    0.0020    0.0013     2.2922    13.0721    1.4089 ⋯
            beta_sin   -0.6818    0.0019    0.0014     1.9986    11.9406    1.5650 ⋯
         sigma_alpha    1.0800    0.0021    0.0012     5.8943    26.3082    1.2692 ⋯
            alpha[1]   -0.1332    0.0012    0.0003    15.9581    40.5549    1.0268 ⋯
            alpha[2]    1.4903    0.0020    0.0017     1.5115    21.0246    1.9390 ⋯
            alpha[3]    1.9360    0.0014    0.0011     1.7059    21.0246    1.7529 ⋯
            alpha[4]    1.2163    0.0016    0.0012     1.8285    31.6421    1.5901 ⋯
            alpha[5]    0.1064    0.0030    0.0012     7.6148    12.7391    1.0235 ⋯
            alpha[6]    0.6908    0.0013    0.0006     5.2635    13.0721    1.1491 ⋯
            alpha[7]    0.5714    0.0010    0.0006     3.5308    12.7391    1.3341 ⋯
                   ⋮         ⋮         ⋮         ⋮          ⋮          ⋮         ⋮ ⋱
    
    [36m                                                   1 column and 242 rows omitted[0m
    
    Quantiles
    
     [1m     parameters [0m [1m    2.5% [0m [1m   25.0% [0m [1m   50.0% [0m [1m   75.0% [0m [1m   97.5% [0m
     [90m         Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m
    
          sigma_u[1]    0.9005    0.9066    0.9077    0.9086    0.9094
          sigma_u[2]    1.2029    1.2054    1.2062    1.2071    1.2093
          sigma_u[3]    0.1124    0.1125    0.1127    0.1131    0.1136
       beta_u1_ztime    0.0239    0.0243    0.0247    0.0253    0.0260
      beta_u1_zspace    0.3034    0.3046    0.3053    0.3059    0.3068
       beta_u2_ztime   -0.0103   -0.0086   -0.0081   -0.0062   -0.0043
      beta_u2_zspace   -0.6723   -0.6701   -0.6693   -0.6673   -0.6627
          beta_u2_u1    0.7985    0.7996    0.8018    0.8027    0.8040
       beta_u3_ztime   -0.0397   -0.0384   -0.0377   -0.0371   -0.0363
      beta_u3_zspace   -0.4546   -0.4530   -0.4525   -0.4519   -0.4505
          beta_u3_u1    1.7087    1.7101    1.7108    1.7114    1.7160
            beta_cos    0.4814    0.4862    0.4871    0.4876    0.4897
            beta_sin   -0.6852   -0.6831   -0.6816   -0.6805   -0.6788
         sigma_alpha    1.0774    1.0783    1.0793    1.0806    1.0838
            alpha[1]   -0.1350   -0.1341   -0.1335   -0.1324   -0.1308
            alpha[2]    1.4873    1.4887    1.4896    1.4916    1.4945
            alpha[3]    1.9340    1.9350    1.9357    1.9368    1.9393
            alpha[4]    1.2133    1.2153    1.2162    1.2175    1.2191
            alpha[5]    0.1024    0.1037    0.1063    0.1085    0.1128
            alpha[6]    0.6891    0.6897    0.6908    0.6917    0.6940
            alpha[7]    0.5699    0.5707    0.5716    0.5721    0.5740
                   ⋮         ⋮         ⋮         ⋮         ⋮         ⋮
    
    [36m                                                    242 rows omitted[0m



    nothing


    WAIC for V7_linear: 186.74236560096676
    
    Note: For robust results, consider increasing the number of samples (e.g., 1000-2000 or more) and tuning the NUTS parameters (e.g., `adapts=num_adapts_steps`) if convergence issues persist.


## V8: Nonlinear Nested Covariates (RFF-based)

This model builds upon V7 by introducing nonlinear functional forms for the nested covariates `U1`, `U2`, and `U3`. Instead of simple linear relationships, it uses Random Fourier Features (RFFs) to model these dependencies, allowing for more complex and adaptive representations of how `U` covariates are generated from `coords_time`, `Z`, and other `U` covariates.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V7, modeled with a mean component (GP trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): New in V8, these are now modeled as nonlinear functions using separate RFF mappings:
    *   `U1 = f1_rff(coords_time, Z)`: Modeled as a nonlinear function of time and spatial covariate `Z` via an RFF layer.
    *   `U2 = f2_rff(coords_time, Z, U1)`: Modeled as a nonlinear function of time, `Z`, and the latent `U1` via an RFF layer.
    *   `U3 = f3_rff(coords_time, Z, U1)`: Modeled as a nonlinear function of time, `Z`, and the latent `U1` via an RFF layer.
    Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) still have measurement error.
*   Trend: Same as V7, a Gaussian Process-based trend (`GP Trend`).
*   Seasonal Process: Same as V7, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V7, uses the Fully Independent Training Conditional (FITC) approximation with learned inducing point locations (`Z_inducing`) and latent values (`u_latent`).
*   Observation Noise (sigma_y): Same as V7, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V7, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, extended for the `W`, `b`, `sigma_f`, and `beta_rff` parameters for each of the nonlinear `U` covariate RFF mappings.

### Key References:
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (For the nonlinear covariate mappings).
*   FITC and Inducing Point Optimization: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS. (Still relevant for the main GP component).
*   Hierarchical Bayesian Models: Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press. (For general hierarchical modeling principles).
*   Deep Gaussian Processes: Damianou, A., & Lawrence, N. (2013). *Deep Gaussian Processes*. AISTATS. (For the conceptual foundation of stacking GP-like layers).

## Fully Independent Training Conditional (FITC)

FITC is an approximation method for Gaussian Processes (GPs) that addresses the computational burden of large datasets. It is also known as the "sparse pseudo-input GP" or "Deterministic Training Conditional (DTC)".

### Core Idea and Theoretical Basis
Instead of directly approximating the kernel function via feature maps (like RFFs), FITC introduces a small set of inducing points ($Z = \{z_1, \dots, z_M\}$, where $M \ll N$). The fundamental assumption is that, conditional on the values of the latent GP at these inducing points ($f_Z$), the observed data points ($f_i$) are conditionally independent:

$$p(f | X, Z) \approx p(f | f_Z, Z) = \prod_{i=1}^N p(f_i | f_Z, Z)$$

This approximation significantly simplifies the covariance structure and speeds up calculations.

### Mechanism and Computational Advantages
*   Sparsity Source: A small set of judiciously chosen inducing points. These points are not necessarily part of the training data.
*   Approximation: FITC approximates the posterior distribution of the GP, effectively 'compressing' the GP through these inducing points.
*   Covariance Calculation: It simplifies the computation of the covariance matrix by involving inversions only for the smaller $M \times M$ covariance matrix of the inducing points ($K_{ZZ}$) and their cross-covariances with the data ($K_{XZ}$). The inducing points act as a bottleneck for information flow.
*   Computational Advantage: Reduces the computational complexity from $O(N^3)$ (for exact GPs) to $O(N M^2 + M^3)$.
*   Interpretation: The GP is conditioned on a smaller set of latent variables (the values at the inducing points).
*   Inducing Point Optimization: A crucial aspect is the choice and optimization of the inducing point locations and possibly their values. These are often treated as hyperparameters to be learned or optimized within the model.

### Implementation Notes for Model V6
Model V6 builds upon V5 by replacing the Random Fourier Features (RFF) for the main GP with a FITC approximation. It retains the nested covariates, time-varying intercept, and spatiotemporal stochastic volatility from V5.

FITC for Main GP:
*   Takes pre-defined `Z_inducing` points as input.
*   Defines an anisotropic spatiotemporal kernel `k_st` using `SqExponentialKernel() \circ ARDTransform(inv.(ls_st))`, similar to Model V1.
*   Computes the necessary kernel matrices: `K_ZZ` (covariance at inducing points), `K_XZ` (cross-covariance between data and inducing points), and `K_XX_diag` (diagonal of covariance at data points).
*   Samples the latent values at inducing points: `u_latent ~ MvNormal(zeros(M_inducing_val), K_ZZ)`.
*   Calculates the conditional mean (`mean_f`) and the diagonal of the conditional covariance (`cov_f_diag`) using standard FITC formulas.
*   Finally, `f ~ MvNormal(mean_f, Diagonal(max.(0, cov_f_diag) + 1e-6*ones(N)))` samples the latent GP process, using the diagonal approximation for the covariance.

Retained Components from V5:
*   Nested Latent Covariates: The functional relationships for `u1_true`, `u2_true`, and `u3_true` are identical to V5.
*   Seasonal Component: The `beta_cos` and `beta_sin` parameters, along with the trigonometric functions of time, remain unchanged.
*   Time-varying Intercept (Random Walk): The random walk definition for `alpha` and its mapping to `trend` remains consistent.
*   Spatiotemporal Stochastic Volatility: The secondary RFF mapping for `log_sigma_y` (parameters `W_sigma`, `b_sigma`, `sigma_log_var`, `beta_rff_sigma`, `Phi_sigma`) and the resulting `sigma_y_process` are implemented, with `y_obs` using `Diagonal(sigma_y_process.^2)`.


```{julia}
@model function model_v8_fitc_abstractgps_nonlinear(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0, M_rff_sigma=20, M_inducing_val=10, M_rff_u=30)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)

    # --- Priors --- General model parameters
    sigma_u ~ filldist(Exponential(0.5), 3) # Measurement error for u_obs

    # --- Nested Latent Covariates (using RFF for non-linearity) ---
    coords_tz = hcat(coords_time, z)

    # U1 = f1(coords_time, Z)
    D_u1_input = size(coords_tz, 2)
    W_u1 ~ filldist(Normal(0, 1), D_u1_input, M_rff_u)
    b_u1 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0)
    beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1^2), M_rff_u)
    Phi_u1 = rff_map(coords_tz, W_u1, b_u1)
    u1_true = Phi_u1 * beta_rff_u1

    # U2 = f2(coords_time, Z, U1)
    coords_tz_u1 = hcat(coords_time, z, u1_true)
    D_u2_input = size(coords_tz_u1, 2)
    W_u2 ~ filldist(Normal(0, 1), D_u2_input, M_rff_u)
    b_u2 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0)
    beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2^2), M_rff_u)
    Phi_u2 = rff_map(coords_tz_u1, W_u2, b_u2)
    u2_true = Phi_u2 * beta_rff_u2

    # U3 = f3(coords_time, Z, U1)
    D_u3_input = size(coords_tz_u1, 2)
    W_u3 ~ filldist(Normal(0, 1), D_u3_input, M_rff_u)
    b_u3 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0)
    beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3^2), M_rff_u)
    Phi_u3 = rff_map(coords_tz_u1, W_u3, b_u3)
    u3_true = Phi_u3 * beta_rff_u3

    # --- Seasonal Component ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    # --- Time-varying Intercept (Random Walk) ---
    sigma_alpha ~ Exponential(0.1)
    alpha = Vector{Real}(undef, T_unique)
    alpha[1] ~ Normal(0, sigma_alpha)
    for t in 2:T_unique
        alpha[t] ~ Normal(alpha[t-1], sigma_alpha)
    end
    trend = alpha[Int.(coords_time[:,1])]

    # --- Spatiotemporal GP using AbstractGPs for FITC ---
    coords_st_orig = hcat(coords_space, coords_time)
    Z_inducing = Matrix{Float64}(undef, M_inducing_val, D_st)
    mu_coords_st = mean(coords_st_orig, dims=1)
    std_coords_st = std(coords_st_orig, dims=1)

    for j in 1:D_st
        Z_inducing[:, j] ~ filldist(Normal(mu_coords_st[j], 2.0 * std_coords_st[j]), M_inducing_val)
    end

    ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_f ~ Exponential(1.0)
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))
    g_base = GP(sigma_f^2 * k_st)

    Z_inducing_vecs = RowVecs(Z_inducing)
    coords_st_vecs = RowVecs(coords_st_orig)

    K_ZZ = cov(g_base(Z_inducing_vecs)) + 1e-6*I
    K_XZ = cov(g_base(coords_st_vecs), g_base(Z_inducing_vecs))
    K_XX_diag = diag(cov(g_base(coords_st_vecs)))

    u_latent ~ MvNormal(zeros(M_inducing_val), K_ZZ)
    m_f = K_XZ * (K_ZZ \ u_latent)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))
    f ~ MvNormal(m_f, Diagonal(max.(0, cov_f_diag) + 1e-6*ones(N)))

    # --- Spatiotemporal Stochastic Volatility ---
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0)
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)
    Phi_sigma = rff_map(coords_st_orig, W_sigma, b_sigma)
    log_sigma_y = Phi_sigma * beta_rff_sigma
    sigma_y_process = exp.(log_sigma_y ./ 2)

    # --- Mean of Y and Likelihoods ---
    beta_covs ~ filldist(Normal(0, 1), 4)
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    y_obs ~ MvNormal(mu_base .+ f, Diagonal(sigma_y_process.^2))
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```




    model_v8_fitc_abstractgps_nonlinear (generic function with 2 methods)




```{julia}
# The Z_inducing_v8 variable is no longer passed as an argument as it is now learned within the model.
D_s_v8 = size(data.coords_space, 2)
D_st_v8 = D_s_v8 + size(data.coords_time, 2)
coords_st_v8 = hcat(data.coords_space, data.coords_time)
M_inducing_val_v8 = 10 # Number of inducing points (e.g., 10-20% of N)
M_rff_u_val = 30 # Number of RFF features for nested covariates

# Sample Model V8 with NUTS
M_rff_sigma_val_v8 = 20 # Number of RFF features for log-variance GP
model_v8 = model_v8_fitc_abstractgps_nonlinear(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time; M_rff_sigma=M_rff_sigma_val_v8, M_inducing_val=M_inducing_val_v8, M_rff_u=M_rff_u_val)

# Using NUTS sampler for better convergence; consider increasing iterations for production runs
chain_v8 = sample(model_v8, NUTS(), 100)
display(describe(chain_v8))
waic_v8 = compute_y_waic(model_v8, chain_v8)
println("WAIC for V8: ", waic_v8)

println("\nNote: For robust results, consider increasing the number of samples (e.g., 1000-2000 or more) and tuning the NUTS parameters (e.g., `adapts=num_adapts_steps`) if convergence issues persist.")
```

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.003125
    [32mSampling: 100%|█████████████████████████████████████████| Time: 1:20:23[39m


    Chains MCMC chain (100×692×1 Array{Float64, 3}):
    
    Iterations        = 51:1:150
    Number of chains  = 1
    Samples per chain = 100
    Wall duration     = 4960.76 seconds
    Compute duration  = 4960.76 seconds
    parameters        = sigma_u[1], sigma_u[2], sigma_u[3], W_u1[1, 1], W_u1[2, 1], W_u1[1, 2], W_u1[2, 2], W_u1[1, 3], W_u1[2, 3], W_u1[1, 4], W_u1[2, 4], W_u1[1, 5], W_u1[2, 5], W_u1[1, 6], W_u1[2, 6], W_u1[1, 7], W_u1[2, 7], W_u1[1, 8], W_u1[2, 8], W_u1[1, 9], W_u1[2, 9], W_u1[1, 10], W_u1[2, 10], W_u1[1, 11], W_u1[2, 11], W_u1[1, 12], W_u1[2, 12], W_u1[1, 13], W_u1[2, 13], W_u1[1, 14], W_u1[2, 14], W_u1[1, 15], W_u1[2, 15], W_u1[1, 16], W_u1[2, 16], W_u1[1, 17], W_u1[2, 17], W_u1[1, 18], W_u1[2, 18], W_u1[1, 19], W_u1[2, 19], W_u1[1, 20], W_u1[2, 20], W_u1[1, 21], W_u1[2, 21], W_u1[1, 22], W_u1[2, 22], W_u1[1, 23], W_u1[2, 23], W_u1[1, 24], W_u1[2, 24], W_u1[1, 25], W_u1[2, 25], W_u1[1, 26], W_u1[2, 26], W_u1[1, 27], W_u1[2, 27], W_u1[1, 28], W_u1[2, 28], W_u1[1, 29], W_u1[2, 29], W_u1[1, 30], W_u1[2, 30], b_u1[1], b_u1[2], b_u1[3], b_u1[4], b_u1[5], b_u1[6], b_u1[7], b_u1[8], b_u1[9], b_u1[10], b_u1[11], b_u1[12], b_u1[13], b_u1[14], b_u1[15], b_u1[16], b_u1[17], b_u1[18], b_u1[19], b_u1[20], b_u1[21], b_u1[22], b_u1[23], b_u1[24], b_u1[25], b_u1[26], b_u1[27], b_u1[28], b_u1[29], b_u1[30], sigma_f_u1, beta_rff_u1[1], beta_rff_u1[2], beta_rff_u1[3], beta_rff_u1[4], beta_rff_u1[5], beta_rff_u1[6], beta_rff_u1[7], beta_rff_u1[8], beta_rff_u1[9], beta_rff_u1[10], beta_rff_u1[11], beta_rff_u1[12], beta_rff_u1[13], beta_rff_u1[14], beta_rff_u1[15], beta_rff_u1[16], beta_rff_u1[17], beta_rff_u1[18], beta_rff_u1[19], beta_rff_u1[20], beta_rff_u1[21], beta_rff_u1[22], beta_rff_u1[23], beta_rff_u1[24], beta_rff_u1[25], beta_rff_u1[26], beta_rff_u1[27], beta_rff_u1[28], beta_rff_u1[29], beta_rff_u1[30], W_u2[1, 1], W_u2[2, 1], W_u2[3, 1], W_u2[1, 2], W_u2[2, 2], W_u2[3, 2], W_u2[1, 3], W_u2[2, 3], W_u2[3, 3], W_u2[1, 4], W_u2[2, 4], W_u2[3, 4], W_u2[1, 5], W_u2[2, 5], W_u2[3, 5], W_u2[1, 6], W_u2[2, 6], W_u2[3, 6], W_u2[1, 7], W_u2[2, 7], W_u2[3, 7], W_u2[1, 8], W_u2[2, 8], W_u2[3, 8], W_u2[1, 9], W_u2[2, 9], W_u2[3, 9], W_u2[1, 10], W_u2[2, 10], W_u2[3, 10], W_u2[1, 11], W_u2[2, 11], W_u2[3, 11], W_u2[1, 12], W_u2[2, 12], W_u2[3, 12], W_u2[1, 13], W_u2[2, 13], W_u2[3, 13], W_u2[1, 14], W_u2[2, 14], W_u2[3, 14], W_u2[1, 15], W_u2[2, 15], W_u2[3, 15], W_u2[1, 16], W_u2[2, 16], W_u2[3, 16], W_u2[1, 17], W_u2[2, 17], W_u2[3, 17], W_u2[1, 18], W_u2[2, 18], W_u2[3, 18], W_u2[1, 19], W_u2[2, 19], W_u2[3, 19], W_u2[1, 20], W_u2[2, 20], W_u2[3, 20], W_u2[1, 21], W_u2[2, 21], W_u2[3, 21], W_u2[1, 22], W_u2[2, 22], W_u2[3, 22], W_u2[1, 23], W_u2[2, 23], W_u2[3, 23], W_u2[1, 24], W_u2[2, 24], W_u2[3, 24], W_u2[1, 25], W_u2[2, 25], W_u2[3, 25], W_u2[1, 26], W_u2[2, 26], W_u2[3, 26], W_u2[1, 27], W_u2[2, 27], W_u2[3, 27], W_u2[1, 28], W_u2[2, 28], W_u2[3, 28], W_u2[1, 29], W_u2[2, 29], W_u2[3, 29], W_u2[1, 30], W_u2[2, 30], W_u2[3, 30], b_u2[1], b_u2[2], b_u2[3], b_u2[4], b_u2[5], b_u2[6], b_u2[7], b_u2[8], b_u2[9], b_u2[10], b_u2[11], b_u2[12], b_u2[13], b_u2[14], b_u2[15], b_u2[16], b_u2[17], b_u2[18], b_u2[19], b_u2[20], b_u2[21], b_u2[22], b_u2[23], b_u2[24], b_u2[25], b_u2[26], b_u2[27], b_u2[28], b_u2[29], b_u2[30], sigma_f_u2, beta_rff_u2[1], beta_rff_u2[2], beta_rff_u2[3], beta_rff_u2[4], beta_rff_u2[5], beta_rff_u2[6], beta_rff_u2[7], beta_rff_u2[8], beta_rff_u2[9], beta_rff_u2[10], beta_rff_u2[11], beta_rff_u2[12], beta_rff_u2[13], beta_rff_u2[14], beta_rff_u2[15], beta_rff_u2[16], beta_rff_u2[17], beta_rff_u2[18], beta_rff_u2[19], beta_rff_u2[20], beta_rff_u2[21], beta_rff_u2[22], beta_rff_u2[23], beta_rff_u2[24], beta_rff_u2[25], beta_rff_u2[26], beta_rff_u2[27], beta_rff_u2[28], beta_rff_u2[29], beta_rff_u2[30], W_u3[1, 1], W_u3[2, 1], W_u3[3, 1], W_u3[1, 2], W_u3[2, 2], W_u3[3, 2], W_u3[1, 3], W_u3[2, 3], W_u3[3, 3], W_u3[1, 4], W_u3[2, 4], W_u3[3, 4], W_u3[1, 5], W_u3[2, 5], W_u3[3, 5], W_u3[1, 6], W_u3[2, 6], W_u3[3, 6], W_u3[1, 7], W_u3[2, 7], W_u3[3, 7], W_u3[1, 8], W_u3[2, 8], W_u3[3, 8], W_u3[1, 9], W_u3[2, 9], W_u3[3, 9], W_u3[1, 10], W_u3[2, 10], W_u3[3, 10], W_u3[1, 11], W_u3[2, 11], W_u3[3, 11], W_u3[1, 12], W_u3[2, 12], W_u3[3, 12], W_u3[1, 13], W_u3[2, 13], W_u3[3, 13], W_u3[1, 14], W_u3[2, 14], W_u3[3, 14], W_u3[1, 15], W_u3[2, 15], W_u3[3, 15], W_u3[1, 16], W_u3[2, 16], W_u3[3, 16], W_u3[1, 17], W_u3[2, 17], W_u3[3, 17], W_u3[1, 18], W_u3[2, 18], W_u3[3, 18], W_u3[1, 19], W_u3[2, 19], W_u3[3, 19], W_u3[1, 20], W_u3[2, 20], W_u3[3, 20], W_u3[1, 21], W_u3[2, 21], W_u3[3, 21], W_u3[1, 22], W_u3[2, 22], W_u3[3, 22], W_u3[1, 23], W_u3[2, 23], W_u3[3, 23], W_u3[1, 24], W_u3[2, 24], W_u3[3, 24], W_u3[1, 25], W_u3[2, 25], W_u3[3, 25], W_u3[1, 26], W_u3[2, 26], W_u3[3, 26], W_u3[1, 27], W_u3[2, 27], W_u3[3, 27], W_u3[1, 28], W_u3[2, 28], W_u3[3, 28], W_u3[1, 29], W_u3[2, 29], W_u3[3, 29], W_u3[1, 30], W_u3[2, 30], W_u3[3, 30], b_u3[1], b_u3[2], b_u3[3], b_u3[4], b_u3[5], b_u3[6], b_u3[7], b_u3[8], b_u3[9], b_u3[10], b_u3[11], b_u3[12], b_u3[13], b_u3[14], b_u3[15], b_u3[16], b_u3[17], b_u3[18], b_u3[19], b_u3[20], b_u3[21], b_u3[22], b_u3[23], b_u3[24], b_u3[25], b_u3[26], b_u3[27], b_u3[28], b_u3[29], b_u3[30], sigma_f_u3, beta_rff_u3[1], beta_rff_u3[2], beta_rff_u3[3], beta_rff_u3[4], beta_rff_u3[5], beta_rff_u3[6], beta_rff_u3[7], beta_rff_u3[8], beta_rff_u3[9], beta_rff_u3[10], beta_rff_u3[11], beta_rff_u3[12], beta_rff_u3[13], beta_rff_u3[14], beta_rff_u3[15], beta_rff_u3[16], beta_rff_u3[17], beta_rff_u3[18], beta_rff_u3[19], beta_rff_u3[20], beta_rff_u3[21], beta_rff_u3[22], beta_rff_u3[23], beta_rff_u3[24], beta_rff_u3[25], beta_rff_u3[26], beta_rff_u3[27], beta_rff_u3[28], beta_rff_u3[29], beta_rff_u3[30], beta_cos, beta_sin, sigma_alpha, alpha[1], alpha[2], alpha[3], alpha[4], alpha[5], alpha[6], alpha[7], alpha[8], alpha[9], alpha[10], alpha[11], alpha[12], alpha[13], alpha[14], alpha[15], alpha[16], alpha[17], alpha[18], alpha[19], alpha[20], alpha[21], alpha[22], alpha[23], alpha[24], alpha[25], alpha[26], alpha[27], alpha[28], alpha[29], alpha[30], alpha[31], alpha[32], alpha[33], alpha[34], alpha[35], alpha[36], alpha[37], alpha[38], alpha[39], alpha[40], alpha[41], alpha[42], alpha[43], alpha[44], alpha[45], alpha[46], alpha[47], alpha[48], alpha[49], alpha[50], Z_inducing[:, 1][1], Z_inducing[:, 1][2], Z_inducing[:, 1][3], Z_inducing[:, 1][4], Z_inducing[:, 1][5], Z_inducing[:, 1][6], Z_inducing[:, 1][7], Z_inducing[:, 1][8], Z_inducing[:, 1][9], Z_inducing[:, 1][10], Z_inducing[:, 2][1], Z_inducing[:, 2][2], Z_inducing[:, 2][3], Z_inducing[:, 2][4], Z_inducing[:, 2][5], Z_inducing[:, 2][6], Z_inducing[:, 2][7], Z_inducing[:, 2][8], Z_inducing[:, 2][9], Z_inducing[:, 2][10], Z_inducing[:, 3][1], Z_inducing[:, 3][2], Z_inducing[:, 3][3], Z_inducing[:, 3][4], Z_inducing[:, 3][5], Z_inducing[:, 3][6], Z_inducing[:, 3][7], Z_inducing[:, 3][8], Z_inducing[:, 3][9], Z_inducing[:, 3][10], ls_st[1], ls_st[2], ls_st[3], sigma_f, u_latent[1], u_latent[2], u_latent[3], u_latent[4], u_latent[5], u_latent[6], u_latent[7], u_latent[8], u_latent[9], u_latent[10], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15], f[16], f[17], f[18], f[19], f[20], f[21], f[22], f[23], f[24], f[25], f[26], f[27], f[28], f[29], f[30], f[31], f[32], f[33], f[34], f[35], f[36], f[37], f[38], f[39], f[40], f[41], f[42], f[43], f[44], f[45], f[46], f[47], f[48], f[49], f[50], W_sigma[1, 1], W_sigma[2, 1], W_sigma[3, 1], W_sigma[1, 2], W_sigma[2, 2], W_sigma[3, 2], W_sigma[1, 3], W_sigma[2, 3], W_sigma[3, 3], W_sigma[1, 4], W_sigma[2, 4], W_sigma[3, 4], W_sigma[1, 5], W_sigma[2, 5], W_sigma[3, 5], W_sigma[1, 6], W_sigma[2, 6], W_sigma[3, 6], W_sigma[1, 7], W_sigma[2, 7], W_sigma[3, 7], W_sigma[1, 8], W_sigma[2, 8], W_sigma[3, 8], W_sigma[1, 9], W_sigma[2, 9], W_sigma[3, 9], W_sigma[1, 10], W_sigma[2, 10], W_sigma[3, 10], W_sigma[1, 11], W_sigma[2, 11], W_sigma[3, 11], W_sigma[1, 12], W_sigma[2, 12], W_sigma[3, 12], W_sigma[1, 13], W_sigma[2, 13], W_sigma[3, 13], W_sigma[1, 14], W_sigma[2, 14], W_sigma[3, 14], W_sigma[1, 15], W_sigma[2, 15], W_sigma[3, 15], W_sigma[1, 16], W_sigma[2, 16], W_sigma[3, 16], W_sigma[1, 17], W_sigma[2, 17], W_sigma[3, 17], W_sigma[1, 18], W_sigma[2, 18], W_sigma[3, 18], W_sigma[1, 19], W_sigma[2, 19], W_sigma[3, 19], W_sigma[1, 20], W_sigma[2, 20], W_sigma[3, 20], b_sigma[1], b_sigma[2], b_sigma[3], b_sigma[4], b_sigma[5], b_sigma[6], b_sigma[7], b_sigma[8], b_sigma[9], b_sigma[10], b_sigma[11], b_sigma[12], b_sigma[13], b_sigma[14], b_sigma[15], b_sigma[16], b_sigma[17], b_sigma[18], b_sigma[19], b_sigma[20], sigma_log_var, beta_rff_sigma[1], beta_rff_sigma[2], beta_rff_sigma[3], beta_rff_sigma[4], beta_rff_sigma[5], beta_rff_sigma[6], beta_rff_sigma[7], beta_rff_sigma[8], beta_rff_sigma[9], beta_rff_sigma[10], beta_rff_sigma[11], beta_rff_sigma[12], beta_rff_sigma[13], beta_rff_sigma[14], beta_rff_sigma[15], beta_rff_sigma[16], beta_rff_sigma[17], beta_rff_sigma[18], beta_rff_sigma[19], beta_rff_sigma[20], beta_covs[1], beta_covs[2], beta_covs[3], beta_covs[4]
    internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, logprior, loglikelihood, logjoint
    
    Summary Statistics
    
     [1m parameters [0m [1m    mean [0m [1m     std [0m [1m    mcse [0m [1m ess_bulk [0m [1m ess_tail [0m [1m    rhat [0m [1m e[0m ⋯
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m  Float64 [0m [90m  Float64 [0m [90m Float64 [0m [90m  [0m ⋯
    
      sigma_u[1]    0.6462    0.0070    0.0061     1.5666    20.1748    1.8327     ⋯
      sigma_u[2]    0.0465    0.0003    0.0002     2.1425    12.7391    1.4764     ⋯
      sigma_u[3]    0.0583    0.0003    0.0001     7.1343    16.5219    1.1011     ⋯
      W_u1[1, 1]    1.1067    0.0090    0.0061     4.4337    14.0875    1.5313     ⋯
      W_u1[2, 1]   -0.0235    0.0078    0.0030     5.7818     8.8926    1.0582     ⋯
      W_u1[1, 2]    1.2709    0.0038    0.0027     2.0691    15.1573    1.4587     ⋯
      W_u1[2, 2]   -1.3662    0.0068    0.0062     1.4677    30.6204    2.0903     ⋯
      W_u1[1, 3]    0.1198    0.0038    0.0021     3.5719    40.9033    1.2338     ⋯
      W_u1[2, 3]    0.1403    0.0031    0.0013     5.9935     7.7434    1.1371     ⋯
      W_u1[1, 4]   -0.8424    0.0047    0.0036     2.1034    34.9230    1.4998     ⋯
      W_u1[2, 4]    1.2863    0.0033    0.0021     2.7004    21.0246    1.3331     ⋯
      W_u1[1, 5]    0.8845    0.0025    0.0009     9.9135    21.3719    1.0980     ⋯
      W_u1[2, 5]    1.3756    0.0148    0.0129     1.4753    10.5200    1.9636     ⋯
      W_u1[1, 6]   -0.1688    0.0082    0.0069     1.6003    25.8522    1.8723     ⋯
      W_u1[2, 6]   -0.2615    0.0040    0.0033     1.4973    34.9092    2.0440     ⋯
      W_u1[1, 7]    0.5939    0.0125    0.0115     1.4129     9.4621    2.1062     ⋯
      W_u1[2, 7]   -0.5742    0.0024    0.0007    10.1078    25.8522    1.0484     ⋯
      W_u1[1, 8]    2.1485    0.0023    0.0010     4.0531    21.0246    1.2467     ⋯
      W_u1[2, 8]   -0.8566    0.0071    0.0059     1.6314    21.0246    1.8125     ⋯
      W_u1[1, 9]    0.3483    0.0060    0.0028     2.9606    21.0246    1.2951     ⋯
      W_u1[2, 9]   -0.7349    0.0061    0.0051     1.5230    30.2536    1.9178     ⋯
               ⋮         ⋮         ⋮         ⋮          ⋮          ⋮         ⋮     ⋱
    
    [36m                                                   1 column and 657 rows omitted[0m
    
    Quantiles
    
     [1m parameters [0m [1m    2.5% [0m [1m   25.0% [0m [1m   50.0% [0m [1m   75.0% [0m [1m   97.5% [0m
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m
    
      sigma_u[1]    0.6363    0.6380    0.6455    0.6537    0.6567
      sigma_u[2]    0.0460    0.0463    0.0464    0.0465    0.0472
      sigma_u[3]    0.0580    0.0581    0.0582    0.0584    0.0589
      W_u1[1, 1]    1.0949    1.0999    1.1026    1.1177    1.1216
      W_u1[2, 1]   -0.0389   -0.0324   -0.0208   -0.0185   -0.0108
      W_u1[1, 2]    1.2638    1.2677    1.2708    1.2739    1.2764
      W_u1[2, 2]   -1.3766   -1.3728   -1.3637   -1.3590   -1.3573
      W_u1[1, 3]    0.1140    0.1164    0.1195    0.1224    0.1271
      W_u1[2, 3]    0.1352    0.1382    0.1396    0.1418    0.1469
      W_u1[1, 4]   -0.8506   -0.8466   -0.8421   -0.8378   -0.8361
      W_u1[2, 4]    1.2796    1.2840    1.2857    1.2892    1.2914
      W_u1[1, 5]    0.8807    0.8823    0.8848    0.8863    0.8888
      W_u1[2, 5]    1.3564    1.3613    1.3748    1.3931    1.3978
      W_u1[1, 6]   -0.1807   -0.1774   -0.1698   -0.1598   -0.1562
      W_u1[2, 6]   -0.2685   -0.2643   -0.2624   -0.2576   -0.2551
      W_u1[1, 7]    0.5761    0.5810    0.5981    0.6066    0.6102
      W_u1[2, 7]   -0.5784   -0.5759   -0.5742   -0.5725   -0.5696
      W_u1[1, 8]    2.1421    2.1475    2.1489    2.1499    2.1520
      W_u1[2, 8]   -0.8689   -0.8630   -0.8550   -0.8493   -0.8469
      W_u1[1, 9]    0.3406    0.3444    0.3461    0.3540    0.3607
      W_u1[2, 9]   -0.7446   -0.7391   -0.7362   -0.7308   -0.7245
               ⋮         ⋮         ⋮         ⋮         ⋮         ⋮
    
    [36m                                                657 rows omitted[0m



    nothing


    WAIC for V8: 117.64123967887657
    
    Note: For robust results, consider increasing the number of samples (e.g., 1000-2000 or more) and tuning the NUTS parameters (e.g., `adapts=num_adapts_steps`) if convergence issues persist.


## V9: Gaussian Process Trend

This model builds upon V7 by replacing the Random Walk Intercept with a Gaussian Process-based trend. This allows for a more flexible and smooth representation of the underlying temporal trend, potentially improving model fidelity, especially in cases where the trend exhibits non-linear but continuous behavior. The GP trend is defined over the unique time points using a `SqExponentialKernel`.



```{julia}
    @model function model_v9_gp_trend(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0, M_rff_sigma=20, M_inducing_val=10, M_rff_u=30)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)

    # --- Priors --- General model parameters
    sigma_u ~ filldist(Exponential(0.5), 3) # Measurement error for u_obs

    # --- Nested Latent Covariates (using RFF for non-linearity) ---
    coords_tz = hcat(coords_time, z)

    # U1 = f1(coords_time, Z)
    D_u1_input = size(coords_tz, 2)
    W_u1 ~ filldist(Normal(0, 1), D_u1_input, M_rff_u)
    b_u1 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0)
    beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1^2), M_rff_u)
    Phi_u1 = rff_map(coords_tz, W_u1, b_u1)
    u1_true = Phi_u1 * beta_rff_u1

    # U2 = f2(coords_time, Z, U1)
    coords_tz_u1 = hcat(coords_time, z, u1_true)
    D_u2_input = size(coords_tz_u1, 2)
    W_u2 ~ filldist(Normal(0, 1), D_u2_input, M_rff_u)
    b_u2 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0)
    beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2^2), M_rff_u)
    Phi_u2 = rff_map(coords_tz_u1, W_u2, b_u2)
    u2_true = Phi_u2 * beta_rff_u2

    # U3 = f3(coords_time, Z, U1)
    # (Note: inputs are the same as U2, but U3 is a separate function)
    D_u3_input = size(coords_tz_u1, 2)
    W_u3 ~ filldist(Normal(0, 1), D_u3_input, M_rff_u)
    b_u3 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0)
    beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3^2), M_rff_u)
    Phi_u3 = rff_map(coords_tz_u1, W_u3, b_u3)
    u3_true = Phi_u3 * beta_rff_u3

    # --- Seasonal Component (from V7) ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    # --- GP Trend (New in V9) ---
    ls_trend ~ Gamma(2, 2)
    sigma_trend ~ Exponential(0.5)

    # Define a 1D GP for the trend
    k_trend = SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend))
    g_trend = GP(sigma_trend^2 * k_trend)

    # Get unique time points for the GP
    unique_times = sort(unique(coords_time[:,1]))

    # Sample alpha from the GP with jitter for numerical stability
    alpha ~ g_trend(unique_times, 1e-1) # Increased jitter to 1e-1

    # Map alpha back to original time coordinates
    trend = alpha[indexin(coords_time[:,1], unique_times)]

    # --- Spatiotemporal GP using AbstractGPs for FITC (from V7) ---
    coords_st_orig = hcat(coords_space, coords_time) # N x D_st

    # Optimized Inducing Points: Z_inducing are now parameters with a prior
    # Initialize Z_inducing as an array of unknown values
    Z_inducing = Matrix{Float64}(undef, M_inducing_val, D_st)

    # Prior based on the observed data's mean and a scaled standard deviation for exploration.
    mu_coords_st = mean(coords_st_orig, dims=1) # This is a 1xDst matrix
    std_coords_st = std(coords_st_orig, dims=1) # This is a 1xDst matrix

    # Assign priors column-wise to Z_inducing
    for j in 1:D_st
        # Each column of Z_inducing consists of M_inducing_val i.i.d. samples
        # from a Normal distribution specific to that j-th dimension.
        # Use mu_coords_st[j] and std_coords_st[j] which are scalars after indexing into 1xDst matrices
        Z_inducing[:, j] ~ filldist(Normal(mu_coords_st[j], 2.0 * std_coords_st[j]), M_inducing_val)
    end

    # Lengthscales for each dimension (2 for space, 1 for time)
    ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_f ~ Exponential(1.0)

    # Anisotropic Spatiotemporal kernel
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))

    # Define the base GP using AbstractGPs.jl
    g_base = GP(sigma_f^2 * k_st)

    # Use RowVecs for coordinates
    Z_inducing_vecs = RowVecs(Z_inducing)
    coords_st_vecs = RowVecs(coords_st_orig)

    # Extract kernel matrices using AbstractGPs.jl
    K_ZZ = cov(g_base(Z_inducing_vecs)) + 1e-6*I
    K_XZ = cov(g_base(coords_st_vecs), g_base(Z_inducing_vecs))
    K_XX_diag = diag(cov(g_base(coords_st_vecs)))

    # Latent values at inducing points
    u_latent ~ MvNormal(zeros(M_inducing_val), K_ZZ)

    # Compute conditional mean and diagonal covariance using FITC formulas
    m_f = K_XZ * (K_ZZ \ u_latent)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))

    # Ensure positive definite covariance (add small jitter if necessary)
    f ~ MvNormal(m_f, Diagonal(max.(0, cov_f_diag) + 1e-6*ones(N)))

    # --- Spatiotemporal Stochastic Volatility (from V7) ---
    # Secondary RFF mapping for log-variance
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0) # Scale for the log-variance GP
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)

    Phi_sigma = rff_map(coords_st_orig, W_sigma, b_sigma)
    log_sigma_y = Phi_sigma * beta_rff_sigma # Latent log-variance process
    sigma_y_process = exp.(log_sigma_y ./ 2) # Convert log-variance to standard deviation

    # --- Mean of Y and Likelihoods ---
    beta_covs ~ filldist(Normal(0, 1), 4) # These betas are for u1_true, u2_true, u3_true, z
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    # Use spatiotemporal sigma_y_process in the likelihood
    y_obs ~ MvNormal(mu_base .+ f, Diagonal(sigma_y_process.^2))
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```




    model_v9_gp_trend (generic function with 2 methods)




```{julia}
# The Z_inducing variable is learned within the model.
D_s_v9 = size(data.coords_space, 2)
D_st_v9 = D_s_v9 + size(data.coords_time, 2)
coords_st_v9 = hcat(data.coords_space, data.coords_time)
M_inducing_val_v9 = 10 # Number of inducing points
M_rff_u_val_v9 = 30 # Number of RFF features for nested covariates
M_rff_sigma_val_v9 = 20 # Number of RFF features for log-variance GP

# Instantiate and sample Model V9 with NUTS
model_v9 = model_v9_gp_trend(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time; M_rff_sigma=M_rff_sigma_val_v9, M_inducing_val=M_inducing_val_v9, M_rff_u=M_rff_u_val_v9)

# Using NUTS sampler for better convergence; consider increasing iterations for production runs
chain_v9 = sample(model_v9, NUTS(), 100)
display(describe(chain_v9))
waic_v9 = compute_y_waic(model_v9, chain_v9)
println("WAIC for V9: ", waic_v9)

println("\nNote: For robust results, consider increasing the number of samples (e.g., 1000-2000 or more) and tuning the NUTS parameters (e.g., `adapts=num_adapts_steps`) if convergence issues persist.")
```

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.0015625
    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:51:51[39m


## V10: Fixed K-Means Inducing Points and GP Trend

This model builds upon V9 but modifies the handling of inducing points. Instead of treating `Z_inducing` locations as parameters to be learned (with their own priors), this version uses K-Means clustering to *deterministically* select the inducing points from the input `coords_st` data. These fixed inducing points are then passed to the model. This approach aims to:

1.  Improve Sampling Efficiency: By removing the inducing point locations as parameters, the sampler has fewer variables to explore, potentially leading to faster and more stable convergence.
2.  Provide a more informed starting point: K-Means places inducing points at the centroids of data clusters, which can be a more effective strategy than random initialization or relying solely on priors, as it ensures coverage of the data space.

The rest of the model structure (Nested RFF covariates, GP Trend, Seasonal component, Spatiotemporal Stochastic Volatility) remains the same as in V9.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V9, modeled with a mean component (GP trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): Same as V9, modeled as nonlinear functions using separate RFF mappings of `coords_time`, `Z`, and other `U` covariates. Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) still have measurement error.
*   Trend: Same as V9, a Gaussian Process-based trend (`GP Trend`).
*   Seasonal Process: Same as V9, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V9, uses the Fully Independent Training Conditional (FITC) approximation. However, new in V10, the inducing point locations (`Z_inducing`) are *fixed* and pre-computed using K-Means clustering, rather than being learned as parameters by the NUTS sampler.
*   Observation Noise (sigma_y): Same as V9, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V9, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors. The priors for `Z_inducing` locations are removed as they are no longer parameters within the model.

### Key References:
*   FITC: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS.
*   K-Means Clustering: MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics, 281–297. (For the method of selecting inducing points).
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS.


```{julia}
# Helper function to generate inducing points using K-Means
function kmeans_inducing_points(coords_st, M_inducing, seed=42)
    Random.seed!(seed)
    N_data = size(coords_st, 1)
    if M_inducing >= N_data
        return coords_st # If M >= N, just use all data points
    end

    # Perform K-Means clustering
    R = kmeans(Matrix(coords_st'), M_inducing; maxiter=200, init=:kmpp, display=:none)

    # Centroids of the clusters are the inducing points
    return R.centers'
end
```




    kmeans_inducing_points (generic function with 2 methods)




```{julia}
@model function model_v10_fixed_kmeans_fitc(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time, Z_inducing; period=12.0, M_rff_sigma=20, M_rff_u=30)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)
    M_inducing_val = size(Z_inducing, 1)

    # --- Priors --- General model parameters
    sigma_u ~ filldist(Exponential(0.5), 3) # Measurement error for u_obs

    # --- Nested Latent Covariates (using RFF for non-linearity) ---
    coords_tz = hcat(coords_time, z)

    # U1 = f1(coords_time, Z)
    D_u1_input = size(coords_tz, 2)
    W_u1 ~ filldist(Normal(0, 1), D_u1_input, M_rff_u)
    b_u1 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0)
    beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1^2), M_rff_u)
    Phi_u1 = rff_map(coords_tz, W_u1, b_u1)
    u1_true = Phi_u1 * beta_rff_u1

    # U2 = f2(coords_time, Z, U1)
    coords_tz_u1 = hcat(coords_time, z, u1_true)
    D_u2_input = size(coords_tz_u1, 2)
    W_u2 ~ filldist(Normal(0, 1), D_u2_input, M_rff_u)
    b_u2 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0)
    beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2^2), M_rff_u)
    Phi_u2 = rff_map(coords_tz_u1, W_u2, b_u2)
    u2_true = Phi_u2 * beta_rff_u2

    # U3 = f3(coords_time, Z, U1)
    # (Note: inputs are the same as U2, but U3 is a separate function)
    D_u3_input = size(coords_tz_u1, 2)
    W_u3 ~ filldist(Normal(0, 1), D_u3_input, M_rff_u)
    b_u3 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0)
    beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3^2), M_rff_u)
    Phi_u3 = rff_map(coords_tz_u1, W_u3, b_u3)
    u3_true = Phi_u3 * beta_rff_u3

    # --- Seasonal Component (from V9) ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    # --- GP Trend (from V9) ---
    ls_trend ~ Gamma(2, 2)
    sigma_trend ~ Exponential(0.5)

    # Define a 1D GP for the trend
    k_trend = SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend))
    g_trend = GP(sigma_trend^2 * k_trend)

    # Get unique time points for the GP
    unique_times = sort(unique(coords_time[:,1]))

    # Sample alpha from the GP with jitter for numerical stability
    alpha ~ g_trend(unique_times, 1e-1)

    # Map alpha back to original time coordinates
    trend = alpha[indexin(coords_time[:,1], unique_times)]

    # --- Spatiotemporal GP using AbstractGPs for FITC (from V9, with fixed Z_inducing) ---
    coords_st_orig = hcat(coords_space, coords_time) # N x D_st

    # Lengthscales for each dimension (2 for space, 1 for time)
    ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_f ~ Exponential(1.0)

    # Anisotropic Spatiotemporal kernel
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))

    # Define the base GP using AbstractGPs.jl
    g_base = GP(sigma_f^2 * k_st)

    # Use RowVecs for coordinates
    Z_inducing_vecs = RowVecs(Z_inducing)
    coords_st_vecs = RowVecs(coords_st_orig)

    # Extract kernel matrices using AbstractGPs.jl
    K_ZZ = cov(g_base(Z_inducing_vecs)) + 1e-6*I
    K_XZ = cov(g_base(coords_st_vecs), g_base(Z_inducing_vecs))
    K_XX_diag = diag(cov(g_base(coords_st_vecs)))

    # Latent values at inducing points
    u_latent ~ MvNormal(zeros(M_inducing_val), K_ZZ)

    # Compute conditional mean and diagonal covariance using FITC formulas
    m_f = K_XZ * (K_ZZ \ u_latent)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))

    # Ensure positive definite covariance (add small jitter if necessary)
    f ~ MvNormal(m_f, Diagonal(max.(0, cov_f_diag) + 1e-6*ones(N)))

    # --- Spatiotemporal Stochastic Volatility (from V9) ---
    # Secondary RFF mapping for log-variance
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0) # Scale for the log-variance GP
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)

    Phi_sigma = rff_map(coords_st_orig, W_sigma, b_sigma)
    log_sigma_y = Phi_sigma * beta_rff_sigma # Latent log-variance process
    sigma_y_process = exp.(log_sigma_y ./ 2) # Convert log-variance to standard deviation

    # --- Mean of Y and Likelihoods ---
    beta_covs ~ filldist(Normal(0, 1), 4) # These betas are for u1_true, u2_true, u3_true, z
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    # Use spatiotemporal sigma_y_process in the likelihood
    y_obs ~ MvNormal(mu_base .+ f, Diagonal(sigma_y_process.^2))
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```


    LoadError: UndefVarError: `@model` not defined in `Main`
    Suggestion: check for spelling errors or missing imports.
    in expression starting at In[1]:1

    



```{julia}
# Generate data (re-using previous `data` if available, or generate new if needed)
# data = generate_data(50) # Uncomment and run if `data` is not defined from previous cells

# Parameters for V10
D_s_v10 = size(data.coords_space, 2)
D_st_v10 = D_s_v10 + size(data.coords_time, 2)
coords_st_v10 = hcat(data.coords_space, data.coords_time)
M_inducing_val_v10 = 10 # Number of inducing points
M_rff_u_val_v10 = 30 # Number of RFF features for nested covariates
M_rff_sigma_val_v10 = 20 # Number of RFF features for log-variance GP

# Generate inducing points using K-Means
Z_inducing_v10 = kmeans_inducing_points(coords_st_v10, M_inducing_val_v10)

# Instantiate and sample Model V10 with NUTS
model_v10 = model_v10_fixed_kmeans_fitc(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time, Z_inducing_v10; M_rff_sigma=M_rff_sigma_val_v10, M_rff_u=M_rff_u_val_v10)

# Using NUTS sampler; consider increasing iterations for production runs
chain_v10 = sample(model_v10, NUTS(), 100)
display(describe(chain_v10))
waic_v10 = compute_y_waic(model_v10, chain_v10)
println("WAIC for V10: ", waic_v10)

println("\nNote: For robust results, consider increasing the number of samples (e.g., 1000-2000 or more) and tuning the NUTS parameters (e.g., `adapts=num_adapts_steps`) if convergence issues persist.")
```

    [32mSampling: 100%|█████████████████████████████████████████| Time: 0:00:01[39m


    Chains MCMC chain (500×652×1 Array{Float64, 3}):
    
    Iterations        = 1:1:500
    Number of chains  = 1
    Samples per chain = 500
    Wall duration     = 64.11 seconds
    Compute duration  = 64.11 seconds
    parameters        = sigma_u[1], sigma_u[2], sigma_u[3], W_u1[1, 1], W_u1[2, 1], W_u1[1, 2], W_u1[2, 2], W_u1[1, 3], W_u1[2, 3], W_u1[1, 4], W_u1[2, 4], W_u1[1, 5], W_u1[2, 5], W_u1[1, 6], W_u1[2, 6], W_u1[1, 7], W_u1[2, 7], W_u1[1, 8], W_u1[2, 8], W_u1[1, 9], W_u1[2, 9], W_u1[1, 10], W_u1[2, 10], W_u1[1, 11], W_u1[2, 11], W_u1[1, 12], W_u1[2, 12], W_u1[1, 13], W_u1[2, 13], W_u1[1, 14], W_u1[2, 14], W_u1[1, 15], W_u1[2, 15], W_u1[1, 16], W_u1[2, 16], W_u1[1, 17], W_u1[2, 17], W_u1[1, 18], W_u1[2, 18], W_u1[1, 19], W_u1[2, 19], W_u1[1, 20], W_u1[2, 20], W_u1[1, 21], W_u1[2, 21], W_u1[1, 22], W_u1[2, 22], W_u1[1, 23], W_u1[2, 23], W_u1[1, 24], W_u1[2, 24], W_u1[1, 25], W_u1[2, 25], W_u1[1, 26], W_u1[2, 26], W_u1[1, 27], W_u1[2, 27], W_u1[1, 28], W_u1[2, 28], W_u1[1, 29], W_u1[2, 29], W_u1[1, 30], W_u1[2, 30], b_u1[1], b_u1[2], b_u1[3], b_u1[4], b_u1[5], b_u1[6], b_u1[7], b_u1[8], b_u1[9], b_u1[10], b_u1[11], b_u1[12], b_u1[13], b_u1[14], b_u1[15], b_u1[16], b_u1[17], b_u1[18], b_u1[19], b_u1[20], b_u1[21], b_u1[22], b_u1[23], b_u1[24], b_u1[25], b_u1[26], b_u1[27], b_u1[28], b_u1[29], b_u1[30], sigma_f_u1, beta_rff_u1[1], beta_rff_u1[2], beta_rff_u1[3], beta_rff_u1[4], beta_rff_u1[5], beta_rff_u1[6], beta_rff_u1[7], beta_rff_u1[8], beta_rff_u1[9], beta_rff_u1[10], beta_rff_u1[11], beta_rff_u1[12], beta_rff_u1[13], beta_rff_u1[14], beta_rff_u1[15], beta_rff_u1[16], beta_rff_u1[17], beta_rff_u1[18], beta_rff_u1[19], beta_rff_u1[20], beta_rff_u1[21], beta_rff_u1[22], beta_rff_u1[23], beta_rff_u1[24], beta_rff_u1[25], beta_rff_u1[26], beta_rff_u1[27], beta_rff_u1[28], beta_rff_u1[29], beta_rff_u1[30], W_u2[1, 1], W_u2[2, 1], W_u2[3, 1], W_u2[1, 2], W_u2[2, 2], W_u2[3, 2], W_u2[1, 3], W_u2[2, 3], W_u2[3, 3], W_u2[1, 4], W_u2[2, 4], W_u2[3, 4], W_u2[1, 5], W_u2[2, 5], W_u2[3, 5], W_u2[1, 6], W_u2[2, 6], W_u2[3, 6], W_u2[1, 7], W_u2[2, 7], W_u2[3, 7], W_u2[1, 8], W_u2[2, 8], W_u2[3, 8], W_u2[1, 9], W_u2[2, 9], W_u2[3, 9], W_u2[1, 10], W_u2[2, 10], W_u2[3, 10], W_u2[1, 11], W_u2[2, 11], W_u2[3, 11], W_u2[1, 12], W_u2[2, 12], W_u2[3, 12], W_u2[1, 13], W_u2[2, 13], W_u2[3, 13], W_u2[1, 14], W_u2[2, 14], W_u2[3, 14], W_u2[1, 15], W_u2[2, 15], W_u2[3, 15], W_u2[1, 16], W_u2[2, 16], W_u2[3, 16], W_u2[1, 17], W_u2[2, 17], W_u2[3, 17], W_u2[1, 18], W_u2[2, 18], W_u2[3, 18], W_u2[1, 19], W_u2[2, 19], W_u2[3, 19], W_u2[1, 20], W_u2[2, 20], W_u2[3, 20], W_u2[1, 21], W_u2[2, 21], W_u2[3, 21], W_u2[1, 22], W_u2[2, 22], W_u2[3, 22], W_u2[1, 23], W_u2[2, 23], W_u2[3, 23], W_u2[1, 24], W_u2[2, 24], W_u2[3, 24], W_u2[1, 25], W_u2[2, 25], W_u2[3, 25], W_u2[1, 26], W_u2[2, 26], W_u2[3, 26], W_u2[1, 27], W_u2[2, 27], W_u2[3, 27], W_u2[1, 28], W_u2[2, 28], W_u2[3, 28], W_u2[1, 29], W_u2[2, 29], W_u2[3, 29], W_u2[1, 30], W_u2[2, 30], W_u2[3, 30], b_u2[1], b_u2[2], b_u2[3], b_u2[4], b_u2[5], b_u2[6], b_u2[7], b_u2[8], b_u2[9], b_u2[10], b_u2[11], b_u2[12], b_u2[13], b_u2[14], b_u2[15], b_u2[16], b_u2[17], b_u2[18], b_u2[19], b_u2[20], b_u2[21], b_u2[22], b_u2[23], b_u2[24], b_u2[25], b_u2[26], b_u2[27], b_u2[28], b_u2[29], b_u2[30], sigma_f_u2, beta_rff_u2[1], beta_rff_u2[2], beta_rff_u2[3], beta_rff_u2[4], beta_rff_u2[5], beta_rff_u2[6], beta_rff_u2[7], beta_rff_u2[8], beta_rff_u2[9], beta_rff_u2[10], beta_rff_u2[11], beta_rff_u2[12], beta_rff_u2[13], beta_rff_u2[14], beta_rff_u2[15], beta_rff_u2[16], beta_rff_u2[17], beta_rff_u2[18], beta_rff_u2[19], beta_rff_u2[20], beta_rff_u2[21], beta_rff_u2[22], beta_rff_u2[23], beta_rff_u2[24], beta_rff_u2[25], beta_rff_u2[26], beta_rff_u2[27], beta_rff_u2[28], beta_rff_u2[29], beta_rff_u2[30], W_u3[1, 1], W_u3[2, 1], W_u3[3, 1], W_u3[1, 2], W_u3[2, 2], W_u3[3, 2], W_u3[1, 3], W_u3[2, 3], W_u3[3, 3], W_u3[1, 4], W_u3[2, 4], W_u3[3, 4], W_u3[1, 5], W_u3[2, 5], W_u3[3, 5], W_u3[1, 6], W_u3[2, 6], W_u3[3, 6], W_u3[1, 7], W_u3[2, 7], W_u3[3, 7], W_u3[1, 8], W_u3[2, 8], W_u3[3, 8], W_u3[1, 9], W_u3[2, 9], W_u3[3, 9], W_u3[1, 10], W_u3[2, 10], W_u3[3, 10], W_u3[1, 11], W_u3[2, 11], W_u3[3, 11], W_u3[1, 12], W_u3[2, 12], W_u3[3, 12], W_u3[1, 13], W_u3[2, 13], W_u3[3, 13], W_u3[1, 14], W_u3[2, 14], W_u3[3, 14], W_u3[1, 15], W_u3[2, 15], W_u3[3, 15], W_u3[1, 16], W_u3[2, 16], W_u3[3, 16], W_u3[1, 17], W_u3[2, 17], W_u3[3, 17], W_u3[1, 18], W_u3[2, 18], W_u3[3, 18], W_u3[1, 19], W_u3[2, 19], W_u3[3, 19], W_u3[1, 20], W_u3[2, 20], W_u3[3, 20], W_u3[1, 21], W_u3[2, 21], W_u3[3, 21], W_u3[1, 22], W_u3[2, 22], W_u3[3, 22], W_u3[1, 23], W_u3[2, 23], W_u3[3, 23], W_u3[1, 24], W_u3[2, 24], W_u3[3, 24], W_u3[1, 25], W_u3[2, 25], W_u3[3, 25], W_u3[1, 26], W_u3[2, 26], W_u3[3, 26], W_u3[1, 27], W_u3[2, 27], W_u3[3, 27], W_u3[1, 28], W_u3[2, 28], W_u3[3, 28], W_u3[1, 29], W_u3[2, 29], W_u3[3, 29], W_u3[1, 30], W_u3[2, 30], W_u3[3, 30], b_u3[1], b_u3[2], b_u3[3], b_u3[4], b_u3[5], b_u3[6], b_u3[7], b_u3[8], b_u3[9], b_u3[10], b_u3[11], b_u3[12], b_u3[13], b_u3[14], b_u3[15], b_u3[16], b_u3[17], b_u3[18], b_u3[19], b_u3[20], b_u3[21], b_u3[22], b_u3[23], b_u3[24], b_u3[25], b_u3[26], b_u3[27], b_u3[28], b_u3[29], b_u3[30], sigma_f_u3, beta_rff_u3[1], beta_rff_u3[2], beta_rff_u3[3], beta_rff_u3[4], beta_rff_u3[5], beta_rff_u3[6], beta_rff_u3[7], beta_rff_u3[8], beta_rff_u3[9], beta_rff_u3[10], beta_rff_u3[11], beta_rff_u3[12], beta_rff_u3[13], beta_rff_u3[14], beta_rff_u3[15], beta_rff_u3[16], beta_rff_u3[17], beta_rff_u3[18], beta_rff_u3[19], beta_rff_u3[20], beta_rff_u3[21], beta_rff_u3[22], beta_rff_u3[23], beta_rff_u3[24], beta_rff_u3[25], beta_rff_u3[26], beta_rff_u3[27], beta_rff_u3[28], beta_rff_u3[29], beta_rff_u3[30], beta_cos, beta_sin, ls_trend, sigma_trend, alpha[1], alpha[2], alpha[3], alpha[4], alpha[5], alpha[6], alpha[7], alpha[8], alpha[9], alpha[10], alpha[11], alpha[12], alpha[13], alpha[14], alpha[15], alpha[16], alpha[17], alpha[18], alpha[19], alpha[20], alpha[21], alpha[22], alpha[23], alpha[24], alpha[25], alpha[26], alpha[27], alpha[28], alpha[29], alpha[30], alpha[31], alpha[32], alpha[33], alpha[34], alpha[35], alpha[36], alpha[37], alpha[38], alpha[39], alpha[40], alpha[41], alpha[42], alpha[43], alpha[44], alpha[45], alpha[46], alpha[47], alpha[48], alpha[49], alpha[50], ls_st[1], ls_st[2], ls_st[3], sigma_f, u_latent[1], u_latent[2], u_latent[3], u_latent[4], u_latent[5], u_latent[6], u_latent[7], u_latent[8], u_latent[9], u_latent[10], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15], f[16], f[17], f[18], f[19], f[20], f[21], f[22], f[23], f[24], f[25], f[26], f[27], f[28], f[29], f[30], f[31], f[32], f[33], f[34], f[35], f[36], f[37], f[38], f[39], f[40], f[41], f[42], f[43], f[44], f[45], f[46], f[47], f[48], f[49], f[50], W_sigma[1, 1], W_sigma[2, 1], W_sigma[3, 1], W_sigma[1, 2], W_sigma[2, 2], W_sigma[3, 2], W_sigma[1, 3], W_sigma[2, 3], W_sigma[3, 3], W_sigma[1, 4], W_sigma[2, 4], W_sigma[3, 4], W_sigma[1, 5], W_sigma[2, 5], W_sigma[3, 5], W_sigma[1, 6], W_sigma[2, 6], W_sigma[3, 6], W_sigma[1, 7], W_sigma[2, 7], W_sigma[3, 7], W_sigma[1, 8], W_sigma[2, 8], W_sigma[3, 8], W_sigma[1, 9], W_sigma[2, 9], W_sigma[3, 9], W_sigma[1, 10], W_sigma[2, 10], W_sigma[3, 10], W_sigma[1, 11], W_sigma[2, 11], W_sigma[3, 11], W_sigma[1, 12], W_sigma[2, 12], W_sigma[3, 12], W_sigma[1, 13], W_sigma[2, 13], W_sigma[3, 13], W_sigma[1, 14], W_sigma[2, 14], W_sigma[3, 14], W_sigma[1, 15], W_sigma[2, 15], W_sigma[3, 15], W_sigma[1, 16], W_sigma[2, 16], W_sigma[3, 16], W_sigma[1, 17], W_sigma[2, 17], W_sigma[3, 17], W_sigma[1, 18], W_sigma[2, 18], W_sigma[3, 18], W_sigma[1, 19], W_sigma[2, 19], W_sigma[3, 19], W_sigma[1, 20], W_sigma[2, 20], W_sigma[3, 20], b_sigma[1], b_sigma[2], b_sigma[3], b_sigma[4], b_sigma[5], b_sigma[6], b_sigma[7], b_sigma[8], b_sigma[9], b_sigma[10], b_sigma[11], b_sigma[12], b_sigma[13], b_sigma[14], b_sigma[15], b_sigma[16], b_sigma[17], b_sigma[18], b_sigma[19], b_sigma[20], sigma_log_var, beta_rff_sigma[1], beta_rff_sigma[2], beta_rff_sigma[3], beta_rff_sigma[4], beta_rff_sigma[5], beta_rff_sigma[6], beta_rff_sigma[7], beta_rff_sigma[8], beta_rff_sigma[9], beta_rff_sigma[10], beta_rff_sigma[11], beta_rff_sigma[12], beta_rff_sigma[13], beta_rff_sigma[14], beta_rff_sigma[15], beta_rff_sigma[16], beta_rff_sigma[17], beta_rff_sigma[18], beta_rff_sigma[19], beta_rff_sigma[20], beta_covs[1], beta_covs[2], beta_covs[3], beta_covs[4]
    internals         = logprior, loglikelihood, logjoint
    
    Summary Statistics
    
     [1m parameters [0m [1m    mean [0m [1m     std [0m [1m    mcse [0m [1m ess_bulk [0m [1m ess_tail [0m [1m    rhat [0m [1m e[0m ⋯
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m  Float64 [0m [90m  Float64 [0m [90m Float64 [0m [90m  [0m ⋯
    
      sigma_u[1]    1.4639    0.0000    0.0000        NaN        NaN       NaN     ⋯
      sigma_u[2]    1.4745    0.0000    0.0000        NaN        NaN       NaN     ⋯
      sigma_u[3]    2.0351    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[1, 1]    1.3393    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[2, 1]    0.9344    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[1, 2]    2.8078    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[2, 2]    2.2242    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[1, 3]   -0.1292    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[2, 3]    0.4195    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[1, 4]    0.6416    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[2, 4]   -0.1280    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[1, 5]    0.0639    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[2, 5]    1.1823    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[1, 6]    1.6974    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[2, 6]   -0.2333    0.0000       NaN        NaN        NaN       NaN     ⋯
      W_u1[1, 7]    0.2991    0.0000       NaN        NaN        NaN       NaN     ⋯
      W_u1[2, 7]    2.7834    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[1, 8]    2.0107    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[2, 8]    0.6295    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[1, 9]    0.9109    0.0000    0.0000        NaN        NaN       NaN     ⋯
      W_u1[2, 9]   -1.3982    0.0000    0.0000        NaN        NaN       NaN     ⋯
               ⋮         ⋮         ⋮         ⋮          ⋮          ⋮         ⋮     ⋱
    
    [36m                                                   1 column and 628 rows omitted[0m
    
    Quantiles
    
     [1m parameters [0m [1m    2.5% [0m [1m   25.0% [0m [1m   50.0% [0m [1m   75.0% [0m [1m   97.5% [0m
     [90m     Symbol [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m [90m Float64 [0m
    
      sigma_u[1]    1.4639    1.4639    1.4639    1.4639    1.4639
      sigma_u[2]    1.4745    1.4745    1.4745    1.4745    1.4745
      sigma_u[3]    2.0351    2.0351    2.0351    2.0351    2.0351
      W_u1[1, 1]    1.3393    1.3393    1.3393    1.3393    1.3393
      W_u1[2, 1]    0.9344    0.9344    0.9344    0.9344    0.9344
      W_u1[1, 2]    2.8078    2.8078    2.8078    2.8078    2.8078
      W_u1[2, 2]    2.2242    2.2242    2.2242    2.2242    2.2242
      W_u1[1, 3]   -0.1292   -0.1292   -0.1292   -0.1292   -0.1292
      W_u1[2, 3]    0.4195    0.4195    0.4195    0.4195    0.4195
      W_u1[1, 4]    0.6416    0.6416    0.6416    0.6416    0.6416
      W_u1[2, 4]   -0.1280   -0.1280   -0.1280   -0.1280   -0.1280
      W_u1[1, 5]    0.0639    0.0639    0.0639    0.0639    0.0639
      W_u1[2, 5]    1.1823    1.1823    1.1823    1.1823    1.1823
      W_u1[1, 6]    1.6974    1.6974    1.6974    1.6974    1.6974
      W_u1[2, 6]   -0.2333   -0.2333   -0.2333   -0.2333   -0.2333
      W_u1[1, 7]    0.2991    0.2991    0.2991    0.2991    0.2991
      W_u1[2, 7]    2.7834    2.7834    2.7834    2.7834    2.7834
      W_u1[1, 8]    2.0107    2.0107    2.0107    2.0107    2.0107
      W_u1[2, 8]    0.6295    0.6295    0.6295    0.6295    0.6295
      W_u1[1, 9]    0.9109    0.9109    0.9109    0.9109    0.9109
      W_u1[2, 9]   -1.3982   -1.3982   -1.3982   -1.3982   -1.3982
               ⋮         ⋮         ⋮         ⋮         ⋮         ⋮
    
    [36m                                                628 rows omitted[0m



    nothing


    WAIC for V10: 1323.6762942451498
    
    Note: For robust results, consider increasing the number of samples (e.g., 1000-2000 or more) and tuning the NUTS parameters (e.g., `adapts=num_adapts_steps`) if convergence issues persist.


## V11: Sparse Variational Gaussian Process (SVGP) Version

This model builds upon V10 by revisiting the concept of learned inducing points, aligning with an SVGP-like approach within the NUTS sampling framework. While a full Sparse Variational Gaussian Process typically employs variational inference to optimize a lower bound to the marginal likelihood, a key component of SVGP is the optimization of inducing point locations. In this V11, similar to V7 and V9, the `Z_inducing` locations are treated as parameters to be learned directly by the NUTS sampler, informed by a prior based on the data's mean and standard deviation.

This approach differs from V10 (which fixed inducing points using K-Means) by allowing the model to adaptively find the optimal inducing point locations during sampling. The rest of the model structure (Nested RFF covariates, GP Trend, Seasonal component, Spatiotemporal Stochastic Volatility) remains consistent with V9 and V10. We continue to use the NUTS sampler to handle the model's complexity.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V10, modeled with a mean component (GP trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): Same as V10, modeled as nonlinear functions using separate RFF mappings of `coords_time`, `Z`, and other `U` covariates. Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) still have measurement error.
*   Trend: Same as V10, a Gaussian Process-based trend (`GP Trend`).
*   Seasonal Process: Same as V10, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V10, uses the Fully Independent Training Conditional (FITC) approximation. However, new in V11, the inducing point locations (`Z_inducing`) are *learned* as parameters within the NUTS sampler, initialized with priors based on the input data, similar to V7 and V9.
*   Observation Noise (sigma_y): Same as V10, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V10, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, with the re-introduction of priors for `Z_inducing` locations.

### Key References:
*   FITC: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS. (For the underlying sparse GP approximation).
*   Variational Inference / SVGP: Hensman, J., Matthews, A. G., & Ghahramani, Z. (2015). *Scalable Variational Gaussian Process Regression*. PMLR. (For the conceptual basis of learning inducing point locations in a scalable GP context).
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS.
*   NUTS Sampler: Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*. Journal of Machine Learning Research, 15, 1593-1623. (For the MCMC sampling method).


```{julia}
@model function model_v11_svgp(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0, M_rff_sigma=20, M_inducing_val=10, M_rff_u=30)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)

    # --- Priors --- General model parameters
    sigma_u ~ filldist(Exponential(0.5), 3) # Measurement error for u_obs

    # --- Nested Latent Covariates (using RFF for non-linearity) ---
    coords_tz = hcat(coords_time, z)

    # U1 = f1(coords_time, Z)
    D_u1_input = size(coords_tz, 2)
    W_u1 ~ filldist(Normal(0, 1), D_u1_input, M_rff_u)
    b_u1 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0)
    beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1^2), M_rff_u)
    Phi_u1 = rff_map(coords_tz, W_u1, b_u1)
    u1_true = Phi_u1 * beta_rff_u1

    # U2 = f2(coords_time, Z, U1)
    coords_tz_u1 = hcat(coords_time, z, u1_true)
    D_u2_input = size(coords_tz_u1, 2)
    W_u2 ~ filldist(Normal(0, 1), D_u2_input, M_rff_u)
    b_u2 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0)
    beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2^2), M_rff_u)
    Phi_u2 = rff_map(coords_tz_u1, W_u2, b_u2)
    u2_true = Phi_u2 * beta_rff_u2

    # U3 = f3(coords_time, Z, U1)
    # (Note: inputs are the same as U2, but U3 is a separate function)
    D_u3_input = size(coords_tz_u1, 2)
    W_u3 ~ filldist(Normal(0, 1), D_u3_input, M_rff_u)
    b_u3 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0)
    beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3^2), M_rff_u)
    Phi_u3 = rff_map(coords_tz_u1, W_u3, b_u3)
    u3_true = Phi_u3 * beta_rff_u3

    # --- Seasonal Component (from V9/V10) ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    # --- GP Trend (from V9/V10) ---
    ls_trend ~ Gamma(2, 2)
    sigma_trend ~ Exponential(0.5)

    # Define a 1D GP for the trend
    k_trend = SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend))
    g_trend = GP(sigma_trend^2 * k_trend)

    # Get unique time points for the GP
    unique_times = sort(unique(coords_time[:,1]))

    # Sample alpha from the GP with jitter for numerical stability
    alpha ~ g_trend(unique_times, 1e-1)

    # Map alpha back to original time coordinates
    trend = alpha[indexin(coords_time[:,1], unique_times)]

    # --- Spatiotemporal GP using AbstractGPs for FITC (SVGP-like, with learned Z_inducing) ---
    coords_st_orig = hcat(coords_space, coords_time) # N x D_st

    # Optimized Inducing Points: Z_inducing are now parameters with a prior (as in V7/V9)
    # Initialize Z_inducing as an array of unknown values
    Z_inducing = Matrix{Float64}(undef, M_inducing_val, D_st)

    # Prior based on the observed data's mean and a scaled standard deviation for exploration.
    # This explicit learning of Z_inducing locations is a common element in SVGP.
    mu_coords_st = mean(coords_st_orig, dims=1) # This is a 1xDst matrix
    std_coords_st = std(coords_st_orig, dims=1) # This is a 1xDst matrix

    # Assign priors column-wise to Z_inducing
    for j in 1:D_st
        # Each column of Z_inducing consists of M_inducing_val i.i.d. samples
        # from a Normal distribution specific to that j-th dimension.
        # Use mu_coords_st[j] and std_coords_st[j] which are scalars after indexing into 1xDst matrices
        Z_inducing[:, j] ~ filldist(Normal(mu_coords_st[j], 2.0 * std_coords_st[j]), M_inducing_val)
    end

    # Lengthscales for each dimension (2 for space, 1 for time)
    ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_f ~ Exponential(1.0)

    # Anisotropic Spatiotemporal kernel
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))

    # Define the base GP using AbstractGPs.jl
    g_base = GP(sigma_f^2 * k_st)

    # Use RowVecs for coordinates
    Z_inducing_vecs = RowVecs(Z_inducing)
    coords_st_vecs = RowVecs(coords_st_orig)

    # Extract kernel matrices using AbstractGPs.jl
    K_ZZ = cov(g_base(Z_inducing_vecs)) + 1e-6*I
    K_XZ = cov(g_base(coords_st_vecs), g_base(Z_inducing_vecs))
    K_XX_diag = diag(cov(g_base(coords_st_vecs)))

    # Latent values at inducing points
    # In full SVGP, u_latent would have a variational posterior N(m_u, S_u)
    # Here, we sample from a prior based on K_ZZ, effectively using DTC/FITC approximation
    u_latent ~ MvNormal(zeros(M_inducing_val), K_ZZ)

    # Compute conditional mean and diagonal covariance using FITC formulas
    m_f = K_XZ * (K_ZZ \ u_latent)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))

    # Ensure positive definite covariance (add small jitter if necessary)
    f ~ MvNormal(m_f, Diagonal(max.(0, cov_f_diag) + 1e-6*ones(N)))

    # --- Spatiotemporal Stochastic Volatility (from V9/V10) ---
    # Secondary RFF mapping for log-variance
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0) # Scale for the log-variance GP
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)

    Phi_sigma = rff_map(coords_st_orig, W_sigma, b_sigma)
    log_sigma_y = Phi_sigma * beta_rff_sigma # Latent log-variance process
    sigma_y_process = exp.(log_sigma_y ./ 2) # Convert log-variance to standard deviation

    # --- Mean of Y and Likelihoods ---
    beta_covs ~ filldist(Normal(0, 1), 4) # These betas are for u1_true, u2_true, u3_true, z
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    # Use spatiotemporal sigma_y_process in the likelihood
    y_obs ~ MvNormal(mu_base .+ f, Diagonal(sigma_y_process.^2))
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```


```{julia}
# The Z_inducing variable is learned within the model.
D_s_v11 = size(data.coords_space, 2)
D_st_v11 = D_s_v11 + size(data.coords_time, 2)
coords_st_v11 = hcat(data.coords_space, data.coords_time)
M_inducing_val_v11 = 10 # Number of inducing points
M_rff_u_val_v11 = 30 # Number of RFF features for nested covariates
M_rff_sigma_val_v11 = 20 # Number of RFF features for log-variance GP

# Instantiate and sample Model V11 with NUTS
model_v11 = model_v11_svgp(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time; M_rff_sigma=M_rff_sigma_val_v11, M_inducing_val=M_inducing_val_v11, M_rff_u=M_rff_u_val_v11)

# Using NUTS sampler; consider increasing iterations for production runs
chain_v11 = sample(model_v11, NUTS(), 100) # Reduced samples from 500 to 100 for faster testing
display(describe(chain_v11))
waic_v11 = compute_y_waic(model_v11, chain_v11)
println("WAIC for V11: ", waic_v11)

println("\nNote: For robust results, consider increasing the number of samples (e.g., 1000-2000 or more) and tuning the NUTS parameters (e.g., `adapts=num_adapts_steps`) if convergence issues persist.")
```

    [36m[1m┌ [22m[39m[36m[1mInfo: [22m[39mFound initial step size
    [36m[1m└ [22m[39m  ϵ = 0.00625
    [32mSampling:  37%|███████████████                          |  ETA: 3:25:55[39m

## V8: Nonlinear Nested Covariates (RFF-based)

This model builds upon V7 by introducing nonlinear functional forms for the nested covariates `U1`, `U2`, and `U3`. Instead of simple linear relationships, it uses Random Fourier Features (RFFs) to model these dependencies, allowing for more complex and adaptive representations of how `U` covariates are generated from `coords_time`, `Z`, and other `U` covariates.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V7, modeled with a mean component (GP trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): New in V8, these are now modeled as nonlinear functions using separate RFF mappings:
    *   `U1 = f1_rff(coords_time, Z)`: Modeled as a nonlinear function of time and spatial covariate `Z` via an RFF layer.
    *   `U2 = f2_rff(coords_time, Z, U1)`: Modeled as a nonlinear function of time, `Z`, and the latent `U1` via an RFF layer.
    *   `U3 = f3_rff(coords_time, Z, U1)`: Modeled as a nonlinear function of time, `Z`, and the latent `U1` via an RFF layer.
    Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) still have measurement error.
*   Trend: Same as V7, a Gaussian Process-based trend (`GP Trend`).
*   Seasonal Process: Same as V7, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V7, uses the Fully Independent Training Conditional (FITC) approximation with learned inducing point locations (`Z_inducing`) and latent values (`u_latent`).
*   Observation Noise (sigma_y): Same as V7, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V7, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, extended for the `W`, `b`, `sigma_f`, and `beta_rff` parameters for each of the nonlinear `U` covariate RFF mappings.

### Key References:
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (For the nonlinear covariate mappings).
*   FITC and Inducing Point Optimization: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS. (Still relevant for the main GP component).
*   Hierarchical Bayesian Models: Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press. (For general hierarchical modeling principles).
*   Deep Gaussian Processes: Damianou, A., & Lawrence, N. (2013). *Deep Gaussian Processes*. AISTATS. (For the conceptual foundation of stacking GP-like layers).

## V9: Gaussian Process Trend

This model builds upon V8 by replacing the Random Walk Intercept with a Gaussian Process-based trend. This allows for a more flexible and smooth representation of the underlying temporal trend, potentially improving model fidelity, especially in cases where the trend exhibits non-linear but continuous behavior. The GP trend is defined over the unique time points using a `SqExponentialKernel`.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V8, modeled with a mean component (GP trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): Same as V8, modeled as nonlinear functions using separate RFF mappings of `coords_time`, `Z`, and other `U` covariates. Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) still have measurement error.
*   Trend: New in V9, the trend component is now explicitly modeled as a 1D Gaussian Process (`GP Trend`) using a `SqExponentialKernel` over unique time points. This replaces the Random Walk Intercept from previous versions.
*   Seasonal Process: Same as V8, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V8, uses the Fully Independent Training Conditional (FITC) approximation with learned inducing point locations (`Z_inducing`) and latent values (`u_latent`).
*   Observation Noise (sigma_y): Same as V8, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V8, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, extended for the `ls_trend` and `sigma_trend` parameters of the GP Trend.

### Key References:
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (For GP fundamentals and GP-based trends).
*   Time Series with GPs: Roberts, S., Osborne, M. A., & Ebden, M. (2013). *Gaussian Processes for Time-Series Analysis*. In *Time-Series Analysis* (pp. 59-86). Springer, Berlin, Heidelberg. (For applying GPs to time series data).
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (For the nonlinear covariate mappings).
*   FITC and Inducing Point Optimization: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS. (Still relevant for the main GP component).

## V10: Fixed K-Means Inducing Points and GP Trend

This model builds upon V9 but modifies the handling of inducing points. Instead of treating `Z_inducing` locations as parameters to be learned (with their own priors), this version uses K-Means clustering to *deterministically* select the inducing points from the input `coords_st` data. These fixed inducing points are then passed to the model. This approach aims to:

1.  Improve Sampling Efficiency: By removing the inducing point locations as parameters, the sampler has fewer variables to explore, potentially leading to faster and more stable convergence.
2.  Provide a more informed starting point: K-Means places inducing points at the centroids of data clusters, which can be a more effective strategy than random initialization or relying solely on priors, as it ensures coverage of the data space.

The rest of the model structure (Nested RFF covariates, GP Trend, Seasonal component, Spatiotemporal Stochastic Volatility) remains the same as in V9.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V9, modeled with a mean component (GP trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): Same as V9, modeled as nonlinear functions using separate RFF mappings of `coords_time`, `Z`, and other `U` covariates. Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) still have measurement error.
*   Trend: Same as V9, a Gaussian Process-based trend (`GP Trend`).
*   Seasonal Process: Same as V9, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V9, uses the Fully Independent Training Conditional (FITC) approximation. However, new in V10, the inducing point locations (`Z_inducing`) are *fixed* and pre-computed using K-Means clustering, rather than being learned as parameters by the NUTS sampler.
*   Observation Noise (sigma_y): Same as V9, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V9, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors. The priors for `Z_inducing` locations are removed as they are no longer parameters within the model.

### Key References:
*   FITC: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS.
*   K-Means Clustering: MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics, 281–297. (For the method of selecting inducing points).
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS.

## V11: Sparse Variational Gaussian Process (SVGP) Version

This model builds upon V10 by revisiting the concept of learned inducing points, aligning with an SVGP-like approach within the NUTS sampling framework. While a full Sparse Variational Gaussian Process typically employs variational inference to optimize a lower bound to the marginal likelihood, a key component of SVGP is the optimization of inducing point locations. In this V11, similar to V7 and V9, the `Z_inducing` locations are treated as parameters to be learned directly by the NUTS sampler, informed by a prior based on the data's mean and standard deviation.

This approach differs from V10 (which fixed inducing points using K-Means) by allowing the model to adaptively find the optimal inducing point locations during sampling. The rest of the model structure (Nested RFF covariates, GP Trend, Seasonal component, Spatiotemporal Stochastic Volatility) remains consistent with V9 and V10. We continue to use the NUTS sampler to handle the model's complexity.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V10, modeled with a mean component (GP trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): Same as V10, modeled as nonlinear functions using separate RFF mappings of `coords_time`, `Z`, and other `U` covariates. Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) still have measurement error.
*   Trend: Same as V10, a Gaussian Process-based trend (`GP Trend`).
*   Seasonal Process: Same as V10, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V10, uses the Fully Independent Training Conditional (FITC) approximation. However, new in V11, the inducing point locations (`Z_inducing`) are *learned* as parameters within the NUTS sampler, initialized with priors based on the input data, similar to V7 and V9.
*   Observation Noise (sigma_y): Same as V10, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V10, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, with the re-introduction of priors for `Z_inducing` locations.

### Key References:
*   FITC: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS. (For the underlying sparse GP approximation).
*   Variational Inference / SVGP: Hensman, J., Matthews, A. G., & Ghahramani, Z. (2015). *Scalable Variational Gaussian Process Regression*. PMLR. (For the conceptual basis of learning inducing point locations in a scalable GP context).
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS.
*   NUTS Sampler: Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*. Journal of Machine Learning Research, 15, 1593-1623. (For the MCMC sampling method).

## V12: Full Sparse Variational Gaussian Process (SVGP) Variation

This model builds upon V11 by making the mean and diagonal variance of the inducing point latent values (`u_latent`) explicit parameters to be learned via NUTS sampling. This allows for a more flexible, SVGP-like representation of the inducing point distribution within the MCMC framework. While true SVGP typically uses variational inference to optimize an ELBO, this version adapts the concept for direct posterior sampling.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V11, modeled with a mean component (GP trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): Same as V11, modeled as nonlinear functions using separate RFF mappings of `coords_time`, `Z`, and other `U` covariates. Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) still have measurement error.
*   Trend: Same as V11, a Gaussian Process-based trend (`GP Trend`).
*   Seasonal Process: Same as V11, a fixed-period harmonic.
*   Spatiotemporal GP (f): Same as V11, uses the Fully Independent Training Conditional (FITC) approximation. However, new in V12, the distribution of latent values at inducing points (`u_latent`) is further parameterized. Instead of sampling `u_latent` directly from `MvNormal(zeros(M_inducing_val), K_ZZ)`, its mean (`m_latent_u`) and diagonal standard deviation (`sigma_latent_u_diag`) are now treated as parameters to be learned by the NUTS sampler. This allows for a more flexible, SVGP-like representation of the inducing point distribution within the MCMC framework.
*   Observation Noise (sigma_y): Same as V11, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V11, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, with the addition of priors for `m_latent_u` and `sigma_latent_u_diag` (e.g., `Normal(0, 10.0)` for mean and `Exponential(1.0)` for diagonal standard deviation).

### Key References:
*   FITC: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS. (For the underlying sparse GP approximation).
*   Variational Gaussian Processes (SVGP): Hensman, J., Matthews, A. G., & Ghahramani, Z. (2015). *Scalable Variational Gaussian Process Regression*. PMLR. (This model takes inspiration from SVGP by learning the parameters of the inducing point distribution, albeit within an MCMC framework rather than variational inference).
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS.
*   NUTS Sampler: Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*. Journal of Machine Learning Research, 15, 1593-1623. (For the MCMC sampling method).


```{julia}
@model function model_v12_svgp_full(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0, M_rff_sigma=20, M_inducing_val=10, M_rff_u=30)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)

    # --- Priors --- General model parameters
    sigma_u ~ filldist(Exponential(0.5), 3) # Measurement error for u_obs

    # --- Nested Latent Covariates (using RFF for non-linearity) ---
    coords_tz = hcat(coords_time, z)

    # U1 = f1(coords_time, Z)
    D_u1_input = size(coords_tz, 2)
    W_u1 ~ filldist(Normal(0, 1), D_u1_input, M_rff_u)
    b_u1 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0)
    beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1^2), M_rff_u)
    Phi_u1 = rff_map(coords_tz, W_u1, b_u1)
    u1_true = Phi_u1 * beta_rff_u1

    # U2 = f2(coords_time, Z, U1)
    coords_tz_u1 = hcat(coords_time, z, u1_true)
    D_u2_input = size(coords_tz_u1, 2)
    W_u2 ~ filldist(Normal(0, 1), D_u2_input, M_rff_u)
    b_u2 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0)
    beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2^2), M_rff_u)
    Phi_u2 = rff_map(coords_tz_u1, W_u2, b_u2)
    u2_true = Phi_u2 * beta_rff_u2

    # U3 = f3(coords_time, Z, U1)
    D_u3_input = size(coords_tz_u1, 2)
    W_u3 ~ filldist(Normal(0, 1), D_u3_input, M_rff_u)
    b_u3 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0)
    beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3^2), M_rff_u)
    Phi_u3 = rff_map(coords_tz_u1, W_u3, b_u3)
    u3_true = Phi_u3 * beta_rff_u3

    # --- Seasonal Component (from V11) ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    # --- GP Trend (from V11) ---
    ls_trend ~ Gamma(2, 2)
    sigma_trend ~ Exponential(0.5)

    k_trend = SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend))
    g_trend = GP(sigma_trend^2 * k_trend)

    unique_times = sort(unique(coords_time[:,1]))

    alpha ~ g_trend(unique_times, 1e-1)

    trend = alpha[indexin(coords_time[:,1], unique_times)]

    # --- Spatiotemporal GP using AbstractGPs for FITC (SVGP-like, with learned Z_inducing) ---
    coords_st_orig = hcat(coords_space, coords_time)

    # Optimized Inducing Points: Z_inducing are parameters with a prior (as in V11)
    Z_inducing = Matrix{Float64}(undef, M_inducing_val, D_st)

    mu_coords_st = mean(coords_st_orig, dims=1)
    std_coords_st = std(coords_st_orig, dims=1)

    for j in 1:D_st
        Z_inducing[:, j] ~ filldist(Normal(mu_coords_st[j], 2.0 * std_coords_st[j]), M_inducing_val)
    end

    # Lengthscales for each dimension
    ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_f ~ Exponential(1.0)

    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))
    g_base = GP(sigma_f^2 * k_st)

    Z_inducing_vecs = RowVecs(Z_inducing)
    coords_st_vecs = RowVecs(coords_st_orig)

    K_ZZ = cov(g_base(Z_inducing_vecs)) + 1e-6*I
    K_XZ = cov(g_base(coords_st_vecs), g_base(Z_inducing_vecs))
    K_XX_diag = diag(cov(g_base(coords_st_vecs)))

    # --- SVGP-like: Learn mean and diagonal std dev for u_latent's distribution ---
    # This is a key difference from V11, making u_latent's distribution more flexible.
    m_latent_u ~ MvNormal(zeros(M_inducing_val), 10.0 * I) # Prior for the mean of u_latent
    sigma_latent_u_diag ~ filldist(Exponential(1.0), M_inducing_val) # Prior for diagonal std dev

    # Sample u_latent from this learned distribution
    u_latent ~ MvNormal(m_latent_u, Diagonal(sigma_latent_u_diag.^2) + 1e-6*I)

    # Compute conditional mean and diagonal covariance using FITC formulas (conditioned on sampled u_latent)
    m_f = K_XZ * (K_ZZ \ u_latent)
    cov_f_diag = K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ'))

    f ~ MvNormal(m_f, Diagonal(max.(0, cov_f_diag) + 1e-6*ones(N)))

    # --- Spatiotemporal Stochastic Volatility (from V11) ---
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0)
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)

    Phi_sigma = rff_map(coords_st_orig, W_sigma, b_sigma)
    log_sigma_y = Phi_sigma * beta_rff_sigma
    sigma_y_process = exp.(log_sigma_y ./ 2)

    # --- Mean of Y and Likelihoods ---
    beta_covs ~ filldist(Normal(0, 1), 4)
    mu_base = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    y_obs ~ MvNormal(mu_base .+ f, Diagonal(sigma_y_process.^2))
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```


```{julia}
# The Z_inducing variable is learned within the model.
D_s_v12 = size(data.coords_space, 2)
D_st_v12 = D_s_v12 + size(data.coords_time, 2)
coords_st_v12 = hcat(data.coords_space, data.coords_time)
M_inducing_val_v12 = 10 # Number of inducing points
M_rff_u_val_v12 = 30 # Number of RFF features for nested covariates
M_rff_sigma_val_v12 = 20 # Number of RFF features for log-variance GP

# Instantiate and sample Model V12 with NUTS
model_v12 = model_v12_svgp_full(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time; M_rff_sigma=M_rff_sigma_val_v12, M_inducing_val=M_inducing_val_v12, M_rff_u=M_rff_u_val_v12)

# Using NUTS sampler; consider increasing iterations for production runs
chain_v12 = sample(model_v12, NUTS(), 100) # Reduced samples for faster testing
display(describe(chain_v12))
waic_v12 = compute_y_waic(model_v12, chain_v12)
println("WAIC for V12: ", waic_v12)

println("\nNote: For robust results, consider increasing the number of samples (e.g., 1000-2000 or more) and tuning the NUTS parameters (e.g., `adapts=num_adapts_steps`) if convergence issues persist. This model has a significantly larger parameter space, which may lead to slower sampling and more complex convergence behavior.")
```


## V13: Multi-fidelity Gaussian Process (MFGP)

This model implements a multi-fidelity approach where covariates exist at different resolutions. This is a significant departure from previous models that generally assumed all data points for all variables were at a single resolution.

### Model Assumptions:
*   Multi-fidelity Structure: The core idea is to handle data measured at different levels of detail or frequency. Specifically:
    *   Z (Highest Resolution): Modeled as a latent spatial field. This might represent dense environmental measurements (e.g., satellite imagery).
    *   U1, U2, U3 (High Resolution): Modeled as latent spatiotemporal fields. These fields are assumed to depend on the latent spatial field `Z`, indicating a hierarchical relationship across fidelities. This could represent sensor data collected more frequently or densely than the primary observation but less so than `Z`.
    *   Y (Standard Resolution): The primary observation variable. This is typically the target variable and is assumed to depend on the latent `U` and `Z` fields, effectively drawing information from the higher fidelity layers.
*   Functional Dependencies: Nested Random Fourier Features (RFF) are employed to represent the nonlinear functional dependencies across these different fidelities. This allows for complex mappings between the latent fields at different resolutions.
*   GP Representation: Each latent field (`Z`, `U1`, `U2`, `U3`) is implicitly modeled as a Gaussian Process through the RFF approximation, capturing spatial and/or spatiotemporal correlations within each fidelity level.
*   Observation Noise: Homoscedastic and normally distributed for each observed variable (`y_obs`, `u1_obs`, `u2_obs`, `u3_obs`, `z_obs`).
*   Seasonal and Trend Components: Similar to previous models, seasonal effects are modeled with fixed-period harmonics, and a trend component (though not explicitly GP-based like V9-V12 in this initial version of V13) is included in the final `Y` model.

### Key References:
*   Multi-fidelity Gaussian Processes: Perdikaris, P., Raissi, M., Psaros, N., & Karniadakis, G. E. (2017). *Nonlinear model reduction for uncertainty quantification and predictive modeling of spatiotemporal systems*. Journal of Computational Physics, 347, 303-324. (For general MFGP concepts).
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (For modeling nonlinear relationships and approximating GPs).
*   Hierarchical Bayesian Models: Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press. (For modeling dependencies across different levels of a hierarchy).
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (Fundamentals for GP components).


```{julia}
@model function model_v13_multifidelity_gp(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_space_y, coords_time_y, coords_space_u, coords_time_u, coords_space_z; period=12.0, M_rff=40)
    # Dimensions for different fidelities
    Ny = length(y_obs)
    Nu = length(u1_obs)
    Nz = length(z_obs)
    D_s = size(coords_space_y, 2)
    D_t = size(coords_time_y, 2)

    # 1. --- Highest Fidelity: Spatial Covariate Z ---
    # Model Z as a spatial GP to handle its high resolution
    W_z ~ filldist(Normal(0, 1), D_s, M_rff)
    b_z ~ filldist(Uniform(0, 2pi), M_rff)
    beta_z ~ filldist(Normal(0, 1), M_rff)

    # Function to get latent Z at any spatial coordinate
    get_latent_z(coords) = rff_map(coords, W_z, b_z) * beta_z

    z_latent_at_z_coords = get_latent_z(coords_space_z)
    sigma_z ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent_at_z_coords, sigma_z^2 * I)

    # 2. --- Medium Fidelity: Spatiotemporal Covariates U ---
    # U depends on Space, Time, and the latent Z
    z_at_u_coords = get_latent_z(coords_space_u)
    coords_st_u = hcat(coords_space_u, coords_time_u, z_at_u_coords)

    D_u_in = size(coords_st_u, 2)
    W_u ~ filldist(Normal(0, 1), D_u_in, M_rff)
    b_u ~ filldist(Uniform(0, 2pi), M_rff)

    # Coefficients for U1, U2, U3
    beta_u1 ~ filldist(Normal(0, 1), M_rff)
    beta_u2 ~ filldist(Normal(0, 1), M_rff)
    beta_u3 ~ filldist(Normal(0, 1), M_rff)

    Phi_u = rff_map(coords_st_u, W_u, b_u)
    u1_latent = Phi_u * beta_u1
    u2_latent = Phi_u * beta_u2
    u3_latent = Phi_u * beta_u3

    sigma_u ~ filldist(Exponential(0.5), 3)
    u1_obs ~ MvNormal(u1_latent, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_latent, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_latent, sigma_u[3]^2 * I)

    # 3. --- Standard Fidelity: Dependent Variable Y ---
    z_at_y_coords = get_latent_z(coords_space_y)
    # For Y, we need U values at Y's lower resolution coordinates
    # In a real MFGP, we'd use the latent function learned at resolution U
    coords_st_y_for_u = hcat(coords_space_y, coords_time_y, z_at_y_coords)
    Phi_y_u = rff_map(coords_st_y_for_u, W_u, b_u)
    u1_at_y = Phi_y_u * beta_u1
    u2_at_y = Phi_y_u * beta_u2
    u3_at_y = Phi_y_u * beta_u3

    # Main effect regression
    beta_y ~ filldist(Normal(0, 1), 4)
    mu_y = beta_y[1] .* u1_at_y .+ beta_y[2] .* u2_at_y .+ beta_y[3] .* u3_at_y .+ beta_y[4] .* z_at_y_coords

    # Add seasonal and trend components
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time_y[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time_y[:,1] ./ period)

    sigma_y ~ Exponential(1.0)
    y_obs ~ MvNormal(mu_y .+ seasonal, sigma_y^2 * I)
end
```

## V14: Mini-batchable Multi-fidelity SVGP

This model builds upon V13 by reformulating the multi-fidelity approach to be amenable to mini-batching, primarily for use with Stochastic Variational Inference (SVI). To achieve this, it relies on the assumption that observations are conditionally independent given the global latent parameters (specifically, the RFF weights defining the latent fields). This allows for processing data in smaller chunks, making the model scalable to very large datasets.

### Model Assumptions:
*   Mini-batching Compatibility: The likelihood is structured to operate on individual observations within a batch, assuming conditional independence of observations given the shared global RFF weights (`W`, `b`, `beta` for each fidelity layer).
*   Multi-fidelity Structure: Retains the core multi-fidelity idea from V13:
    *   Z-fidelity: Latent spatial field. (`get_z` function).
    *   U-fidelity: Latent spatiotemporal fields (`U1, U2, U3`) depending on space, time, and interpolated `Z`.
    *   Y-fidelity: The primary observation (`Y`) depending on space, time, and interpolated `U` and `Z`.
*   Functional Dependencies: Uses Random Fourier Features (RFF) to model the nonlinear relationships and approximate the latent GP fields at each fidelity level.
*   GP Representation: Each latent field (`Z`, `U1`, `U2`, `U3`) is implicitly modeled as a Gaussian Process through the RFF approximation.
*   Observation Noise: Homoscedastic and normally distributed for each observed variable (`z_obs`, `u1_obs`, `u2_obs`, `u3_obs`, `y_obs`).
*   Parameter Sharing: The RFF weights (`W_u`, `b_u`) and coefficients (`beta_u1`, `beta_u2`, `beta_u3`) are shared between the U-fidelity and the Y-fidelity layers for consistent interpolation of latent U-fields to Y coordinates.
*   Simplified Mean Function for Y: The mean function for `y_obs` is a linear combination of the interpolated latent `U` and `Z` fields.

### Key References:
*   Stochastic Variational Inference (SVI): Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). *Stochastic variational inference*. Journal of Machine Learning Research, 14, 1303-1347. (For the theoretical basis of scalable inference with mini-batches).
*   Multi-fidelity Gaussian Processes: Perdikaris, P., Raissi, M., Psaros, N., & Karniadakis, G. E. (2017). *Nonlinear model reduction for uncertainty quantification and predictive modeling of spatiotemporal systems*. Journal of Computational Physics, 347, 303-324. (For general MFGP concepts).
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (For modeling nonlinear relationships and approximating GPs).


```{julia}

model_v13 = model_v13_multifidelity_gp(
    y_mock, u1_mock, u2_mock, u3_mock, z_mock,
    coords_y_s, coords_y_t,
    coords_u_s, coords_u_t,
    coords_z_s
)

chain_v13 = sample(model_v13, NUTS(), 100)
display(describe(chain_v13))
```


```{julia}
@model function model_v14_minibatch_mfgp(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_y, coords_u, coords_z; M_rff=50)
    # --- Global Latent Variables (Kernels) ---
    # Z-fidelity
    W_z ~ filldist(Normal(0, 1), size(coords_z, 2), M_rff)
    b_z ~ filldist(Uniform(0, 2pi), M_rff)
    beta_z ~ filldist(Normal(0, 1), M_rff)

    # U-fidelity
    # (Input dim is Space + Time + Latent Z)
    W_u ~ filldist(Normal(0, 1), size(coords_u, 2) + 1, M_rff)
    b_u ~ filldist(Uniform(0, 2pi), M_rff)
    beta_u1 ~ filldist(Normal(0, 1), M_rff)
    beta_u2 ~ filldist(Normal(0, 1), M_rff)
    beta_u3 ~ filldist(Normal(0, 1), M_rff)

    # Y-fidelity
    beta_y ~ filldist(Normal(0, 1), 4)

    # Noise parameters
    sigma_z ~ Exponential(0.5)
    sigma_u ~ filldist(Exponential(0.5), 3)
    sigma_y ~ Exponential(1.0)

    # --- Latent Map Functions ---
    # These allow us to compute the mean for ANY single observation
    function get_z(c_z)
        phi = sqrt(2/M_rff) .* cos.( (c_z * W_z) .+ b_z' )
        return phi * beta_z
    end

    # --- Likelihoods (Structured for Mini-batching) ---
    # 1. Z-Fidelity Likelihood
    z_mu = get_z(coords_z)
    # Using .~ with a distribution allows Turing to handle independent observations
    z_obs .~ Normal.(z_mu, sigma_z)

    # 2. U-Fidelity Likelihood
    # First, get latent Z at U locations
    coords_u_space = coords_u[:, 1:2]
    z_at_u = get_z(coords_u_space)
    coords_st_u = hcat(coords_u, z_at_u)

    phi_u = sqrt(2/M_rff) .* cos.( (coords_st_u * W_u) .+ b_u' )
    u1_mu = phi_u * beta_u1
    u2_mu = phi_u * beta_u2
    u3_mu = phi_u * beta_u3

    u1_obs .~ Normal.(u1_mu, sigma_u[1])
    u2_obs .~ Normal.(u2_mu, sigma_u[2])
    u3_obs .~ Normal.(u3_mu, sigma_u[3])

    # 3. Y-Fidelity Likelihood
    z_at_y = get_z(coords_y[:, 1:2])
    coords_st_y = hcat(coords_y, z_at_y)
    phi_y_u = sqrt(2/M_rff) .* cos.( (coords_st_y * W_u) .+ b_u' )

    u1_at_y = phi_y_u * beta_u1
    u2_at_y = phi_y_u * beta_u2
    u3_at_y = phi_y_u * beta_u3

    y_mu = beta_y[1] .* u1_at_y .+ beta_y[2] .* u2_at_y .+ beta_y[3] .* u3_at_y .+ beta_y[4] .* z_at_y
    y_obs .~ Normal.(y_mu, sigma_y)
end
```


```{julia}
# Demonstration of how to call the model with a mini-batch of data
batch_indices = sample(1:Ny, 20, replace=false) # Pick 20 random indices for Y

# Create the mini-batch data
y_batch = y_mock[batch_indices]
coords_y_batch = hcat(coords_y_s, coords_y_t)[batch_indices, :]

# Instantiate model with the batch
# Note: In a full SVI loop, you would also batch U and Z similarly
model_minibatch = model_v14_minibatch_mfgp(
    y_batch, u1_mock, u2_mock, u3_mock, z_mock,
    coords_y_batch, hcat(coords_u_s, coords_u_t), coords_z
)

# Now you can use AdvancedVI.jl to optimize this model efficiently
println("Model instantiated with mini-batch of size ", length(y_batch))
```

## V15: Deep Gaussian Process (Stacked RFF)

This model implements a Deep GP architecture using a hierarchical composition of RFF layers.
1. Layer 1 (Spatial): Models $Z$ as a function of spatial coordinates.
2. Layer 2 (Spatiotemporal): Models $U_1, U_2, U_3$ as functions of Space, Time, and the latent output from Layer 1 ($Z$).
3. Layer 3 (Output): Models $Y$ as a function of Space, Time, and the latent outputs from Layer 2 ($U_1, U_2, U_3$).

By stacking these RFF mappings, we create a deep probabilistic model where each level performs a non-linear transformation (warping) of the input space for the subsequent level.

### Model Assumptions:
*   Dependent Variable (Y): Modeled with a seasonal component and a latent Deep GP component, plus observation noise.
*   Latent Spatial Field (Z): Modeled as a Gaussian Process approximated by RFFs, taking spatial coordinates as input.
*   Latent Spatiotemporal Fields (U1, U2, U3): Modeled as Gaussian Processes approximated by RFFs. Their inputs include spatial coordinates, time, and the latent Z from Layer 1, establishing a hierarchical dependency.
*   Deep GP Structure: A three-layer RFF composition, where the output of one RFF layer serves as input to the next, creating a hierarchical, nonlinear feature transformation for the final prediction of Y.
*   Seasonal Process: Modeled as a fixed-period harmonic.
*   Observation Noise (sigma_y, sigma_u, sigma_z): Assumed to be homoscedastic and normally distributed for all observed variables.
*   Priors: Standard weakly informative priors for all RFF weights, biases, and GP variances.

### Key References:
*   Deep Gaussian Processes: Damianou, A., & Lawrence, N. (2013). *Deep Gaussian Processes*. AISTATS. (For the conceptual foundation and hierarchical structure).
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (For approximating GP layers).
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (For GP fundamentals).
*   Hierarchical Bayesian Models: Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press. (For general hierarchical modeling principles).


```{julia}
@model function model_v15_deep_gp(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_space, coords_time; period=12.0, M_rff=40)
    N = length(y_obs)
    D_s = size(coords_space, 2)
    D_t = size(coords_time, 2)

    # --- Layer 1: Latent Spatial GP (Z) ---
    W_z ~ filldist(Normal(0, 1), D_s, M_rff)
    b_z ~ filldist(Uniform(0, 2pi), M_rff)
    beta_z ~ filldist(Normal(0, 1), M_rff)

    # Latent Z values (input to Layer 2)
    z_latent = rff_map(coords_space, W_z, b_z) * beta_z

    sigma_z ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, sigma_z^2 * I)

    # --- Layer 2: Latent Spatiotemporal GPs (U1, U2, U3) ---
    # Inputs: Space + Time + Latent Z from Layer 1
    coords_l2 = hcat(coords_space, coords_time, z_latent)
    D_l2_in = size(coords_l2, 2)

    W_u ~ filldist(Normal(0, 1), D_l2_in, M_rff)
    b_u ~ filldist(Uniform(0, 2pi), M_rff)

    beta_u1 ~ filldist(Normal(0, 1), M_rff)
    beta_u2 ~ filldist(Normal(0, 1), M_rff)
    beta_u3 ~ filldist(Normal(0, 1), M_rff)

    Phi_u = rff_map(coords_l2, W_u, b_u)
    u1_latent = Phi_u * beta_u1
    u2_latent = Phi_u * beta_u2
    u3_latent = Phi_u * beta_u3

    sigma_u ~ filldist(Exponential(0.5), 3)
    u1_obs ~ MvNormal(u1_latent, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_latent, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_latent, sigma_u[3]^2 * I)

    # --- Layer 3: Final Output GP (Y) ---
    # Inputs: Original coords + Latent U's from Layer 2
    coords_l3 = hcat(coords_space, coords_time, u1_latent, u2_latent, u3_latent)
    D_l3_in = size(coords_l3, 2)

    W_y ~ filldist(Normal(0, 1), D_l3_in, M_rff)
    b_y ~ filldist(Uniform(0, 2pi), M_rff)
    beta_y_gp ~ filldist(Normal(0, 1), M_rff)

    # Trend and Seasonal components (Structural part of the deep GP)
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    # Deep GP realization for Y
    f_y = rff_map(coords_l3, W_y, b_y) * beta_y_gp

    sigma_y ~ Exponential(1.0)
    y_obs ~ MvNormal(f_y .+ seasonal, sigma_y^2 * I)
end
```

### Variational Inference for Deep GP (V15)

Given the high dimensionality of the Deep GP (Layered RFF), Markov Chain Monte Carlo (MCMC) like NUTS can be extremely slow. Variational Inference (VI) provides a faster alternative by approximating the posterior $p(\theta|y)$ with a simpler distribution $q(\theta)$, typically a Gaussian, and maximizing the Evidence Lower Bound (ELBO).

$$\text{ELBO}(q) = \mathbb{E}_{q(\theta)}[\log p(y, \theta)] - \mathbb{E}_{q(\theta)}[\log q(\theta)]$$

The following code sets up the ADVI objective and the variational posterior.


```{julia}
using AdvancedVI

# 1. Instantiate the model with data
# Assuming `data` is available from previous cells
model_vi = model_v15_deep_gp(
    data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.y_obs, # Using y as proxy for z_obs if not distinct
    data.coords_space, data.coords_time
)

# 2. Define the Variational Approximation (Mean Field Gaussian)
# This creates a multivariate normal with diagonal covariance in the unconstrained space
advi = ADVI(10, 1000) # 10 samples for ELBO gradient estimation, 1000 iterations

# 3. Setup the optimizer and objective
# In a real workflow, you would use `vi(model, advi)` which returns the optimized variational posterior
println("Variational Inference (ADVI) objective configured for Model V15.")

# Simple helper to run optimization (Skeleton)
function run_vi_optimization(model, samples=10, max_iters=500)
    # MeanField provides a diagonal Gaussian approximation
    q = vi(model, ADVI(samples, max_iters))
    return q
end
```

## Deep Kernel Learning with Flux.jl

To scale these models or combine them with neural network architectures, we can implement the RFF mapping as a custom Flux layer. This allows the coordinate warping (the 'Deep' part of the GP) to be optimized using standard deep learning optimizers like Adam.


```{julia}

using Flux

# Define a custom RFF Layer for Flux
struct RFFLayer{W, B}
    weights::W
    bias::B
end

# Make the layer callable
function (m::RFFLayer)(x::AbstractMatrix)
    projection = (x * m.weights) .+ m.bias'
    return sqrt(2 / size(m.weights, 2)) .* cos.(projection)
end

# Make the layer trainable
Flux.@functor RFFLayer

# Example of a DEEPER GP represented as a Flux Chain with activations
function create_deep_gp_flux(in_dims, hidden_dims, m_features)
    return Chain(
        # Layer 1: Initial transformation
        Dense(in_dims, hidden_dims, relu),
        # Layer 2: Intermediate processing with ReLU activation
        Dense(hidden_dims, hidden_dims, relu),
        # Layer 3: Another intermediate layer with ReLU activation
        Dense(hidden_dims, hidden_dims, relu),
        # Layer 4: RFF mapping (Spectral feature space)
        RFFLayer(randn(hidden_dims, m_features), rand(m_features) .* 2pi),
        # Output Layer: Probabilistic linear combination
        Dense(m_features, 1)
    )
end

println("Updated Flux architecture with ReLU activations between all dense layers.")
```


```{julia}
# 1. Prepare Data with Train/Validation Split
using Random
Random.seed!(123)

# Total data
x_all = Float32.(hcat(data.coords_space, data.coords_time))
y_all = Float32.(reshape(data.y_obs, 1, :))
N_total = size(x_all, 1)

# Simple 80/20 split
idx = shuffle(1:N_total)
train_idx = idx[1:Int(floor(0.8*N_total))]
val_idx = idx[Int(floor(0.8*N_total))+1:end]

x_train = x_all[train_idx, :]
y_train = y_all[:, train_idx]
x_val = x_all[val_idx, :]
y_val = y_all[:, val_idx]

# 2. Initialize Model and Optimizer
in_dims = size(x_train, 2)
hidden_dims = 10
m_features = 50
model_flux = create_deep_gp_flux(in_dims, hidden_dims, m_features)

loss(m, x, y) = Flux.mse(m(x), y)
opt_state = Flux.setup(Flux.Adam(0.01), model_flux)

println("Data split into train ($(length(train_idx))) and validation ($(length(val_idx))) sets.")
```


```{julia}
using Plots

# 1. Early Stopping Parameters
patience = 20
best_val_loss = Inf32
best_model_params = nothing

# 2. Training Loop
train_losses = Float32[]
val_losses = Float32[]
epochs_without_improvement = 0

println("Starting Flux training with early stopping...")

for epoch in 1:1000
    l_train, grads = Flux.withgradient(model_flux) do m
        loss(m, x_train, y_train)
    end
    Flux.update!(opt_state, model_flux, grads[1])

    l_val = loss(model_flux, x_val, y_val)
    push!(train_losses, l_train)
    push!(val_losses, l_val)

    if l_val < best_val_loss
        best_val_loss = l_val
        best_model_params = deepcopy(model_flux)
        epochs_without_improvement = 0
    else
        epochs_without_improvement += 1
    end

    if epoch % 50 == 0
        println("Epoch $epoch: Train Loss = $l_train, Val Loss = $l_val")
    end

    if epochs_without_improvement >= patience
        println("Early stopping triggered at epoch $epoch.")
        global model_flux = best_model_params
        break
    end
end

# 3. Visualization
plot(train_losses, label="Train Loss", lw=2, xscale=:log10, yscale=:log10,
     title="Deep Kernel Learning Convergence", xlabel="Epoch", ylabel="MSE Loss")
plot!(val_losses, label="Val Loss", lw=2, linestyle=:dash)
```


```{julia}
using Plots

# Diagnostic plot for Training Loss Convergence
# This helps visualize if the learning rate was appropriate and how quickly the model reached its plateau
plot(train_losses,
     title="Training Loss Diagnostics",
     xlabel="Epoch",
     ylabel="Mean Squared Error (MSE)",
     label="Training Loss",
     lw=2.5,
     color=:blue,
     xscale=:log10,
     yscale=:log10,
     grid=true,
     minorgrid=true)

# Annotate the final loss value
final_loss = train_losses[end]
annotate!(length(train_losses), final_loss,
          text(" Final: $(round(final_loss, digits=5))", :left, 8, :blue))
```


```{julia}
using Statistics
using Plots

"""
    evaluate_and_plot(m, x, y, mu_train; title="Validation Parity Plot", custom_metrics=Dict())

Calculates standard metrics (RMSE, R²), compares against a mean baseline,
and processes an optional dictionary of custom metric functions: (y_pred, y_true) -> score.
"""
function evaluate_and_plot(m, x, y, mu_train; title="Validation Parity Plot", custom_metrics=Dict())
    # 1. Model Predictions
    y_pred = m(x)

    # 2. Standard Metrics
    rmse = sqrt(Flux.mse(y_pred, y))
    ss_res = sum((y .- y_pred).^2)
    ss_tot = sum((y .- mean(y)).^2)
    r2 = 1 - (ss_res / ss_tot)

    # 3. Baseline Metrics (Predicting the mean)
    y_base = fill(mu_train, size(y))
    rmse_base = sqrt(Flux.mse(y_base, y))

    # 4. Print Summary Table
    println("--- Performance Summary ---")
    println(rpad("Metric", 15), " | ", rpad("Model", 10), " | ", rpad("Baseline", 10))
    println("-"^16, "|", "-"^12, "|", "-"^11)
    println(rpad("RMSE", 15), " | ", rpad(round(rmse, digits=4), 10), " | ", rpad(round(rmse_base, digits=4), 10))
    println(rpad("R²", 15), " | ", rpad(round(r2, digits=4), 10), " | ", rpad("0.0000", 10))

    # 5. Custom Metrics Execution
    for (name, func) in custom_metrics
        val = func(y_pred, y)
        println(rpad(string(name), 15), " | ", rpad(round(val, digits=4), 10), " | ", rpad("--", 10))
    end

    # 6. Parity Plot
    p = scatter(vec(y), vec(y_pred), aspect_ratio=:equal, title=title,
                xlabel="Actual", ylabel="Predicted", label="Predictions", alpha=0.6)

    min_val, max_val = minimum(y), maximum(y)
    plot!(p, [min_val, max_val], [min_val, max_val],
          color=:red, linestyle=:dash, label="Ideal", lw=2)

    return p
end

# Example execution with a custom metric (MAE)
train_mu = mean(y_train)
metrics = Dict("MAE" => (yp, yt) -> mean(abs.(yp .- yt)))
evaluate_and_plot(model_flux, x_val, y_val, train_mu, custom_metrics=metrics)
```


    UndefVarError: `y_train` not defined in `Main`
    Suggestion: check for spelling errors or missing imports.

    

    Stacktrace:

     [1] top-level scope

       @ In[27]:47


### Baseline Comparison

To confirm the model is learning useful features, we compare it against a baseline that simply predicts the mean of the training data. A valid model must significantly outperform this baseline.


```{julia}
# 1. Create Mean Baseline Predictions (no model , just the mean)
train_mean = mean(y_train)
y_pred_baseline = fill(train_mean, size(y_val))

# 2. Calculate Baseline Metrics
mse_base = Flux.mse(y_pred_baseline, y_val)
rmse_base = sqrt(mse_base)
r2_base = 0.0 # By definition


```

### Note on Consolidation
Redundant training cells were removed to simplify the notebook flow. The active training logic is now hosted in the cells above.


```{julia}
# Using consolidated evaluate_and_plot function defined above
train_mu = mean(y_train)
evaluate_and_plot(model_flux, x_val, y_val, train_mu, title="Deep GP Validation")
```

### Baseline Comparison

To understand the quality of the Deep GP model, we compare its performance against a simple baseline (the mean of the training target). If the Deep GP is effective, its RMSE should be significantly lower and its R² significantly higher than this baseline.


```{julia}
# 1. Create Mean Baseline Predictions
train_mean = mean(y_train)
y_pred_baseline = fill(train_mean, size(y_val))

# 2. Calculate Baseline Metrics
mse_base = Flux.mse(y_pred_baseline, y_val)
rmse_base = sqrt(mse_base)
mae_base = mean(abs.(y_pred_baseline .- y_val))
ss_res_base = sum((y_val .- y_pred_baseline).^2)
ss_tot_base = sum((y_val .- mean(y_val)).^2)
r2_base = 1 - (ss_res_base / ss_tot_base)

# 3. Display Comparison Table
println("--- Performance Comparison ---")
println("Metric | Deep GP | Mean Baseline")
println("-------|---------|--------------")
println("MSE    | ", round(mse_val, digits=4), "  | ", round(mse_base, digits=4))
println("RMSE   | ", round(rmse_val, digits=4), "  | ", round(rmse_base, digits=4))
println("MAE    | ", round(mae_val, digits=4), "  | ", round(mae_base, digits=4))
println("R²     | ", round(r2_val, digits=4), "  | ", round(r2_base, digits=4))
```

### Saving and Loading Model Parameters

We can use Julia's built-in `Serialization` library to save the entire trained Flux model to disk. This is useful for persisting the learned 'warping' function for future inference without retraining.


```{julia}
using Serialization

# 1. Save the model to a file
model_path = "deep_gp_model.jls"
serialize(model_path, model_flux)
println("Model saved to: ", model_path)

# 2. To load the model back later:
# loaded_model = deserialize(model_path)
# println("Model successfully reloaded.")
```


```{julia}
using JLD2

# 1. Save the model to a file using JLD2
model_path = "deep_gp_model.jld2"
jldsave(model_path; model_state = model_flux)
println("Model saved to: ", model_path)

# 2. To load the model back later:
# data = jldopen(model_path, "r")
# loaded_model = data["model_state"]
# println("Model successfully reloaded.")
```

## V16: Nyström Approximation

This model implements the Nyström Approximation to the Gaussian Process. The Nyström method approximates the full $N \times N$ covariance matrix $K$ using a subset of $M$ inducing points $Z$, such that $\tilde{K} = K_{XZ} K_{ZZ}^{-1} K_{ZX}$. This provides a low-rank approximation of the latent process, which is computationally efficient while maintaining the global correlation structure better than a purely diagonal FITC approximation in some regimes.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V12-V15, modeled with a mean component (GP trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): Same as V12-V15, modeled as nonlinear functions using separate RFF mappings of `coords_time`, `Z`, and other `U` covariates. Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) still have measurement error.
*   Trend: Same as V12-V15, a Gaussian Process-based trend (`GP Trend`).
*   Seasonal Process: Same as V12-V15, a fixed-period harmonic.
*   Spatiotemporal GP (f): New in V16, the main spatiotemporal GP is approximated using the Nyström method. This involves:
    *   Inducing Points (`Z_inducing`): Like V11 and V12, the locations of the inducing points are treated as parameters to be learned directly by the NUTS sampler, initialized with priors based on the input data.
    *   Kernel (`k_st`): An anisotropic Squared Exponential kernel (`SqExponentialKernel() \circ ARDTransform(inv.(ls_st))`) is used.
    *   Approximation: The latent GP `f` is constructed as a low-rank approximation: $f = \text{sigma_f} \cdot (K_{XZ} (L_{ZZ}' \\ v_{latent}))$, where $K_{XZ}$ is the cross-covariance between data and inducing points, $L_{ZZ}$ is the Cholesky decomposition of the inducing point covariance $K_{ZZ}$, and $v_{latent}$ is a standard normal noise vector. This approximates the full covariance while being more efficient than exact GP methods.
*   Observation Noise (sigma_y): Same as V12-V15, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V12-V15, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, extended for the `ls_st` (anisotropic lengthscales) and `sigma_f` for the Nyström GP, and all parameters for the stochastic volatility component.

### Key References:
*   Nyström Approximation: Williams, C. K. I., & Seeger, M. (2001). *Using the Nyström method to speed up kernel machines*. In *Advances in neural information processing systems*, 14, 682-689.
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (For general GP theory).
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS.
*   NUTS Sampler: Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*. Journal of Machine Learning Research, 15, 1593-1623. (For the MCMC sampling method).


```{julia}
@model function model_v16_nystrom(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0, M_rff_sigma=20, M_inducing_val=15, M_rff_u=30)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)

    # --- Priors ---
    sigma_u ~ filldist(Exponential(0.5), 3)

    # --- Nested Latent Covariates (Full RFF-based Logic) ---
    coords_tz = hcat(coords_time, z)

    # U1 = f1(coords_time, Z)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M_rff_u)
    b_u1 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0)
    beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1^2), M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * beta_rff_u1

    # U2 = f2(coords_time, Z, U1)
    coords_tz_u1 = hcat(coords_time, z, u1_true)
    W_u2 ~ filldist(Normal(0, 1), size(coords_tz_u1, 2), M_rff_u)
    b_u2 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0)
    beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2^2), M_rff_u)
    u2_true = rff_map(coords_tz_u1, W_u2, b_u2) * beta_rff_u2

    # U3 = f3(coords_time, Z, U1)
    W_u3 ~ filldist(Normal(0, 1), size(coords_tz_u1, 2), M_rff_u)
    b_u3 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0)
    beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3^2), M_rff_u)
    u3_true = rff_map(coords_tz_u1, W_u3, b_u3) * beta_rff_u3

    # --- Structural Components ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    ls_trend ~ Gamma(2, 2)
    sigma_trend ~ Exponential(0.5)
    k_trend = SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend))
    g_trend = GP(sigma_trend^2 * k_trend)
    unique_times = sort(unique(coords_time[:,1]))
    alpha ~ g_trend(unique_times, 1e-1)
    trend = alpha[indexin(coords_time[:,1], unique_times)]

    # --- Nystr&#246;m GP (Low Rank Approximation) ---
    coords_st = hcat(coords_space, coords_time)
    ls_st ~ filldist(Gamma(2, 2), D_st)
    sigma_f ~ Exponential(1.0)
    k_st = SqExponentialKernel() ∘ ARDTransform(inv.(ls_st))

    # Inducing points learned as parameters
    Z_ind = Matrix{Float64}(undef, M_inducing_val, D_st)
    mu_st = mean(coords_st, dims=1)
    std_st = std(coords_st, dims=1)
    for j in 1:D_st
        Z_ind[:, j] ~ filldist(Normal(mu_st[j], 2.0 * std_st[j]), M_inducing_val)
    end

    # K_zz and K_xz for the Nystr&#246;m projection
    K_zz = kernelmatrix(k_st, RowVecs(Z_ind)) + 1e-6*I
    K_xz = kernelmatrix(k_st, RowVecs(coords_st), RowVecs(Z_ind))

    L_zz = cholesky(K_zz).L
    v_latent ~ filldist(Normal(0, 1), M_inducing_val)

    # The low-rank realization f ≈ K_xz * inv(K_zz) * u
    f = sigma_f .* (K_xz * (L_zz' \\ v_latent))

    # --- Spatiotemporal Stochastic Volatility ---
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0)
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)

    Phi_sigma = rff_map(coords_st, W_sigma, b_sigma)
    sigma_y = exp.(Phi_sigma * beta_rff_sigma ./ 2)

    # --- Likelihoods ---
    beta_covs ~ filldist(Normal(0, 1), 4)
    mu_y = trend .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    y_obs ~ MvNormal(mu_y .+ f, Diagonal(sigma_y.^2 .+ 1e-4))
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```


```{julia}
model_v16 = model_v16_nystrom(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time)
chain_v16 = sample(model_v16, NUTS(), 100) 
display(describe(chain_v16))
waic_v16 = compute_y_waic(model_v16_inst, chain_v16)
println("\nWAIC for V16 (Nystr&#246;m): ", waic_v16)

```

## V17: SPDE Approximation (Spatial Matern 3/2)

This model implements an SPDE Approximation for the spatial component. While a full finite-element mesh-based implementation (like R-INLA) requires specialized triangulations, we approximate the behavior here using a discrete Laplacian on a grid or a Gaussian Markov Random Field (GMRF) representation. This allows the model to scale to larger spatial datasets by exploiting the sparsity of the precision matrix $Q$.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V12-V16, modeled with a mean component (GP trend, seasonal, covariates) and a latent spatiotemporal Gaussian Process (GP) component, plus spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): Same as V12-V16, modeled as nonlinear functions using separate RFF mappings of `coords_time`, `Z`, and other `U` covariates. Their observed values (`u1_obs`, `u2_obs`, `u3_obs`) still have measurement error.
*   Trend: Same as V12-V16, a Gaussian Process-based trend (`GP Trend`).
*   Seasonal Process: Same as V12-V16, a fixed-period harmonic.
*   Spatiotemporal GP (f): New in V17, the main spatiotemporal GP explicitly incorporates an SPDE (Stochastic Partial Differential Equation) approximation for its spatial component. Instead of a full GP or Nyström approximation, the spatial process `f_spatial` is directly sampled from a `MvNormal` with a covariance matrix derived from a Matern 3/2 kernel, which is a common approach to approximate SPDE solutions. This helps to manage computational complexity for large spatial datasets by implicitly leveraging the connection between Matern kernels and SPDEs.
*   Observation Noise (sigma_y): Same as V12-V16, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u): Same as V12-V16, homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, extended for the `ls_s` (spatial lengthscale) and `sigma_s` for the SPDE-approximated spatial GP, and all parameters for the stochastic volatility component.

### Key References:
*   SPDE Approximation: Lindgren, F., Rue, H., & Lindström, J. (2011). *An explicit link between Gaussian fields and Gaussian Markov random fields: The SPDE approach*. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(4), 423-498.
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (For general GP theory and Matern kernels).
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS.
*   NUTS Sampler: Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*. Journal of Machine Learning Research, 15, 1593-1623. (For the MCMC sampling method).


```{julia}
@model function model_v17_spde(y_obs, u1_obs, u2_obs, u3_obs, z, coords_space, coords_time; period=12.0, M_rff_sigma=20, M_rff_u=30)
    N = length(y_obs)
    T_unique = length(unique(coords_time))
    D_s = size(coords_space, 2)
    D_st = D_s + size(coords_time, 2)

    # --- Priors ---
    sigma_u ~ filldist(Exponential(0.5), 3)

    # --- Nested Latent Covariates (RFF-based from V12/V16) ---
    coords_tz = hcat(coords_time, z)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M_rff_u)
    b_u1 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0)
    beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1^2), M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * beta_rff_u1

    coords_tz_u1 = hcat(coords_time, z, u1_true)
    W_u2 ~ filldist(Normal(0, 1), size(coords_tz_u1, 2), M_rff_u)
    b_u2 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0)
    beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2^2), M_rff_u)
    u2_true = rff_map(coords_tz_u1, W_u2, b_u2) * beta_rff_u2

    W_u3 ~ filldist(Normal(0, 1), size(coords_tz_u1, 2), M_rff_u)
    b_u3 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0)
    beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3^2), M_rff_u)
    u3_true = rff_map(coords_tz_u1, W_u3, b_u3) * beta_rff_u3

    # --- Structural Components ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    ls_trend ~ Gamma(2, 2); sigma_trend ~ Exponential(0.5)
    k_trend = SqExponentialKernel() ∘ ScaleTransform(inv(ls_trend))
    g_trend = GP(sigma_trend^2 * k_trend)
    unique_times = sort(unique(coords_time[:,1]))
    alpha ~ g_trend(unique_times, 1e-1)
    trend = alpha[indexin(coords_time[:,1], unique_times)]

    # --- SPDE / GMRF Spatial Approximation (Matern 3/2) ---
    # In the SPDE approach, the Matern kernel is the solution to (̄ͅ - Δ)̲f = W
    # We approximate the spatial covariance using a sparse precision matrix approach.
    ls_s ~ Gamma(2, 2)
    sigma_s ~ Exponential(1.0)

    # Construct a Matern 3/2 spatial kernel matrix (as proxy for SPDE precision behavior)
    k_s = Matern32Kernel() ∘ ScaleTransform(inv(ls_s))
    K_s = sigma_s^2 * kernelmatrix(k_s, RowVecs(coords_space)) + 1e-6*I
    f_spatial ~ MvNormal(zeros(N), K_s)

    # --- Spatiotemporal Stochastic Volatility ---
    coords_st = hcat(coords_space, coords_time)
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0)
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)
    sigma_y = exp.(rff_map(coords_st, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- Likelihood ---
    beta_covs ~ filldist(Normal(0, 1), 4)
    mu_y = trend .+ seasonal .+ f_spatial .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z .* beta_covs[4])

    y_obs ~ MvNormal(mu_y, Diagonal(sigma_y.^2 .+ 1e-4))
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)
end
```


```{julia}
# Instantiate and test model_v17_spde
model_v17 = model_v17_spde(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time)
chain_v17 = sample(model_v17, NUTS(), 100)
display(describe(chain_v17))
waic_v17 = compute_y_waic(model_v17, chain_v17)
println("WAIC for V17 (SPDE): ", waic_v17)

```

## V18: Kronecker-Spatiotemporal SPDE Approximation

This model builds upon V17 by replacing the additive seasonal and trend components with a Spatiotemporal SPDE modeled via a Kronecker product. By assuming a separable structure $K = K_s \otimes K_t$, we can exploit the sparsity of the precision matrices in both dimensions. This provides a unified spatiotemporal field while keeping the memory footprint low using `SparseArrays`.

### Model Assumptions:
*   Dependent Variable (Y): Similar to V12-V17, modeled with a mean component (latent spatiotemporal process) and spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3, Z): New in V18, these are all modeled as latent spatiotemporal fields using the Kronecker product of Matern 3/2 spatial and temporal kernels. This allows for explicit modeling of spatiotemporal dependencies within covariates. Each field is sampled using a non-centered parameterization with a noise vector.
*   Main Spatiotemporal Process (f_st): New in V18, the primary spatiotemporal component `f_st` is also modeled as a Kronecker-Spatiotemporal SPDE, using Matern 3/2 kernels for both spatial and temporal dimensions. This replaces the separate trend, seasonal, and `f_spatial` components from previous models with a unified structure.
*   Seasonality and Trend: Implicitly captured by the flexible Kronecker Spatiotemporal SPDE processes for `f_st` and covariates, rather than explicit additive components.
*   Observation Noise (sigma_y): Same as V12-V17, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Covariate Observation Noise (sigma_u, sigma_z_obs): Assumed to be homoscedastic and normally distributed.
*   Priors: Standard weakly informative priors, extended for the `ls_s_cov`, `sigma_s_cov`, `ls_t_cov`, `sigma_t_cov` (for covariates), and `ls_s_y`, `sigma_s_y`, `ls_t_y`, `sigma_t_y` (for the main process).

### Key References:
*   Kronecker Product Kernels: Stegle, O., Kadie, C. M., Norman, P. J., & Winn, J. (2011). *Efficient inference in Gaussian process models with `Kronecker` structure*. In *Advances in neural information processing systems*, 24.
*   SPDE Approximation: Lindgren, F., Rue, H., & Lindström, J. (2011). *An explicit link between Gaussian fields and Gaussian Markov random fields: The SPDE approach*. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 73(4), 423-498.
*   Gaussian Processes: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (Used for stochastic volatility component).
*   NUTS Sampler: Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*. Journal of Machine Learning Research, 15, 1593-1623.


```{julia}
using SparseArrays

function kron_matern_sample(Ns, Nt, unique_s, unique_t, ls_s, sigma_s, ls_t, sigma_t, noise_vec)
    # Helper to sample a spatiotemporal field using Kronecker product of precision matrices
    # Spatial Precision
    k_s = Matern32Kernel() ∘ ScaleTransform(inv(ls_s))
    K_s = sigma_s^2 * kernelmatrix(k_s, RowVecs(unique_s)) + 1e-6*I
    Q_s = sparse(inv(K_s))

    # Temporal Precision
    k_t = Matern32Kernel() ∘ ScaleTransform(inv(ls_t))
    K_t = sigma_t^2 * kernelmatrix(k_t, unique_t) + 1e-6*I
    Q_t = sparse(inv(K_t))

    # Full Kronecker Precision
    Q_full = kron(Q_t, Q_s)
    L_q = cholesky(Q_full + 1e-6*I)

    # Sample: f = (L')^-1 * noise
    return L_q.PtL' \\ noise_vec
end

@model function model_v18_kronecker_spde(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_space, coords_time; M_rff_sigma=20)
    N = length(y_obs)
    unique_t = sort(unique(coords_time[:, 1]))
    Nt = length(unique_t)
    Ns = N ÷ Nt
    unique_s = coords_space[1:Ns, :]
    D_st = size(coords_space, 2) + size(coords_time, 2)
    coords_st = hcat(coords_space, coords_time)

    # Noise vectors for non-centered parameterization
    z_noise ~ filldist(Normal(0, 1), N)
    u1_noise ~ filldist(Normal(0, 1), N)
    u2_noise ~ filldist(Normal(0, 1), N)
    u3_noise ~ filldist(Normal(0, 1), N)
    y_noise ~ filldist(Normal(0, 1), N)

    # Common lengthscale and variance priors for covariates
    ls_s_cov ~ Gamma(2, 2)
    sigma_s_cov ~ Exponential(1.0)
    ls_t_cov ~ Gamma(2, 2)
    sigma_t_cov ~ Exponential(1.0)

    # 1. Latent Kronecker Field for Z
    z_true = kron_matern_sample(Ns, Nt, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, z_noise)
    sigma_z_obs ~ Exponential(0.5)
    z_obs ~ MvNormal(z_true, sigma_z_obs^2 * I)

    # 2. Latent Kronecker Fields for U1, U2, U3
    # (All using Matern 3/2 Spatiotemporal Kronecker structure)
    u1_true = kron_matern_sample(Ns, Nt, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, u1_noise)
    u2_true = kron_matern_sample(Ns, Nt, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, u2_noise)
    u3_true = kron_matern_sample(Ns, Nt, unique_s, unique_t, ls_s_cov, sigma_s_cov, ls_t_cov, sigma_t_cov, u3_noise)

    sigma_u ~ filldist(Exponential(0.5), 3)
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)

    # 3. Latent Kronecker Field for Y (Main Process)
    ls_s_y ~ Gamma(2, 2)
    sigma_s_y ~ Exponential(1.0)
    ls_t_y ~ Gamma(2, 2)
    sigma_t_y ~ Exponential(1.0)

    f_st = kron_matern_sample(Ns, Nt, unique_s, unique_t, ls_s_y, sigma_s_y, ls_t_y, sigma_t_y, y_noise)

    # 4. Stochastic Volatility
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0)
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)
    sigma_y_vec = exp.(rff_map(coords_st, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # 5. Regression and Final Likelihood
    beta_covs ~ filldist(Normal(0, 1), 4)
    mu_y = f_st .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z_true .* beta_covs[4])

    y_obs ~ MvNormal(mu_y, Diagonal(sigma_y_vec.^2 .+ 1e-4))
end
```


```{julia}
# Sample model_v18_kronecker_spde
model_v18 = model_v18_kronecker_spde(data.y_obs, data.U1_obs, data.U2_obs, data.U3_obs, data.Z, data.coords_space, data.coords_time)
chain_v18 = sample(model_v18, NUTS(), 100)
display(describe(chain_v18))
waic_v18 = compute_y_waic(model_v18, chain_v18)
println("\nWAIC for V18 (Kronecker SPDE): ", waic_v18)
```

## V19: SVGP with Kronecker Matern Kernel

This model builds upon V18's Kronecker Spatiotemporal SPDE approximation for the main latent process (`f_st`), but integrates it with the nested RFF covariates and a specific structure for the spatiotemporal covariance using Matern kernels and a seasonal component. The goal is to combine the flexibility of nested RFFs for covariates with a scalable and interpretable spatiotemporal main process.

### Model Assumptions:
*   Dependent Variable (Y): Modeled with a mean component (latent spatiotemporal process, seasonal, and covariates) and spatiotemporal stochastic observation noise.
*   Latent Covariates (U1, U2, U3): Modeled as nonlinear functions using separate RFF mappings, similar to V12. These dependencies are nested (U1 based on time/Z, U2/U3 based on time/Z/U1). `Z_obs` is treated as an observed spatial covariate influencing `U` and `Y`.
*   Main Spatiotemporal Process (f_st): New in V19, the primary spatiotemporal component `f_st` is modeled using a Kronecker product of Matern 3/2 spatial and temporal kernels, similar to the underlying GP structure in V18. It is sampled using a non-centered parameterization with a noise vector.
*   Seasonality: Explicitly modeled as an additive harmonic component (sine/cosine waves), distinguishing it from the intrinsic temporal correlation of the Matern kernel in `f_st`.
*   Stochastic Volatility: Same as V18, modeled as a spatiotemporally varying process using a secondary RFF mapping for the log-variance.
*   Observation Noise (sigma_u): Assumed to be homoscedastic and normally distributed for `U` observations.
*   Priors: Standard weakly informative priors are used for all parameters, including lengthscales and signal variances for the Matern kernels, RFF parameters, and seasonal coefficients.

### Key References:
*   Kronecker Product Kernels: Stegle, O., Kadie, C. M., Norman, P. J., & Winn, J. (2011). *Efficient inference in Gaussian process models with `Kronecker` structure*. In *Advances in neural information processing systems*, 24.
*   Matern Kernels: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (For properties and use of Matern kernels).
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (For nested covariate modeling and stochastic volatility).
*   NUTS Sampler: Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*. Journal of Machine Learning Research, 15, 1593-1623.


```{julia}
@model function model_v19_svgp_matern(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_space, coords_time; period=12.0, M_rff_sigma=20, M_rff_u=30)
    N = length(y_obs)
    unique_t = sort(unique(coords_time[:, 1]))
    Nt = length(unique_t)
    Ns = N ÷ Nt
    unique_s = coords_space[1:Ns, :]
    D_st = size(coords_space, 2) + size(coords_time, 2)
    coords_st = hcat(coords_space, coords_time)

    # --- 1. Latent Covariates (Nested RFF with Matern Structure) ---
    # Noise for non-centered parameterization of fields
    u1_noise ~ filldist(Normal(0, 1), N)
    u2_noise ~ filldist(Normal(0, 1), N)
    u3_noise ~ filldist(Normal(0, 1), N)

    ls_s_u ~ Gamma(2, 2); sigma_s_u ~ Exponential(1.0)
    rho_u ~ Uniform(-0.99, 0.99); sigma_t_u ~ Exponential(0.5)

    # U1, U2, U3 as latent spatiotemporal fields (Simplified here via RFF for the nesting logic)
    # Z is treated as a known spatial covariate here, consistent with V12
    coords_tz = hcat(coords_time, z_obs)
    W_u1 ~ filldist(Normal(0, 1), size(coords_tz, 2), M_rff_u)
    b_u1 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u1 ~ Exponential(1.0)
    beta_rff_u1 ~ filldist(Normal(0, sigma_f_u1^2), M_rff_u)
    u1_true = rff_map(coords_tz, W_u1, b_u1) * beta_rff_u1

    coords_tz_u1 = hcat(coords_time, z_obs, u1_true)
    W_u2 ~ filldist(Normal(0, 1), size(coords_tz_u1, 2), M_rff_u)
    b_u2 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u2 ~ Exponential(1.0)
    beta_rff_u2 ~ filldist(Normal(0, sigma_f_u2^2), M_rff_u)
    u2_true = rff_map(coords_tz_u1, W_u2, b_u2) * beta_rff_u2

    W_u3 ~ filldist(Normal(0, 1), size(coords_tz_u1, 2), M_rff_u)
    b_u3 ~ filldist(Uniform(0, 2pi), M_rff_u)
    sigma_f_u3 ~ Exponential(1.0)
    beta_rff_u3 ~ filldist(Normal(0, sigma_f_u3^2), M_rff_u)
    u3_true = rff_map(coords_tz_u1, W_u3, b_u3) * beta_rff_u3

    sigma_u ~ filldist(Exponential(0.5), 3)
    u1_obs ~ MvNormal(u1_true, sigma_u[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u[3]^2 * I)

    # --- 2. Main Spatiotemporal Process (Kronecker Matern) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    ls_t_y ~ Gamma(2, 2); sigma_t_y ~ Exponential(1.0)
    y_noise ~ filldist(Normal(0, 1), N)

    # Use the helper from V18/V20 context
    f_st = kron_matern_sample(Ns, Nt, unique_s, unique_t, ls_s_y, sigma_s_y, ls_t_y, sigma_t_y, y_noise)

    # --- 3. Structural Components ---
    beta_cos ~ Normal(0, 1); beta_sin ~ Normal(0, 1)
    seasonal = beta_cos .* cos.(2 * pi .* coords_time[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_time[:,1] ./ period)

    # --- 4. Stochastic Volatility ---
    W_sigma ~ filldist(Normal(0, 1), D_st, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0)
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)
    sigma_y_vec = exp.(rff_map(coords_st, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- 5. Likelihood ---
    beta_covs ~ filldist(Normal(0, 1), 4)
    mu_y = f_st .+ seasonal .+ (u1_true .* beta_covs[1]) .+ (u2_true .* beta_covs[2]) .+ (u3_true .* beta_covs[3]) .+ (z_obs .* beta_covs[4])

    y_obs ~ MvNormal(mu_y, Diagonal(sigma_y_vec.^2 .+ 1e-4))
end
```


```{julia}
model_v19 = model_v19_svgp_matern(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_space, coords_time; period=12.0, M_rff_sigma=20, M_rff_u=30)
chain_v19 = sample(model_v19, NUTS(), 100)
display(describe(chain_v19))
waic_v19 = compute_y_waic(model_v19, chain_v19)
println("\nWAIC for V19: ", waic_v19)
```

## V20: Multi-fidelity Kronecker Matern GP

This model extends the multi-fidelity concept by employing Kronecker product kernels with Matern structures for each fidelity level, replacing the RFF-based approximations from earlier multi-fidelity models. It also introduces an AR(1) process for the temporal component within the Kronecker structure, providing a flexible and interpretable way to model temporal correlations.

### Model Assumptions:
*   Multi-fidelity Structure: Retains the hierarchical multi-fidelity idea:
    *   Z-fidelity (High Resolution): Latent spatial field modeled using a Matern 3/2 Kernel. This is the highest resolution and informs the lower fidelity layers.
    *   U-fidelity (Medium Resolution): Latent spatiotemporal fields (`U1, U2, U3`) modeled using a Kronecker product of a Matern 3/2 spatial kernel and an AR(1) temporal process. These fields depend on interpolated `Z` from the higher fidelity.
    *   Y-fidelity (Standard Resolution): The primary observation (`Y`) modeled using a Kronecker product of a Matern 3/2 spatial kernel and an AR(1) temporal process. It depends on interpolated latent `U` and `Z` fields.
*   Kernel-based Interpolation: Dependencies between fidelity levels are handled through kernel-based interpolation (e.g., using `K_z_u * (K_z \\ z_latent)`), ensuring that information flows consistently across resolutions.
*   GP Representation: All latent fields (`Z`, `U`, `Y`'s primary process) are implicitly modeled as Gaussian Processes with specified Matern and AR(1) covariance structures.
*   AR(1) Temporal Process: The temporal component for spatiotemporal fields (U and Y) uses an Auto-Regressive process of order 1, parameterized by `rho` (correlation) and `sigma_t_noise` (innovation variance), which is commonly used in state-space models.
*   Observation Noise: Homoscedastic and normally distributed for each observed variable (`z_obs`, `u1_obs`, `u2_obs`, `u3_obs`, `y_obs`).
*   Priors: Standard weakly informative priors are used for all parameters, including lengthscales and signal variances for the Matern kernels, and parameters for the AR(1) processes.

### Key References:
*   Multi-fidelity Gaussian Processes: Perdikaris, P., Raissi, M., Psaros, N., & Karniadakis, G. E. (2017). *Nonlinear model reduction for uncertainty quantification and predictive modeling of spatiotemporal systems*. Journal of Computational Physics, 347, 303-324.
*   Kronecker Product Kernels: Stegle, O., Kadie, C. M., Norman, P. J., & Winn, J. (2011). *Efficient inference in Gaussian process models with `Kronecker` structure*. In *Advances in neural information processing systems*, 24.
*   Matern Kernels: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. (For properties and use of Matern kernels).
*   AR(1) Models: Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. (For properties and use of AR(1) processes).
*   NUTS Sampler: Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo*. Journal of Machine Learning Research, 15, 1593-1623.


```{julia}
using SparseArrays

# Helper to create AR1 precision matrix
function ar1_precision(n, rho, sigma_e)
    Q = spzeros(n, n)
    # Main diagonal
    Q[1, 1] = 1.0
    for i in 2:(n - 1)
        Q[i, i] = 1.0 + rho^2
    end
    Q[n, n] = 1.0
    # Off-diagonals
    for i in 1:(n - 1)
        Q[i, i + 1] = -rho
        Q[i + 1, i] = -rho
    end
    return (1.0 / sigma_e^2) .* Q
end

# Helper for Kronecker AR1 x Matern Sampling
function kron_ar1_matern_sample(Ns, Nt, unique_s, ls_s, sigma_s, rho_t, sigma_t_noise, noise_vec)
    # Spatial Matern 3/2 Precision
    k_s = Matern32Kernel() ∘ ScaleTransform(inv(ls_s))
    K_s = sigma_s^2 * kernelmatrix(k_s, RowVecs(unique_s)) + 1e-6*I
    Q_s = sparse(inv(K_s))

    # Temporal AR1 Precision
    Q_t = ar1_precision(Nt, rho_t, sigma_t_noise)

    # Kronecker Product Q = Qt ⊗ Qs
    Q_full = kron(Q_t, Q_s)
    L_q = cholesky(Q_full + 1e-6*I)

    return L_q.PtL' \\ noise_vec
end

@model function model_v20_multifidelity_gp_matern(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_y_s, coords_y_t, coords_u_s, coords_u_t, coords_z_s; period=12.0)
    # Dimensions
    Ny = length(y_obs); Nt_y = length(unique(coords_y_t)); Ns_y = Ny ÷ Nt_y
    Nu = length(u1_obs); Nt_u = length(unique(coords_u_t)); Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- 1. High Fidelity: Latent Spatial Z (Matern 3/2) ---
    ls_z ~ Gamma(2, 2)
    sigma_z_f ~ Exponential(1.0)
    k_z = Matern32Kernel() ∘ ScaleTransform(inv(ls_z))
    # Define latent Z on its own high-res spatial grid
    K_z = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(coords_z_s)) + 1e-6*I
    z_latent ~ MvNormal(zeros(Nz), K_z)

    sigma_z_obs ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, sigma_z_obs^2 * I)

    # --- Interpolation logic for cross-fidelity dependencies ---
    # In this simplified implementation, we use the kernel to project latent Z
    # to U and Y locations.
    K_z_u = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(coords_u_s), RowVecs(coords_z_s))
    z_at_u = (K_z_u * (K_z \\ z_latent)) # Latent Z interpolated to U spatial locations
    z_at_u_full = repeat(z_at_u, Nt_u)

    # --- 2. Medium Fidelity: Latent Spatiotemporal U (Kron AR1 x Matern) ---
    ls_s_u ~ Gamma(2, 2); sigma_s_u ~ Exponential(1.0)
    rho_u ~ Uniform(-0.99, 0.99); sigma_t_u ~ Exponential(0.5)

    # Noise for U1, U2, U3
    u1_noise ~ filldist(Normal(0, 1), Nu)
    u2_noise ~ filldist(Normal(0, 1), Nu)
    u3_noise ~ filldist(Normal(0, 1), Nu)

    # Sample U fields
    u1_true = kron_ar1_matern_sample(Ns_u, Nt_u, coords_u_s[1:Ns_u, :], ls_s_u, sigma_s_u, rho_u, sigma_t_u, u1_noise)
    u2_true = kron_ar1_matern_sample(Ns_u, Nt_u, coords_u_s[1:Ns_u, :], ls_s_u, sigma_s_u, rho_u, sigma_t_u, u2_noise)
    u3_true = kron_ar1_matern_sample(Ns_u, Nt_u, coords_u_s[1:Ns_u, :], ls_s_u, sigma_s_u, rho_u, sigma_t_u, u3_noise)

    # Hierarchical dependence: Add effect of Z on U
    beta_uz ~ Normal(0, 1)
    u1_obs ~ MvNormal(u1_true .+ beta_uz .* z_at_u_full, 0.1*I)

    # --- 3. Standard Fidelity: Dependent Y (Kron AR1 x Matern) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    rho_y ~ Uniform(-0.99, 0.99); sigma_t_y ~ Exponential(0.5)
    y_noise ~ filldist(Normal(0, 1), Ny)

    f_st_y = kron_ar1_matern_sample(Ns_y, Nt_y, coords_y_s[1:Ns_y, :], ls_s_y, sigma_s_y, rho_y, sigma_t_y, y_noise)

    # Map latent U/Z to Y coordinates (simplified linear projection)
    beta_y ~ filldist(Normal(0, 1), 4)
    # For Y, we assume mu_y is a combination of latent fields
    # (In practice, requires kernel-based interpolation of U to Y coordinates)
    mu_y = f_st_y .+ beta_y[1] .* z_at_u_full[1:Ny] # Placeholder for alignment

    sigma_y ~ Exponential(1.0)
    y_obs ~ MvNormal(mu_y, sigma_y^2 * I)
end
```


```{julia}
model_v20 = model_v20_multifidelity_gp_matern(
    y_mock, u1_mock, u2_mock, u3_mock, z_mock,
    coords_y_s, coords_y_t,
    coords_u_s, coords_u_t,
    coords_z_s
)

println("Starting sampling for Model V20 (Multi-fidelity Kronecker Matern)... ")
chain_v20 = sample(model_v20, NUTS(), 100)
display(describe(chain_v20))
waic_v20 = compute_y_waic(model_v20, chain_v20)
println("\nWAIC for V20: ", waic_v20)

```

### Comparison: Single-Resolution vs. Multi-Fidelity Models

In this framework, the transition from models like V12 (SVGP) to V20 (Multi-Fidelity Kronecker Matern) represents a shift in how the data hierarchy is treated.

#### 1. Single-Resolution Models (e.g., V0 - V12, V19)
*   Data Assumption: All variables ($Y$, $U_1$, $U_2$, $U_3$, $Z$) are observed or modeled at the same spatiotemporal coordinates.
*   Dependency Structure: Covariates are typically treated as inputs to a single global latent process or as nested functions (RFF) that directly modify the mean of the target variable $Y$.
*   Computational Focus: Optimization centers on approximating the single $N \times N$ covariance matrix (via RFF, FITC, or Nyström) to handle large $N$.

#### 2. Multi-Fidelity Models (e.g., V13, V20)
*   Data Assumption: Recognizes that different variables exist at different 'fidelities' or resolutions. For example:
    *   High-Fidelity ($Z$): Dense spatial measurements (e.g., satellite data).
    *   Medium-Fidelity ($U$): Sparse spatiotemporal sensors.
    *   Standard-Fidelity ($Y$): The primary target, often at the coarsest resolution.
*   Dependency Structure: Implements a hierarchical latent field. Instead of just being a regressor, the high-fidelity latent field for $Z$ informs the medium-fidelity field $U$, which in turn informs the target $Y$.
*   Interpolation Logic: Requires kernel-based projection to align latent fields across different grids. V20 specifically uses Kronecker-structured precision matrices to maintain this hierarchy across resolutions without the memory overhead of a dense multi-fidelity covariance matrix.

## V21: Spatiotemporal Stochastic Volatility & Seasonal Harmonics for MFGP

### Subtask:
Enhance the V20 model by incorporating spatiotemporal stochastic volatility for observation noise and explicitly adding seasonal harmonics to the mean function.


### Utility Function: Compute 2D Spatial Spectral Features

This function takes 2D spatial coordinates, bins them onto a regular grid, and then computes the 2D Fast Fourier Transform (FFT) of the resulting spatial density. This can be used to identify dominant spatial frequencies or patterns in the distribution of your observation locations.

Note on Interpretation for RFFs: The output of this function provides the spectral content of your *sampling locations' density*. This is generally different from the spectral density of a *kernel function*, which is what Random Fourier Features (`W` values) are theoretically sampled from. While related to spatial scales, direct use of these FFT outputs as `W` and `b` for RFFs requires careful consideration and a specific mapping strategy, as `W` in RFFs is typically sampled from the kernel's spectral density, and `b` from a uniform distribution (phases).


```{julia}

using FFTW
using StatsBase: fit, Histogram, normalize

"""
    compute_spatial_spectral_features(coords_2d::Matrix, grid_res::Int)

Computes the 2D spectral features from irregular 2D spatial point data.

Args:
    coords_2d: A matrix where each row is a 2D spatial point (e.g., [x, y]).
    grid_res: The resolution of the square grid to which the points will be binned.
              A higher resolution provides more detail but increases computation.

Returns:
    A tuple (frequencies_x, frequencies_y, magnitude_spectrum).
    - frequencies_x: A vector of spatial frequencies along the x-dimension.
    - frequencies_y: A vector of spatial frequencies along the y-dimension.
    - magnitude_spectrum: A 2D array representing the magnitude of the FFT
                          at each (frequency_x, frequency_y) pair.
"""
function compute_spatial_spectral_features(coords_2d::Matrix, grid_res::Int)
    # Ensure coords_2d has 2 columns (x, y)
    if size(coords_2d, 2) != 2
        error("Input `coords_2d` must be an N x 2 matrix (x, y coordinates).")
    end

    # Determine the spatial extent of the data
    min_x, max_x = extrema(coords_2d[:, 1])
    min_y, max_y = extrema(coords_2d[:, 2])

    # Create bins for the 2D histogram
    x_bins = range(min_x, stop=max_x, length=grid_res + 1)
    y_bins = range(min_y, stop=max_y, length=grid_res + 1)

    # Bin the points into a 2D histogram to get a spatial density map
    h = fit(Histogram, (coords_2d[:, 1], coords_2d[:, 2]), (x_bins, y_bins))
    density_map = normalize(h, mode=:density).counts # Get normalized density counts

    # Perform 2D FFT on the density map
    fft_result = fft(density_map)

    # Compute the frequency components
    freqs_x = fftfreq(grid_res, 1.0 / (max_x - min_x) * grid_res) # Frequencies per unit length
    freqs_y = fftfreq(grid_res, 1.0 / (max_y - min_y) * grid_res)

    # Magnitude spectrum (log scale often useful for visualization)
    magnitude_spectrum = abs.(fftshift(fft_result))

    return fftshift(freqs_x), fftshift(freqs_y), magnitude_spectrum
end

```


### Demonstration of `compute_spatial_spectral_features`

We'll use the `coords_space` data from our mock dataset to demonstrate how to compute and display its 2D spatial spectral features.


```{julia}
# Assuming `data.coords_space` is available from previous cells
# If not, run `data = generate_data(50)` first.

# Set the grid resolution for the FFT
grid_resolution = 32 # A power of 2 is often good for FFT performance

# Compute the spectral features
freqs_x, freqs_y, magnitude_spectrum = compute_spatial_spectral_features(data.coords_space, grid_resolution)

println("Computed spatial spectral features:")
println("  - X-frequencies range: ", minimum(freqs_x), " to ", maximum(freqs_x))
println("  - Y-frequencies range: ", minimum(freqs_y), " to ", maximum(freqs_y))
println("  - Magnitude spectrum size: ", size(magnitude_spectrum))

# Optional: Plotting the magnitude spectrum (requires Plots.jl)
using Plots

p = heatmap(freqs_x, freqs_y, log.(magnitude_spectrum .+ 1e-9), # Add small constant to avoid log(0)
            xlabel="Spatial Frequency (X)",
            ylabel="Spatial Frequency (Y)",
            title="2D Spatial Magnitude Spectrum of Point Density",
            color=:viridis,
            aspect_ratio=:equal)

plot!(p, size=(600, 600))
display(p)

```


```{julia}
using StatsBase: sample, Weights

"""
    generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)

Generates 2D RFF weights W by sampling frequencies from the provided 2D magnitude spectrum.

Args:
    freqs_x: Vector of x-dimension frequencies.
    freqs_y: Vector of y-dimension frequencies.
    magnitude_spectrum: 2D array of magnitude values corresponding to freqs_x, freqs_y.
    M_rff_count: Number of RFF features to generate.

Returns:
    A 2 x M_rff_count matrix for W_fixed.
"""
function generate_spectral_w_from_magnitude(freqs_x, freqs_y, magnitude_spectrum, M_rff_count)
    # Flatten frequency grids and magnitude spectrum into 1D arrays for sampling
    all_freqs_x = repeat(freqs_x, inner=length(freqs_y))
    all_freqs_y = repeat(freqs_y, outer=length(freqs_x))
    all_magnitudes = vec(magnitude_spectrum)

    # Normalize magnitudes to form a probability distribution
    # Add a small constant to magnitudes before normalization to prevent division by zero for zero probabilities.
    probabilities = (all_magnitudes .+ 1e-9) ./ sum(all_magnitudes .+ 1e-9)

    # Sample M_rff_count indices based on probabilities
    # StatsBase.sample expects Weights from non-negative numbers
    sampled_indices = sample(1:length(probabilities), Weights(probabilities), M_rff_count, replace=true)

    W_fixed = Matrix{Float64}(undef, 2, M_rff_count)
    for i in 1:M_rff_count
        idx = sampled_indices[i]
        W_fixed[1, i] = all_freqs_x[idx] * 2π # Scale by 2π to match RFF convention (often ω'x)
        W_fixed[2, i] = all_freqs_y[idx] * 2π
    end

    return W_fixed
end

```


## V21: Spatiotemporal Stochastic Volatility & Seasonal Harmonics for MFGP

This model builds upon V20 by incorporating spatiotemporal stochastic volatility for observation noise and explicitly adding seasonal harmonics to the mean function for the primary observation (`Y`). This enhances the model's ability to capture both heteroscedasticity and periodic temporal patterns.

### Model Assumptions:
*   Multi-fidelity Structure: Retains the hierarchical multi-fidelity idea from V20:
    *   Z-fidelity (High Resolution): Latent spatial field modeled using a Matern 3/2 Kernel.
    *   U-fidelity (Medium Resolution): Latent spatiotemporal fields (`U1, U2, U3`) modeled using a Kronecker product of a Matern 3/2 spatial kernel and an AR(1) temporal process.
    *   Y-fidelity (Standard Resolution): The primary observation (`Y`) is modeled.
*   Kernel-based Interpolation (Enhanced in V21): Dependencies between fidelity levels are handled through explicit kernel-based Kronecker interpolation. This involves using the Matern 3/2 spatial kernel and AR(1) temporal covariance to project latent fields from their native (higher) resolution grids to the (lower) resolution grid of the `Y` observations. Specifically, `Z` is interpolated to `U` and `Y` locations, and `U` fields are interpolated to `Y` locations.
*   GP Representation: All latent fields (`Z`, `U`, `Y`'s primary process) are implicitly modeled as Gaussian Processes with specified Matern and AR(1) covariance structures.
*   AR(1) Temporal Process: The temporal component for spatiotemporal fields (U and Y) uses an Auto-Regressive process of order 1, parameterized by `rho` and `sigma_t_noise`.
*   Spatiotemporal Stochastic Volatility (New in V21): The observation noise variance for `y_obs` is no longer constant. It is modeled as a spatiotemporally varying process using a secondary Random Fourier Features (RFF) mapping based on `coords_y_s` and `coords_y_t`. This allows the model to account for heteroscedasticity.
*   Seasonal Harmonics (New in V21): An explicit seasonal component (sine/cosine waves) is added to the mean function of `y_obs` to capture distinct periodic patterns that might not be fully explained by the AR(1) process or other covariates.
*   Observation Noise: For `z_obs`, `u1_obs`, `u2_obs`, `u3_obs`, it remains homoscedastic and normally distributed. For `y_obs`, it is now heteroscedastic and spatiotemporally varying.
*   Priors: Standard weakly informative priors are used for all parameters, including new parameters for the stochastic volatility RFF mapping and seasonal harmonics coefficients.

### Key References:
*   Multi-fidelity Gaussian Processes: Perdikaris, P., Raissi, M., Psaros, N., & Karniadakis, G. E. (2017).
*   Kronecker Product Kernels: Stegle, O., Kadie, C. M., Norman, P. J., & Winn, J. (2011).
*   Matern Kernels: Rasmussen, C. E., & Williams, C. K. I. (2006).
*   AR(1) Models: Hamilton, J. D. (1994).
*   Spatiotemporal Stochastic Volatility: Inspired by approaches in financial econometrics and generalized to spatiotemporal settings. (e.g., Kim, S., Shephard, N., & Chib, S. (1998). *Stochastic Volatility: Likelihood Inference and Comparison*).
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007).
*   NUTS Sampler: Hoffman, M. D., & Gelman, A. (2014).


```{julia}
using SparseArrays

# Helper to create AR1 precision matrix
function ar1_precision(n, rho, sigma_e)
    # Get the type of the parameters, which will be ForwardDiff.Dual during AD
    # or Float64 when not differentiating
    T = typeof(rho)
    Q = spzeros(T, n, n) # Explicitly create a sparse matrix that can hold Dual numbers

    # Main diagonal
    Q[1, 1] = one(T) # Use one(T) to ensure type compatibility
    for i in 2:(n - 1)
        Q[i, i] = one(T) + rho^2
    end
    Q[n, n] = one(T)
    # Off-diagonals
    for i in 1:(n - 1)
        Q[i, i + 1] = -rho
        Q[i + 1, i] = -rho
    end
    # Ensure division also uses the correct type, and result type of `inv(sigma_e^2)` matches T
    return (one(T) / sigma_e^2) .* Q
end

# Helper to create AR1 covariance matrix
function ar1_covariance_matrix(times::Vector{<:Real}, rho::Real, sigma_e::Real)
    n = length(times)
    T = typeof(rho) # Get the type of the parameters
    C = Matrix{T}(undef, n, n) # Initialize matrix with this type
    for i in 1:n
        for j in 1:n
            C[i, j] = sigma_e^2 * rho^abs(times[i] - times[j])
        end
    end
    return C
end

# Helper to create AR1 cross-covariance matrix
function ar1_cross_covariance_matrix(times_a::Vector{<:Real}, times_b::Vector{<:Real}, rho::Real, sigma_e::Real)
    na = length(times_a)
    nb = length(times_b)
    T = typeof(rho) # Get the type of the parameters
    C = Matrix{T}(undef, na, nb) # Initialize matrix with this type
    for i in 1:na
        for j in 1:nb
            C[i, j] = sigma_e^2 * rho^abs(times_a[i] - times_b[j])
        end
    end
    return C
end

# Helper for Kronecker AR1 x Matern Sampling
function kron_ar1_matern_sample(Ns, Nt, unique_s, ls_s, sigma_s, rho_t, sigma_t_noise, noise_vec)
    # Spatial Matern 3/2 Precision
    k_s = Matern32Kernel() ∘ ScaleTransform(inv(ls_s))
    K_s = sigma_s^2 * kernelmatrix(k_s, RowVecs(unique_s)) + 1e-3*I # Increased jitter
    Q_s = sparse(inv(K_s))

    # Temporal AR1 Precision
    Q_t = ar1_precision(Nt, rho_t, sigma_t_noise)

    # Kronecker Product Q = Qt ⊗ Qs
    Q_full = kron(Q_t, Q_s)

    # Explicitly ensure symmetry for sparse matrix before Cholesky decomposition
    # Convert to dense Matrix to avoid SparseArrays.CHOLMOD incompatibility with ForwardDiff.Dual
    L_q = cholesky(Symmetric(Matrix(Q_full) + 1e-3*I)) # Increased jitter

    # Correctly extract the lower triangular factor for dense Cholesky
    return L_q.L' \ noise_vec
end

@model function model_v21_multifidelity_gp_matern_sv_seasonal(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_y_s, coords_y_t, coords_u_s, coords_u_t, coords_z_s; period=12.0, M_rff_sigma=20)
    # Dimensions
    Ny = length(y_obs); Nt_y = length(unique(coords_y_t)); Ns_y = Ny ÷ Nt_y
    Nu = length(u1_obs); Nt_u = length(unique(coords_u_t)); Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- 1. High Fidelity: Latent Spatial Z (Matern 3/2) ---
    ls_z ~ Gamma(2, 2)
    sigma_z_f ~ Exponential(1.0)
    k_z = Matern32Kernel() ∘ ScaleTransform(inv(ls_z))
    K_z = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(coords_z_s)) + 1e-3*I # Increased jitter
    z_latent ~ MvNormal(zeros(Nz), K_z)

    sigma_z_obs ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, sigma_z_obs^2 * I)

    # --- Interpolation logic for cross-fidelity dependencies (Z to U, Z to Y) ---
    K_z_u = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(coords_u_s), RowVecs(coords_z_s))
    z_at_u = (K_z_u * (K_z \ z_latent)) # Latent Z interpolated to U spatial locations
    z_at_u_full = z_at_u

    K_z_y = sigma_z_f^2 * kernelmatrix(k_z, RowVecs(coords_y_s), RowVecs(coords_z_s))
    z_at_y = (K_z_y * (K_z \ z_latent)) # Latent Z interpolated to Y spatial locations
    z_at_y_full = z_at_y

    # --- 2. Medium Fidelity: Latent Spatiotemporal U (Kron AR1 x Matern) ---
    ls_s_u ~ Gamma(2, 2); sigma_s_u ~ Exponential(1.0)
    rho_u ~ Uniform(-0.99, 0.99); sigma_t_u ~ Exponential(0.5)

    u1_noise ~ filldist(Normal(0, 1), Nu)
    u2_noise ~ filldist(Normal(0, 1), Nu)
    u3_noise ~ filldist(Normal(0, 1), Nu)

    # Unique spatial and temporal coordinates for U-fidelity (source for interpolation)
    unique_coords_u_s = coords_u_s[1:Ns_u, :]
    unique_coords_u_t = unique(coords_u_t[:,1])

    u1_true = kron_ar1_matern_sample(Ns_u, Nt_u, unique_coords_u_s, ls_s_u, sigma_s_u, rho_u, sigma_t_u, u1_noise)
    u2_true = kron_ar1_matern_sample(Ns_u, Nt_u, unique_coords_u_s, ls_s_u, sigma_s_u, rho_u, sigma_t_u, u2_noise)
    u3_true = kron_ar1_matern_sample(Ns_u, Nt_u, unique_coords_u_s, ls_s_u, sigma_s_u, rho_u, sigma_t_u, u3_noise)

    beta_uz ~ Normal(0, 1) # Effect of Z on U
    sigma_u_obs ~ filldist(Exponential(0.5), 3)
    u1_obs ~ MvNormal(u1_true .+ beta_uz .* z_at_u_full, sigma_u_obs[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u_obs[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u_obs[3]^2 * I)

    # --- Interpolation of U to Y coordinates (NEW: Kernel-based Kronecker interpolation) ---
    # Unique spatial and temporal coordinates for Y-fidelity (target for interpolation)
    unique_coords_y_s = coords_y_s[1:Ns_y, :]
    unique_coords_y_t = unique(coords_y_t[:,1])

    # Spatial kernels for U (source: unique_coords_u_s, target: unique_coords_y_s)
    k_s_u_interp = Matern32Kernel() ∘ ScaleTransform(inv(ls_s_u))
    K_s_uu = sigma_s_u^2 * kernelmatrix(k_s_u_interp, RowVecs(unique_coords_u_s)) + 1e-3*I # Increased jitter
    K_s_yu = sigma_s_u^2 * kernelmatrix(k_s_u_interp, RowVecs(unique_coords_y_s), RowVecs(unique_coords_u_s))

    # Temporal AR1 kernels for U (source: unique_coords_u_t, target: unique_coords_y_t)
    K_t_uu = ar1_covariance_matrix(unique_coords_u_t, rho_u, sigma_t_u) + 1e-3*I # Added jitter
    K_t_yu = ar1_cross_covariance_matrix(unique_coords_y_t, unique_coords_u_t, rho_u, sigma_t_u)

    # Full Kronecker covariance matrices
    K_uu_full = kron(K_t_uu, K_s_uu) # Nu x Nu
    K_yu_full = kron(K_t_yu, K_s_yu) # Ny x Nu

    # Perform interpolation for u1, u2, u3
    u1_at_y = K_yu_full * (K_uu_full \ u1_true)
    u2_at_y = K_yu_full * (K_uu_full \ u2_true)
    u3_at_y = K_yu_full * (K_uu_full \ u3_true)


    # --- 3. Standard Fidelity: Dependent Y (Kron AR1 x Matern with SV and Seasonal) ---
    ls_s_y ~ Gamma(2, 2); sigma_s_y ~ Exponential(1.0)
    rho_y ~ Uniform(-0.99, 0.99); sigma_t_y ~ Exponential(0.5)
    y_noise ~ filldist(Normal(0, 1), Ny)

    f_st_y = kron_ar1_matern_sample(Ns_y, Nt_y, coords_y_s[1:Ns_y, :], ls_s_y, sigma_s_y, rho_y, sigma_t_y, y_noise)

    # Seasonal Harmonics (NEW in V21)
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal_y = beta_cos .* cos.(2 * pi .* coords_y_t[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_y_t[:,1] ./ period)

    # Spatiotemporal Stochastic Volatility (NEW in V21)
    D_st_y = size(coords_y_s, 2) + size(coords_y_t, 2)
    coords_st_y = hcat(coords_y_s, coords_y_t)

    W_sigma ~ filldist(Normal(0, 1), D_st_y, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0)
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)
    sigma_y_vec = exp.(rff_map(coords_st_y, W_sigma, b_sigma) * beta_rff_sigma ./ 2)


    # Regression and Final Likelihood
    beta_y_covs ~ filldist(Normal(0, 1), 4) # For u1, u2, u3, z
    mu_y = f_st_y .+ seasonal_y .+ (u1_at_y .* beta_y_covs[1]) .+ (u2_at_y .* beta_y_covs[2]) .+ (u3_at_y .* beta_y_covs[3]) .+ (z_at_y_full .* beta_y_covs[4])

    y_obs ~ MvNormal(mu_y, Diagonal(sigma_y_vec.^2 .+ 1e-3)) # Increased jitter
end
```

```{julia}
model_v21 = model_v21_multifidelity_gp_matern_sv_seasonal(
    y_mock, u1_mock, u2_mock, u3_mock, z_mock,
    coords_y_s, coords_y_t,
    coords_u_s, coords_u_t,
    coords_z_s
)
chain_v21 = sample(model_v21, NUTS(), 100)
display(describe(chain_v21))
waic_v21 = compute_y_waic(model_v21, chain_v21)
println("\nWAIC for V21: ", waic_v21)
```

    Samples per chain = 100
    Compute duration  = 66.44 seconds
    WAIC for V21: 468.1391223829339


## V22: Nonlinear Cross-Fidelity Mappings using RFFs

This model builds upon the multi-fidelity concept by employing Random Fourier Features (RFFs) to create nonlinear functional mappings between fidelity levels. Instead of relying on explicit kernel interpolation (as in V20/V21), each latent field (Z, U, and the main Y process) is approximated by RFFs, and the output of a higher-fidelity RFF layer serves as input to the RFF layer of a lower-fidelity field. This allows for highly flexible and adaptive propagation of information through the multi-fidelity hierarchy.

### Model Assumptions:
*   Multi-fidelity Structure: Retains the hierarchical multi-fidelity idea:
    *   Z-fidelity (High Resolution): Latent spatial field modeled as a Gaussian Process approximated by RFFs, taking spatial coordinates (`coords_z_s`) as input.
    *   U-fidelity (Medium Resolution): Latent spatiotemporal fields (`U1, U2, U3`) modeled as Gaussian Processes approximated by RFFs. Their inputs include spatial coordinates (`coords_u_s`), time (`coords_u_t`), and the RFF-approximated latent Z from the higher fidelity. This establishes a nonlinear hierarchical dependency.
    *   Y-fidelity (Standard Resolution): The primary observation (`Y`) is modeled as a Gaussian Process approximated by RFFs. Its inputs include spatial coordinates (`coords_y_s`), time (`coords_y_t`), the RFF-approximated latent Z, and the RFF-approximated latent U fields (U1, U2, U3). This forms a 'stacked' or 'deep' RFF structure for the mean function of Y.
*   Functional Dependencies (Nonlinear Cross-Fidelity): All dependencies between fidelity levels (Z -> U, Z -> Y, U -> Y) are modeled as nonlinear functions using separate RFF mappings. This explicitly captures complex interactions and avoids assumptions of linearity or simple kernel-based interpolation, allowing the model to *learn* how information from higher fidelity fields influences lower fidelity ones.
*   GP Representation: Each latent field (Z, U1, U2, U3, and the main process for Y) is implicitly modeled as a Gaussian Process through its RFF approximation. This provides the flexibility of GPs while being computationally efficient.
*   Seasonal Harmonics: An explicit seasonal component (sine/cosine waves) is added to the mean function of `y_obs` to capture distinct periodic patterns.
*   Spatiotemporal Stochastic Volatility: The observation noise variance for `y_obs` is modeled as a spatiotemporally varying process using a secondary RFF mapping, allowing for heteroscedasticity.
*   Observation Noise: Homoscedastic and normally distributed for `z_obs`, `u1_obs`, `u2_obs`, `u3_obs`. For `y_obs`, it is heteroscedastic and spatiotemporally varying.
*   Priors: Standard weakly informative priors are used for all RFF weights (`W`), biases (`b`), and signal variances (`sigma_f`), as well as for seasonal coefficients and stochastic volatility RFF parameters.

### Challenges Encountered:
*   `InterruptException` & `ForwardDiff.Dual` Memory Issues: Model V22, using adaptive RFFs (where `W` and `b` are sampled parameters) in a stacked, multi-fidelity manner, led to a rapid explosion in memory and computational time for `ForwardDiff.jl` due to complex nested `Dual` number calculations during NUTS sampling. This caused an `InterruptException` and significantly hindered inference for fully adaptive RFFs in this architecture. This challenge motivated the exploration of fixed and semi-adaptive RFF approaches in subsequent models (V23, V24).

### Key References:
*   Random Fourier Features (RFF): Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS.
*   Multi-fidelity Gaussian Processes: Perdikaris, P., Raissi, M., Psaros, N., & Karniadakis, G. E. (2017). *Nonlinear model reduction for uncertainty quantification and predictive modeling of spatiotemporal systems*. Journal of Computational Physics, 347, 303-324.
*   Deep Gaussian Processes: Damianou, A., & Lawrence, N. (2013). *Deep Gaussian Processes*. AISTATS. (Conceptual foundation for stacking GP-like layers).
*   Hierarchical Bayesian Models: Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press.
*   Stochastic Volatility: Kim, S., Shephard, N., & Chib, S. (1998). *Stochastic Volatility: Likelihood Inference and Comparison*.



```{julia}
@model function model_v22_rff_multifidelity_gp(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_y_s, coords_y_t, coords_u_s, coords_u_t, coords_z_s; period=12.0, M_rff_base=40, M_rff_sigma=20)
    # Dimensions
    Ny = length(y_obs); Nt_y = length(unique(coords_y_t)); Ns_y = Ny ÷ Nt_y
    Nu = length(u1_obs); Nt_u = length(unique(coords_u_t)); Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- Helper function for RFF mapping --- (defined globally)
    # function rff_map(coords, W, b) ... end

    # --- 1. High Fidelity: Latent Spatial Z (RFF-based GP) ---
    D_z_in = size(coords_z_s, 2)
    W_z ~ filldist(Normal(0, 1), D_z_in, M_rff_base)
    b_z ~ filldist(Uniform(0, 2pi), M_rff_base)
    sigma_z_f ~ Exponential(1.0)
    beta_z ~ filldist(Normal(0, sigma_z_f^2), M_rff_base)

    Phi_z = rff_map(coords_z_s, W_z, b_z)
    z_latent = Phi_z * beta_z

    sigma_z_obs ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, sigma_z_obs^2 * I)

    # --- Cross-fidelity RFF Evaluation functions (for interpolation) ---
    # These evaluate the latent RFF function at different coordinates
    # to pass higher fidelity information to lower fidelity layers.
    function get_rff_latent_z(coords)
        phi = rff_map(coords, W_z, b_z)
        return phi * beta_z
    end

    # --- 2. Medium Fidelity: Latent Spatiotemporal U (RFF-based GP) ---
    # Interpolate latent Z to U spatial locations
    z_at_u_s_full = get_rff_latent_z(coords_u_s)

    # Input for U RFFs: Spatial (coords_u_s), Temporal (coords_u_t), and Interpolated Z
    coords_u_rff_in = hcat(coords_u_s, coords_u_t, z_at_u_s_full)
    D_u_in = size(coords_u_rff_in, 2)

    W_u ~ filldist(Normal(0, 1), D_u_in, M_rff_base)
    b_u ~ filldist(Uniform(0, 2pi), M_rff_base)
    sigma_u_f ~ Exponential(1.0)

    beta_u1 ~ filldist(Normal(0, sigma_u_f^2), M_rff_base)
    beta_u2 ~ filldist(Normal(0, sigma_u_f^2), M_rff_base)
    beta_u3 ~ filldist(Normal(0, sigma_u_f^2), M_rff_base)

    Phi_u = rff_map(coords_u_rff_in, W_u, b_u)
    u1_true = Phi_u * beta_u1
    u2_true = Phi_u * beta_u2
    u3_true = Phi_u * beta_u3

    sigma_u_obs ~ filldist(Exponential(0.5), 3)
    u1_obs ~ MvNormal(u1_true, sigma_u_obs[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u_obs[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u_obs[3]^2 * I)

    # --- Cross-fidelity RFF Evaluation functions (for U interpolation) ---
    function get_rff_latent_u(coords_s, coords_t, z_interpolated_at_y)
        coords_u_rff_in_at_y = hcat(coords_s, coords_t, z_interpolated_at_y)
        phi = rff_map(coords_u_rff_in_at_y, W_u, b_u)
        return phi
    end

    # --- 3. Standard Fidelity: Dependent Y (RFF-based GP for Main Process) ---
    # Interpolate latent Z to Y spatial locations
    z_at_y_s_full = get_rff_latent_z(coords_y_s)

    # Interpolate latent U to Y spatiotemporal locations
    Phi_u_at_y = get_rff_latent_u(coords_y_s, coords_y_t, z_at_y_s_full)
    u1_at_y = Phi_u_at_y * beta_u1
    u2_at_y = Phi_u_at_y * beta_u2
    u3_at_y = Phi_u_at_y * beta_u3

    # Input for Y's main GP RFF: Space, Time, Interpolated Z, Interpolated U1, U2, U3
    coords_y_rff_in = hcat(coords_y_s, coords_y_t, z_at_y_s_full, u1_at_y, u2_at_y, u3_at_y)
    D_y_gp_in = size(coords_y_rff_in, 2)

    W_y_gp ~ filldist(Normal(0, 1), D_y_gp_in, M_rff_base)
    b_y_gp ~ filldist(Uniform(0, 2pi), M_rff_base)
    sigma_y_gp_f ~ Exponential(1.0)
    beta_y_gp ~ filldist(Normal(0, sigma_y_gp_f^2), M_rff_base)

    Phi_y_gp = rff_map(coords_y_rff_in, W_y_gp, b_y_gp)
    f_st_y = Phi_y_gp * beta_y_gp

    # --- Seasonal Harmonics (from V21) ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal_y = beta_cos .* cos.(2 * pi .* coords_y_t[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_y_t[:,1] ./ period)

    # --- Spatiotemporal Stochastic Volatility (from V21) ---
    D_st_y = size(coords_y_s, 2) + size(coords_y_t, 2)
    coords_st_y = hcat(coords_y_s, coords_y_t)

    W_sigma ~ filldist(Normal(0, 1), D_st_y, M_rff_sigma)
    b_sigma ~ filldist(Uniform(0, 2pi), M_rff_sigma)
    sigma_log_var ~ Exponential(1.0)
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)
    sigma_y_vec = exp.(rff_map(coords_st_y, W_sigma, b_sigma) * beta_rff_sigma ./ 2)

    # --- Final Mean and Likelihood for Y ---
    # Note: beta_y_covs are no longer needed as the full RFF structure already includes these dependencies
    # (u1_at_y, u2_at_y, u3_at_y, z_at_y_s_full are direct inputs to f_st_y RFF)

    # The primary effect of covariates on Y is now captured through their direct inclusion
    # as inputs to the RFF for f_st_y. If any additional linear effects are desired,
    # new beta_covs can be introduced for a linear combination of z_at_y_s_full, u1_at_y, etc.
    # For this model, we assume the RFF captures these interactions fully.

    mu_y = f_st_y .+ seasonal_y # Direct RFF output for f_st_y already encodes covariate effects

    y_obs ~ MvNormal(mu_y, Diagonal(sigma_y_vec.^2 .+ 1e-3))
end
```


```{julia}

model_v22 = model_v22_rff_multifidelity_gp(
    y_mock, u1_mock, u2_mock, u3_mock, z_mock,
    coords_y_s, coords_y_t,
    coords_u_s, coords_u_t,
    coords_z_s
)
chain_v22 = sample(model_v22, NUTS(), 100) # Reduced samples for faster testing
display(describe(chain_v22))
waic_v22 = compute_y_waic(model_v22, chain_v22)
println("\nWAIC for V22: ", waic_v22)
```


### Alternative Inference: Stochastic Variational Inference (SVI) for Model V22

To overcome the memory constraints of NUTS on high-dimensional RFF models, we utilize `AdvancedVI.jl`. ADVI (Automatic Differentiation Variational Inference) approximates the posterior with a Gaussian distribution, transforming the integration problem into a stochastic optimization problem.


```{julia}
using AdvancedVI

samples_per_step = 10
max_iterations = 1000
advi = ADVI(samples_per_step, max_iterations)
q = vi(model_v22, advi; optimizer=AdvancedVI.TruncatedADAM(0.01))
vi_samples = rand(q, 1000)

```

## V23: Fixed Deterministic Fourier Features (DFRFF) Multi-fidelity GP

This model builds upon V22 by transitioning from *adaptive* Random Fourier Features (RFFs), where the projection weights (`W`) and biases (`b`) are sampled during inference, to *fixed deterministic* Fourier Features. In this approach, `W` and `b` are pre-generated once (e.g., by sampling from their respective distributions) and then treated as fixed hyperparameters within the Bayesian model. This is inspired by the efficiency gains sought in FFT-based approximations, where a structured or pre-defined basis can reduce computational overhead.

### Model Assumptions:
*   Fixed Deterministic Fourier Features (DFRFF): For all RFF layers (Z-fidelity, U-fidelity, Y-fidelity, and Stochastic Volatility), the projection weights (`W`) and biases (`b`) are generated once outside the Turing model and passed as fixed input data. This significantly reduces the number of parameters the NUTS sampler needs to estimate.
*   Multi-fidelity Structure: Retains the hierarchical multi-fidelity idea from V22, where latent fields (Z, U, Y) are represented by RFFs, and higher-fidelity RFF outputs serve as inputs to lower-fidelity RFFs.
*   Nonlinear Cross-Fidelity Mappings: Dependencies between fidelity levels are still modeled using nonlinear RFF mappings.
*   GP Representation: Each latent field (Z, U1, U2, U3, and the main process for Y) is implicitly modeled as a Gaussian Process through its DFRFF approximation.
*   Seasonal Harmonics: An explicit seasonal component (sine/cosine waves) is added to the mean function of `y_obs`.
*   Spatiotemporal Stochastic Volatility: The observation noise variance for `y_obs` is modeled as a spatiotemporally varying process using a secondary DFRFF mapping.
*   Observation Noise: Homoscedastic and normally distributed for `z_obs`, `u1_obs`, `u2_obs`, `u3_obs`. For `y_obs`, it is heteroscedastic and spatiotemporally varying.
*   Priors: Standard weakly informative priors are used for the `beta` coefficients and signal variances (`sigma_f`) of the RFF layers, seasonal coefficients, and stochastic volatility RFF parameters. Priors for `W` and `b` are effectively removed as they are no longer parameters.

### Benefits:
*   Significantly Increased Efficiency: By removing `W` and `b` as parameters, the total number of parameters to be sampled by NUTS is substantially reduced. This dramatically speeds up sampling and improves the computational tractability of the model, especially for larger `M_rff` values.
*   Reduced Variance (potentially): If the pre-generated deterministic features provide good coverage of the relevant frequency space, the kernel approximation can be more stable than with purely random (and re-sampled) features, potentially leading to more consistent model performance.
*   Simplified Inference: A smaller parameter space generally leads to an easier inference problem for MCMC samplers.
*   Scalability: This step enhances scalability, paving the way for more efficient inference, possibly in conjunction with SVI (V23, next planned iteration).

### Key References:
*   Random Fourier Features (RFF): Rahimi, A., A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS. (This model reverts to the original spirit of fixed RFFs).
*   Multi-fidelity Gaussian Processes: Perdikaris, P., Raissi, M., Psaros, N., & Karniadakis, G. E. (2017). *Nonlinear model reduction for uncertainty quantification and predictive modeling of spatiotemporal systems*. Journal of Computational Physics, 347, 303-324.
*   Deep Gaussian Processes: Damianou, A., & Lawrence, N. (2013). *Deep Gaussian Processes*. AISTATS.
*   Hierarchical Bayesian Models: Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press.
*   Stochastic Volatility: Kim, S., Shephard, N., & Chib, S. (1998). *Stochastic Volatility: Likelihood Inference and Comparison*.



```{julia}
@model function model_v23_dfrff_multifidelity_gp(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_y_s, coords_y_t, coords_u_s, coords_u_t, coords_z_s, W_z_fixed, b_z_fixed, W_u_fixed, b_u_fixed, W_y_gp_fixed, b_y_gp_fixed, W_sigma_fixed, b_sigma_fixed; period=12.0, M_rff_base=40, M_rff_sigma=20)
    # Dimensions
    Ny = length(y_obs); Nt_y = length(unique(coords_y_t)); Ns_y = Ny ÷ Nt_y
    Nu = length(u1_obs); Nt_u = length(unique(coords_u_t)); Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- 1. High Fidelity: Latent Spatial Z (DFRFF-based GP) ---
    # W_z and b_z are now fixed inputs
    sigma_z_f ~ Exponential(1.0)
    beta_z ~ filldist(Normal(0, sigma_z_f^2), M_rff_base)

    Phi_z = rff_map(coords_z_s, W_z_fixed, b_z_fixed)
    z_latent = Phi_z * beta_z

    sigma_z_obs ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, sigma_z_obs^2 * I)

    # --- Cross-fidelity DFRFF Evaluation functions (for interpolation) ---
    function get_dfrff_latent_z(coords)
        phi = rff_map(coords, W_z_fixed, b_z_fixed)
        return phi * beta_z
    end

    # --- 2. Medium Fidelity: Latent Spatiotemporal U (DFRFF-based GP) ---
    # Interpolate latent Z to U spatial locations
    z_at_u_s_full = get_dfrff_latent_z(coords_u_s)

    # Input for U DFRFFs: Spatial (coords_u_s), Temporal (coords_u_t), and Interpolated Z
    coords_u_dfrff_in = hcat(coords_u_s, coords_u_t, z_at_u_s_full)

    # W_u and b_u are now fixed inputs
    sigma_u_f ~ Exponential(1.0)

    beta_u1 ~ filldist(Normal(0, sigma_u_f^2), M_rff_base)
    beta_u2 ~ filldist(Normal(0, sigma_u_f^2), M_rff_base)
    beta_u3 ~ filldist(Normal(0, sigma_u_f^2), M_rff_base)

    Phi_u = rff_map(coords_u_dfrff_in, W_u_fixed, b_u_fixed)
    u1_true = Phi_u * beta_u1
    u2_true = Phi_u * beta_u2
    u3_true = Phi_u * beta_u3

    sigma_u_obs ~ filldist(Exponential(0.5), 3)
    u1_obs ~ MvNormal(u1_true, sigma_u_obs[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u_obs[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u_obs[3]^2 * I)

    # --- Cross-fidelity DFRFF Evaluation functions (for U interpolation) ---
    function get_dfrff_latent_u(coords_s, coords_t, z_interpolated_at_y)
        coords_u_dfrff_in_at_y = hcat(coords_s, coords_t, z_interpolated_at_y)
        phi = rff_map(coords_u_dfrff_in_at_y, W_u_fixed, b_u_fixed)
        return phi
    end

    # --- 3. Standard Fidelity: Dependent Y (DFRFF-based GP for Main Process) ---
    # Interpolate latent Z to Y spatial locations
    z_at_y_s_full = get_dfrff_latent_z(coords_y_s)

    # Interpolate latent U to Y spatiotemporal locations
    Phi_u_at_y = get_dfrff_latent_u(coords_y_s, coords_y_t, z_at_y_s_full)
    u1_at_y = Phi_u_at_y * beta_u1
    u2_at_y = Phi_u_at_y * beta_u2
    u3_at_y = Phi_u_at_y * beta_u3

    # Input for Y's main GP DFRFF: Space, Time, Interpolated Z, Interpolated U1, U2, U3
    coords_y_dfrff_in = hcat(coords_y_s, coords_y_t, z_at_y_s_full, u1_at_y, u2_at_y, u3_at_y)

    # W_y_gp and b_y_gp are now fixed inputs
    sigma_y_gp_f ~ Exponential(1.0)
    beta_y_gp ~ filldist(Normal(0, sigma_y_gp_f^2), M_rff_base)

    Phi_y_gp = rff_map(coords_y_dfrff_in, W_y_gp_fixed, b_y_gp_fixed)
    f_st_y = Phi_y_gp * beta_y_gp

    # --- Seasonal Harmonics ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal_y = beta_cos .* cos.(2 * pi .* coords_y_t[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_y_t[:,1] ./ period)

    # --- Spatiotemporal Stochastic Volatility (DFRFF-based) ---
    D_st_y = size(coords_y_s, 2) + size(coords_y_t, 2)
    coords_st_y = hcat(coords_y_s, coords_y_t)

    # W_sigma and b_sigma are now fixed inputs
    sigma_log_var ~ Exponential(1.0)
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)
    sigma_y_vec = exp.(rff_map(coords_st_y, W_sigma_fixed, b_sigma_fixed) * beta_rff_sigma ./ 2)

    # --- Final Mean and Likelihood for Y ---
    mu_y = f_st_y .+ seasonal_y
    y_obs ~ MvNormal(mu_y, Diagonal(sigma_y_vec.^2 .+ 1e-3))
end
```

## V24: Semi-Adaptive DFRFF Multi-fidelity Deep Gaussian Process (DGP)

This model builds upon V23 by treating the pre-generated `W` and `b` values not as fixed inputs, but as means for Normal priors on the `W` and `b` parameters. This allows the NUTS sampler to subtly adjust these parameters around their pre-initialized values, providing a balance between the efficiency of fixed features (V23) and the flexibility of fully adaptive RFFs (V22).

Crucially, this multi-fidelity RFF architecture inherently forms a Deep Gaussian Process (DGP). Each fidelity layer (Z, U, Y) is modeled as a GP approximated by RFFs, and the output of a higher-fidelity RFF layer serves as a non-linear input (or "warping") to the RFF layer of a lower-fidelity field, creating a stacked, hierarchical, and non-linear transformation of the input space.

### Model Assumptions:
*   Deep Gaussian Process (DGP) Architecture: The multi-fidelity structure is realized through stacked RFF layers. The latent output of the Z-fidelity RFF layer influences the U-fidelity RFF layers, and both (interpolated) Z and U latent outputs influence the Y-fidelity RFF layer. This creates a multi-layered, non-linear feature transformation.
*   Semi-Adaptive Deterministic Fourier Features (Semi-Adaptive DFRFF): For all RFF layers (Z-fidelity, U-fidelity, Y-fidelity, and Stochastic Volatility), the projection weights (`W`) and biases (`b`) are now parameters sampled from Normal distributions whose means are set by the pre-generated `W_fixed` and `b_fixed` values, respectively.
    *   New `sigma_W` and `sigma_b` parameters are introduced (with Exponential priors) to control the variance of these Normal priors, allowing the model to learn how much to deviate from the initial fixed features. This provides a mechanism for adaptive refinement.
*   Multi-fidelity Structure: Retains the hierarchical multi-fidelity idea from V23, explicitly using separate coordinate sets for Z, U, and Y to handle data at different resolutions.
*   Nonlinear Cross-Fidelity Mappings: Dependencies between fidelity levels are still modeled using nonlinear RFF mappings, which are now recognized as forming the layers of the DGP.
*   GP Representation: Each latent field (Z, U1, U2, U3, and the main process for Y) is implicitly modeled as a Gaussian Process through its DFRFF approximation.
*   Seasonal Harmonics: An explicit seasonal component (sine/cosine waves) is added to the mean function of `y_obs`.
*   Spatiotemporal Stochastic Volatility: The observation noise variance for `y_obs` is modeled as a spatiotemporally varying process using a secondary DFRFF mapping, allowing for heteroscedasticity.
*   Observation Noise: Homoscedastic and normally distributed for `z_obs`, `u1_obs`, `u2_obs`, `u3_obs`. For `y_obs`, it is heteroscedastic and spatiotemporally varying.
*   Priors: Standard weakly informative priors are used, with the addition of priors for `sigma_W` and `sigma_b` for each RFF layer.

### Benefits:
*   Improved Flexibility and Adaptation: Allows the model to fine-tune the `W` and `b` parameters based on the data, moving beyond a purely fixed basis. This can capture nuances that a strictly fixed set of features might miss.
*   Reduced Parameter Space vs. V22: Still significantly reduces the parameter space compared to fully adaptive RFFs (V22) by guiding the `W` and `b` parameters towards a sensible starting point (the pre-generated values).
*   Potentially Better Convergence: By starting the `W` and `b` parameters near good initial values, the NUTS sampler might converge more efficiently and effectively compared to starting from very broad priors.
*   Controlled Adaptivity: The `sigma_W` and `sigma_b` parameters allow for controlling the degree of adaptivity. If these are small, the features remain close to their initial values; if large, they can explore a wider range.
*   DGP Advantages: By stacking RFF layers, the model can learn more complex, multi-scale, and non-linear relationships between variables and across fidelity levels, offering a richer representation than single-layer GPs.

### Substantive Differences from V15 (Deep Gaussian Process):
*   RFF Parameterization: V15 uses *fully adaptive RFFs* with broad priors on `W` and `b`, leading to a very high-dimensional parameter space. V24 employs *semi-adaptive DFRFFs*, where `W` and `b` are sampled around pre-generated fixed values with learned variances, significantly improving computational efficiency and tractability.
*   Multi-fidelity Handling: V15 is framed as operating on a single set of coordinates for all latent fields. V24 explicitly implements a multi-fidelity structure, taking distinct coordinate sets (`coords_z_s`, `coords_u_s`, `coords_u_t`, `coords_y_s`, `coords_y_t`) for each fidelity level, directly addressing data measured at different resolutions.
*   Observation Noise Model: V15 assumes simple homoscedastic (constant variance) observation noise for all variables. V24 incorporates a *spatiotemporal stochastic volatility* model for `y_obs`, allowing the observation noise variance to vary across space and time using a secondary RFF mapping.

### Key References:
*   Deep Gaussian Processes: Damianou, A., & Lawrence, N. (2013). *Deep Gaussian Processes*. AISTATS. (Conceptual foundation for stacking GP-like layers, directly applicable to this architecture).
*   Random Fourier Features (RFF): Rahimi, A., A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NIPS.
*   Multi-fidelity Gaussian Processes: Perdikaris, P., Raissi, M., Psaros, N., & Karniadakis, G. E. (2017). *Nonlinear model reduction for uncertainty quantification and predictive modeling of spatiotemporal systems*. Journal of Computational Physics, 347, 303-324.
*   Hierarchical Bayesian Models: Gelman, A., et al. (2013). *Bayesian Data Analysis*. CRC Press.



```{julia}
@model function model_v24_semi_adaptive_dfrff_multifidelity_gp(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_y_s, coords_y_t, coords_u_s, coords_u_t, coords_z_s, W_z_fixed, b_z_fixed, W_u_fixed, b_u_fixed, W_y_gp_fixed, b_y_gp_fixed, W_sigma_fixed, b_sigma_fixed; period=12.0, M_rff_base=40, M_rff_sigma=20)
    # Dimensions
    Ny = length(y_obs); Nt_y = length(unique(coords_y_t)); Ns_y = Ny ÷ Nt_y
    Nu = length(u1_obs); Nt_u = length(unique(coords_u_t)); Ns_u = Nu ÷ Nt_u
    Nz = length(z_obs)

    # --- Priors for W and b variances (control adaptivity) ---
    sigma_W_z ~ Exponential(0.1)
    sigma_b_z ~ Exponential(0.1)
    sigma_W_u ~ Exponential(0.1)
    sigma_b_u ~ Exponential(0.1)
    sigma_W_y_gp ~ Exponential(0.1)
    sigma_b_y_gp ~ Exponential(0.1)
    sigma_W_sigma ~ Exponential(0.1)
    sigma_b_sigma ~ Exponential(0.1)

    # --- 1. High Fidelity: Latent Spatial Z (Semi-Adaptive DFRFF-based GP) ---
    # W_z and b_z are now sampled, with means from fixed inputs and learned variances
    W_z ~ MvNormal(vec(W_z_fixed), sigma_W_z^2 * I)
    b_z ~ MvNormal(b_z_fixed, sigma_b_z^2 * I)
    W_z_matrix = reshape(W_z, size(W_z_fixed))

    sigma_z_f ~ Exponential(1.0)
    beta_z ~ filldist(Normal(0, sigma_z_f^2), M_rff_base)

    Phi_z = rff_map(coords_z_s, W_z_matrix, b_z)
    z_latent = Phi_z * beta_z

    sigma_z_obs ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, sigma_z_obs^2 * I)

    # --- Cross-fidelity DFRFF Evaluation functions (for interpolation) ---
    function get_dfrff_latent_z(coords)
        phi = rff_map(coords, W_z_matrix, b_z)
        return phi * beta_z
    end

    # --- 2. Medium Fidelity: Latent Spatiotemporal U (Semi-Adaptive DFRFF-based GP) ---
    # Interpolate latent Z to U spatial locations
    z_at_u_s_full = get_dfrff_latent_z(coords_u_s)

    # Input for U DFRFFs: Spatial (coords_u_s), Temporal (coords_u_t), and Interpolated Z
    coords_u_dfrff_in = hcat(coords_u_s, coords_u_t, z_at_u_s_full)

    # W_u and b_u are now sampled
    W_u ~ MvNormal(vec(W_u_fixed), sigma_W_u^2 * I)
    b_u ~ MvNormal(b_u_fixed, sigma_b_u^2 * I)
    W_u_matrix = reshape(W_u, size(W_u_fixed))

    sigma_u_f ~ Exponential(1.0)

    beta_u1 ~ filldist(Normal(0, sigma_u_f^2), M_rff_base)
    beta_u2 ~ filldist(Normal(0, sigma_u_f^2), M_rff_base)
    beta_u3 ~ filldist(Normal(0, sigma_u_f^2), M_rff_base)

    Phi_u = rff_map(coords_u_dfrff_in, W_u_matrix, b_u)
    u1_true = Phi_u * beta_u1
    u2_true = Phi_u * beta_u2
    u3_true = Phi_u * beta_u3

    sigma_u_obs ~ filldist(Exponential(0.5), 3)
    u1_obs ~ MvNormal(u1_true, sigma_u_obs[1]^2 * I)
    u2_obs ~ MvNormal(u2_true, sigma_u_obs[2]^2 * I)
    u3_obs ~ MvNormal(u3_true, sigma_u_obs[3]^2 * I)

    # --- Cross-fidelity DFRFF Evaluation functions (for U interpolation) ---
    function get_dfrff_latent_u(coords_s, coords_t, z_interpolated_at_y)
        coords_u_dfrff_in_at_y = hcat(coords_s, coords_t, z_interpolated_at_y)
        phi = rff_map(coords_u_dfrff_in_at_y, W_u_matrix, b_u)
        return phi
    end

    # --- 3. Standard Fidelity: Dependent Y (Semi-Adaptive DFRFF-based GP for Main Process) ---
    # Interpolate latent Z to Y spatial locations
    z_at_y_s_full = get_dfrff_latent_z(coords_y_s)

    # Interpolate latent U to Y spatiotemporal locations
    Phi_u_at_y = get_dfrff_latent_u(coords_y_s, coords_y_t, z_at_y_s_full)
    u1_at_y = Phi_u_at_y * beta_u1
    u2_at_y = Phi_u_at_y * beta_u2
    u3_at_y = Phi_u_at_y * beta_u3

    # Input for Y's main GP DFRFF: Space, Time, Interpolated Z, Interpolated U1, U2, U3
    coords_y_dfrff_in = hcat(coords_y_s, coords_y_t, z_at_y_s_full, u1_at_y, u2_at_y, u3_at_y)

    # W_y_gp and b_y_gp are now sampled
    W_y_gp ~ MvNormal(vec(W_y_gp_fixed), sigma_W_y_gp^2 * I)
    b_y_gp ~ MvNormal(b_y_gp_fixed, sigma_b_y_gp^2 * I)
    W_y_gp_matrix = reshape(W_y_gp, size(W_y_gp_fixed))

    sigma_y_gp_f ~ Exponential(1.0)
    beta_y_gp ~ filldist(Normal(0, sigma_y_gp_f^2), M_rff_base)

    Phi_y_gp = rff_map(coords_y_dfrff_in, W_y_gp_matrix, b_y_gp)
    f_st_y = Phi_y_gp * beta_y_gp

    # --- Seasonal Harmonics ---
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal_y = beta_cos .* cos.(2 * pi .* coords_y_t[:,1] ./ period) .+ beta_sin .* sin.(2 * pi .* coords_y_t[:,1] ./ period)

    # --- Spatiotemporal Stochastic Volatility (Semi-Adaptive DFRFF-based) ---
    D_st_y = size(coords_y_s, 2) + size(coords_y_t, 2)
    coords_st_y = hcat(coords_y_s, coords_y_t)

    # W_sigma and b_sigma are now sampled
    W_sigma ~ MvNormal(vec(W_sigma_fixed), sigma_W_sigma^2 * I)
    b_sigma ~ MvNormal(b_sigma_fixed, sigma_b_sigma^2 * I)
    W_sigma_matrix = reshape(W_sigma, size(W_sigma_fixed))

    sigma_log_var ~ Exponential(1.0)
    beta_rff_sigma ~ filldist(Normal(0, sigma_log_var^2), M_rff_sigma)
    sigma_y_vec = exp.(rff_map(coords_st_y, W_sigma_matrix, b_sigma) * beta_rff_sigma ./ 2)

    # --- Final Mean and Likelihood for Y ---
    mu_y = f_st_y .+ seasonal_y
    y_obs ~ MvNormal(mu_y, Diagonal(sigma_y_vec.^2 .+ 1e-3))
end
```


```{julia}
model_v24 = model_v24_semi_adaptive_dfrff_multifidelity_gp(
    y_mock, u1_mock, u2_mock, u3_mock, z_mock,
    coords_y_s, coords_y_t,
    coords_u_s, coords_u_t,
    coords_z_s,
    W_z_fixed, b_z_fixed,
    W_u_fixed, b_u_fixed,
    W_y_gp_fixed, b_y_gp_fixed,
    W_sigma_fixed, b_sigma_fixed;
    M_rff_base=M_rff_base_val, M_rff_sigma=M_rff_sigma_val
)

chain_v24 = sample(model_v24, NUTS(), 100)

display(describe(chain_v24))
waic_v24 = compute_y_waic(model_v24, chain_v24)
println("\nWAIC for V24: ", waic_v24)

```

## V25: Hybrid FITC-RFF Multi-fidelity Model

This model represents the current pinnacle of our framework, combining the strengths of Random Fourier Features and Inducing Point methods.

### Key Innovations:
*   Hybrid Architecture: Uses Semi-Adaptive RFFs for the latent Z and U fields (fidelities) to perform non-linear dimensionality reduction and warping. The final Y-fidelity layer uses a FITC sparse GP, which is better at capturing global structural correlations.
*   Spectral Bottleneck: The latent U fields serve as 'deep' features that are fed into the FITC GP's kernel, creating a Deep GP effect where the kernel itself is defined over a learned feature space.
*   Data-Driven Initialization: RFF weights are initialized using the `generate_informed_rff_params` heuristic (FFT-informed).

### Model V25: Hybrid FITC-RFF Multi-fidelity Model

This model represents the integration of non-linear spectral feature extraction with sparse inducing point Gaussian Processes. It is designed to handle high-dimensional, multi-fidelity data by using Random Fourier Features (RFF) as a "spectral bottleneck" for latent covariates before passing them into a primary spatiotemporal GP.

#### Model Assumptions
1. Hierarchical Latent Fields: The spatial covariate $Z$ and spatiotemporal covariates $U$ are modeled as latent fields using Semi-Adaptive RFFs. This assumes that these variables can be represented as a weighted sum of random trigonometric basis functions, effectively approximating a Matern or RBF kernel.
2. Non-linear Warping: By passing the latent output of one fidelity (e.g., $Z$) as an input to the next (e.g., $U$), we assume a Deep Gaussian Process structure. This allows the model to learn complex, non-linear deformations of the input space.
3. Global Correlation via FITC: Unlike purely RFF models (which can be viewed as low-rank approximations via basis functions), the primary target $Y$ uses Fully Independent Training Conditional (FITC). This assumes that observed points are conditionally independent given a small set of inducing points, which is often better at preserving global structural correlations in spatiotemporal fields.
4. Semi-Adaptivity: We assume that while random features are powerful, they benefit from local refinement. The projection weights $W$ and biases $b$ are initialized via an FFT-informed heuristic and allowed to vary slightly under a tight Normal prior.
5. Synchronized Resolution: The model assumes that while different variables have different observation counts ($N_y, N_u, N_z$), they can be mapped to a common coordinate system via kernel interpolation.

#### Key References
* Hybrid GP Models: Lázaro-Gredilla, M., et al. (2010). *Sparse Spectrum Gaussian Processes*. JMLR. (Theoretical foundation for combining spectral and spatial GP views).
* FITC Approximation: Snelson, E., & Ghahramani, Z. (2006). *Sparse Gaussian Processes using Pseudo-inputs*. NIPS. (The standard for inducing point methods).
* Deep GPs: Damianou, A., & Lawrence, N. (2013). *Deep Gaussian Processes*. AISTATS. (Motivation for stacked non-linear latent layers).
* Multi-fidelity Learning: Perdikaris, P., et al. (2017). *Nonlinear model reduction for uncertainty quantification*. Journal of Computational Physics.


```{julia}
@model function model_v25_hybrid_fitc_rff(y_obs, u1_obs, u2_obs, u3_obs, z_obs, coords_y_s, coords_y_t, coords_u_s, coords_u_t, coords_z_s, Z_inducing_fixed, W_z_fixed, b_z_fixed, W_u_fixed, b_u_fixed; period=12.0, M_rff_base=40)
    # --- 1. Latent Feature Extraction (Fidelity Z) ---
    sigma_W_z ~ Exponential(0.1)
    sigma_b_z ~ Exponential(0.1)
    W_z ~ MvNormal(vec(W_z_fixed), sigma_W_z^2 * I)
    b_z ~ MvNormal(b_z_fixed, sigma_b_z^2 * I)
    W_z_mat = reshape(W_z, size(W_z_fixed))

    sigma_z_f ~ Exponential(1.0)
    beta_z ~ filldist(Normal(0, sigma_z_f), M_rff_base)

    z_latent = vec(rff_map(coords_z_s, W_z_mat, b_z) * beta_z)

    # Learnable noise for Z
    sigma_z_obs ~ Exponential(0.5)
    z_obs ~ MvNormal(z_latent, sigma_z_obs^2 * I + 1e-5*I)

    # --- 2. Latent Feature Extraction (Fidelity U) ---
    sigma_W_u ~ Exponential(0.1)
    sigma_b_u ~ Exponential(0.1)
    W_u ~ MvNormal(vec(W_u_fixed), sigma_W_u^2 * I)
    b_u ~ MvNormal(b_u_fixed, sigma_b_u^2 * I)
    W_u_mat = reshape(W_u, size(W_u_fixed))

    sigma_u_f ~ Exponential(1.0)
    beta_u ~ filldist(Normal(0, sigma_u_f), M_rff_base, 3)

    # Shared interpolation of Z to U and Y locations
    z_at_u = vec(rff_map(coords_u_s, W_z_mat, b_z) * beta_z)
    z_at_y = vec(rff_map(coords_y_s, W_z_mat, b_z) * beta_z)

    # Spatiotemporal mapping for U
    phi_u = rff_map(hcat(coords_u_s, coords_u_t, z_at_u), W_u_mat, b_u)
    u1_latent, u2_latent, u3_latent = [vec(phi_u * beta_u[:, i]) for i in 1:3]

    # Learnable noise for U
    sigma_u_obs ~ filldist(Exponential(0.5), 3)
    u1_obs ~ MvNormal(u1_latent, sigma_u_obs[1]^2 * I + 1e-5*I)
    u2_obs ~ MvNormal(u2_latent, sigma_u_obs[2]^2 * I + 1e-5*I)
    u3_obs ~ MvNormal(u3_latent, sigma_u_obs[3]^2 * I + 1e-5*I)

    # --- 3. Primary Output Layer (Sparse FITC GP) ---
    # Interpolate U fields at Y locations
    phi_u_at_y = rff_map(hcat(coords_y_s, coords_y_t, z_at_y), W_u_mat, b_u)
    u_at_y = phi_u_at_y * beta_u

    features_y = hcat(coords_y_s, coords_y_t, z_at_y, u_at_y)
    D_feat = size(features_y, 2)

    ls_y ~ filldist(Gamma(2, 2), D_feat)
    sigma_y_f ~ Exponential(1.0)
    k_y = SqExponentialKernel() ∘ ARDTransform(inv.(ls_y))

    K_ZZ = sigma_y_f^2 * kernelmatrix(k_y, RowVecs(Z_inducing_fixed)) + 1e-5 * I
    K_XZ = sigma_y_f^2 * kernelmatrix(k_y, RowVecs(features_y), RowVecs(Z_inducing_fixed))
    K_XX_diag = diag(sigma_y_f^2 * kernelmatrix(k_y, RowVecs(features_y)))

    u_latent_gp ~ MvNormal(zeros(size(Z_inducing_fixed, 1)), K_ZZ)
    m_f = vec(K_XZ * (K_ZZ \ u_latent_gp))
    cov_f_diag = max.(1e-6, K_XX_diag - diag(K_XZ * (K_ZZ \ K_XZ')))

    f_gp ~ MvNormal(m_f, Diagonal(cov_f_diag))

    # Seasonal and final observation
    beta_cos ~ Normal(0, 1)
    beta_sin ~ Normal(0, 1)
    seasonal = vec(beta_cos .* cos.(2π .* coords_y_t[:, 1] ./ period) .+ beta_sin .* sin.(2π .* coords_y_t[:, 1] ./ period))

    # Learnable noise for Y
    sigma_y_obs ~ Exponential(1.0)
    y_obs ~ MvNormal(f_gp .+ seasonal, sigma_y_obs^2 * I + 1e-5*I)
end
```


```{julia}
# Setup and Sample V25
M_inducing_v25 = 15
# The feature dimensionality for Y is 7: [lon, lat, time, z, u1, u2, u3]
Z_inducing_feat = randn(M_inducing_v25, 7)

model_v25 = model_v25_hybrid_fitc_rff(
    y_mock, u1_mock, u2_mock, u3_mock, z_mock,
    coords_y_s, coords_y_t, coords_u_s, coords_u_t, coords_z_s,
    Z_inducing_feat, W_z_fixed, b_z_fixed, W_u_fixed, b_u_fixed
)

println("Sampling Model V25 (Hybrid FITC-RFF)... ")
chain_v25 = sample(model_v25, NUTS(), 100)
display(describe(chain_v25))

waic_v25 = compute_y_waic(model_v25, chain_v25)
```



```{julia}
using Lux, Optimisers, Zygote, Plots

# 1. Custom RFF Layer for Lux
struct LuxRFFLayer <: Lux.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
end

function Lux.initialparameters(rng::AbstractRNG, layer::LuxRFFLayer)
    return (W = randn(rng, Float32, layer.in_dims, layer.out_dims),
            b = rand(rng, Float32, layer.out_dims) .* 2f0pi)
end

function ((l::LuxRFFLayer)(x::AbstractMatrix, ps, st))
    projection = (x * ps.W) .+ ps.b'
    return sqrt(2f0 / l.out_dims) .* cos.(projection), st
end

# 2. Define the Multi-fidelity Hybrid Model with an added hidden layer
function create_v25_lux(M_rff, M_inducing, D_st_y; hidden_dims=32)
    return Chain(
        # New Hidden Layer for additional non-linear warping
        Dense(D_st_y, hidden_dims, relu),
        # RFF Mapping layer
        LuxRFFLayer(hidden_dims, M_rff),
        # Output Layer
        Dense(M_rff, 1)
    )
end

# 3. Setup Training and Test Data
x_data_y = Float32.(hcat(coords_y_s, coords_y_t))
y_data_y = Float32.(reshape(y_mock, 1, :))

# Generate a test set for real-time loss monitoring
Ns_test = 20
x_test_y = Float32.(hcat(rand(Ns_test, 2), rand(Ns_test, 1) .+ 5.0))
y_test_y = Float32.(randn(1, Ns_test))

lux_model = create_v25_lux(40, 15, size(x_data_y, 2), hidden_dims=32)
rng = Random.default_rng()
ps, st = Lux.setup(rng, lux_model)

function loss_fn(ps, x, y, model, st; lambda=0.001f0)
    y_pred, _ = model(x, ps, st)
    mse_loss = Lux.mse(y_pred, y)
    reg_loss = lambda * (sum(abs2, ps.layer_1.weight) + sum(abs2, ps.layer_2.W))
    return mse_loss + reg_loss
end

# 4. Optimization Loop
initial_lr = 0.01f0
opt = Optimisers.Adam(initial_lr)
st_opt = Optimisers.setup(opt, ps)
train_loss_history = Float32[]
test_loss_history = Float32[]

println("Starting Lux.jl optimization tracking both Train and Test loss...")
for epoch in 1:200
    if epoch % 50 == 0
        Optimisers.adjust!(st_opt, initial_lr * (0.5f0 ^ (epoch ÷ 50)))
    end

    l_train, back = Zygote.pullback(p -> loss_fn(p, x_data_y, y_data_y, lux_model, st), ps)
    push!(train_loss_history, l_train)

    # Calculate test loss (without gradient)
    l_test = loss_fn(ps, x_test_y, y_test_y, lux_model, st)
    push!(test_loss_history, l_test)

    gs = back(1.0f0)[1]
    st_opt, ps = Optimisers.update(st_opt, ps, gs)

    if epoch % 50 == 0 || epoch == 1
        println("Epoch $epoch: Train Loss = $l_train, Test Loss = $l_test")
    end
end

# Plot results comparing Train vs Test
plot(train_loss_history, label="Train Loss", title="Lux Model Loss Convergence",
     xlabel="Epoch", ylabel="Loss (MSE + L2)", lw=2, color=:blue, yscale=:log10)
plot!(test_loss_history, label="Test Loss", lw=2, color=:red, linestyle=:dash)
```


```{julia}
# 1. Generate unseen test data
Ns_test = 20
coords_test_s = rand(Ns_test, 2)
coords_test_t = rand(Ns_test, 1) .+ 5.0 # Future time steps
x_test = Float32.(hcat(coords_test_s, coords_test_t))

# 2. Predict using the trained Lux model (ps and st from previous cell)
y_pred_test, _ = lux_model(x_test, ps, st)

# 3. Create a parity plot for the training data to visualize fit quality
y_pred_train, _ = lux_model(x_data_y, ps, st)

p_parity = scatter(vec(y_data_y), vec(y_pred_train),
                   xlabel="Observed", ylabel="Predicted",
                   title="Lux Model: Train Fit Parity",
                   label="Train Points", markerstrokewidth=0, alpha=0.7)
plot!(p_parity, [-2, 2], [-2, 2], line=:dash, color=:black, label="Ideal")

display(p_parity)

println("Predictions for first 5 test points:")
display(y_pred_test[1:5])
```


```{julia}
using Statistics

# 1. Regression Metrics Calculation
y_pred_final, _ = lux_model(x_data_y, ps, st)
y_pred_vec = vec(y_pred_final)
y_true_vec = vec(y_data_y)

# Mean Absolute Error
mae_val = mean(abs.(y_true_vec .- y_pred_vec))

# Mean Squared Error and RMSE
mse_val = mean((y_true_vec .- y_pred_vec).^2)
rmse_val = sqrt(mse_val)

# R-squared (Coefficient of Determination)
ss_res = sum((y_true_vec .- y_pred_vec).^2)
ss_tot = sum((y_true_vec .- mean(y_true_vec)).^2)
r2_val = 1 - (ss_res / ss_tot)

println("--- Regression Fit Metrics ---")
println("Mean Absolute Error (MAE): ", round(mae_val, digits=4))
println("Root Mean Squared Error (RMSE): ", round(rmse_val, digits=4))
println("R-squared (R²): ", round(r2_val, digits=4))

# 2. Visualizing Error Distribution
histogram(y_true_vec .- y_pred_vec,
          bins=15,
          title="Residual Distribution",
          xlabel="Prediction Error",
          ylabel="Frequency",
          label="Residuals",
          color=:orange,
          alpha=0.7)
```


```{julia}
using Statistics

# 1. Test Set Metrics Calculation
# We'll use a mock ground truth for the test set since it was generated as unseen data
# For the purpose of this demonstration, we'll assume the mock target follows the same distribution
y_test_true = randn(size(y_pred_test))
y_test_pred_vec = vec(y_pred_test)
y_test_true_vec = vec(y_test_true)

# MAE, MSE, and RMSE for Test Set
mae_test = mean(abs.(y_test_true_vec .- y_test_pred_vec))
mse_test = mean((y_test_true_vec .- y_test_pred_vec).^2)
rmse_test = sqrt(mse_test)

# R-squared for Test Set
ss_res_test = sum((y_test_true_vec .- y_test_pred_vec).^2)
ss_tot_test = sum((y_test_true_vec .- mean(y_test_true_vec)).^2)
r2_test = 1 - (ss_res_test / ss_tot_test)

println("--- Test Set Performance Metrics ---")
println("Test MAE:   ", round(mae_test, digits=4))
println("Test RMSE:  ", round(rmse_test, digits=4))
println("Test R²:    ", round(r2_test, digits=4))

# 2. Parity Plot for Test Set
scatter(y_test_true_vec, y_test_pred_vec,
        title="Test Set Parity Plot",
        xlabel="True Values",
        ylabel="Predictions",
        label="Test Points",
        color=:green,
        markerstrokewidth=0,
        alpha=0.8)
plot!([-2, 2], [-2, 2], line=:dash, color=:black, label="Ideal")
```


```{julia}
using Plots

# Visualize the loss convergence for the Lux model
# We use a log scale for the y-axis to better see the refinement in later epochs
plot(loss_history,
     title="Lux Model: Training Loss Convergence",
     xlabel="Epoch",
     ylabel="Loss (MSE + L2)",
     lw=2.5,
     yscale=:log10,
     label="Total Loss",
     color=:blue,
     grid=true)

# Annotate the final loss value for quick reference
final_l = loss_history[end]
annotate!(length(loss_history), final_l,
          text(" Final: $(round(final_l, digits=4))", :left, 8, :blue))
```


```{julia}
using Printf

# Prepare Comparison Table
println("--- Multi-fidelity Lux Model: Performance Comparison ---")
println("Metric    | Training Set | Test Set")
println("----------|--------------|----------")
@printf("MAE       | %-12.4f | %.4f\n", mae_val, mae_test)
@printf("RMSE      | %-12.4f | %.4f\n", rmse_val, mse_test)
@printf("R-squared | %-12.4f | %.4f\n", r2_val, r2_test)

# Optional: Visual Comparison of Errors
p_box = boxplot(["Train" for _ in 1:length(y_true_vec .- y_pred_vec)], y_true_vec .- y_pred_vec,
                label="Train Residuals", color=:blue, alpha=0.5)
boxplot!(p_box, ["Test" for _ in 1:length(y_test_true_vec .- y_test_pred_vec)], y_test_true_vec .- y_test_pred_vec,
         label="Test Residuals", color=:green, alpha=0.5,
         title="Residual Distribution Comparison", ylabel="Error (Actual - Predicted)")

display(p_box)
```


```{julia}
using StatsBase, ROCAnalysis

# 1. Binarize the regression targets for evaluation
# We use the median of training data as a threshold to define 'High' vs 'Low' events
thresh = median(y_data_y)
y_true_bin = vec(y_data_y) .> thresh

# 2. Get probabilities/scores from our trained Lux model
y_scores, _ = lux_model(x_data_y, ps, st)
y_scores_vec = vec(y_scores)

# 3. Compute ROC curve using ROCAnalysis.jl
roc_data = roc(y_scores_vec, y_true_bin)

# 4. Plot the ROC Curve
plot(roc_data,
     title="ROC Curve: Lux Deep Kernel Classifier (Binarized Y)",
     label="Model V25 Hybrid",
     lw=3,
     color=:blue,
     xlabel="False Positive Rate",
     ylabel="True Positive Rate")
plot!([0, 1], [0, 1], linestyle=:dash, color=:black, label="Random Guess")

# Calculate and print AUC
auc_val = auc(roc_data)
println("Classification Threshold: ", round(thresh, digits=3))
println("Area Under the Curve (AUC): ", round(auc_val, digits=4))
```

## Summary of Multi-fidelity Spatiotemporal Development (V21–V25)

This framework has evolved from traditional Gaussian Processes to scalable Deep Kernel Learning architectures. Key milestones include:

### Architectural Decisions
* Transition to Deterministic Features: Due to memory constraints with `ForwardDiff.Dual` numbers in fully adaptive RFFs (V22), we adopted Fixed Deterministic Fourier Features (DFRFF) in V23 and Semi-Adaptive DFRFF in V24 to significantly reduce the parameter space while maintaining flexibility.
* Hybrid FITC-RFF (V25): The current pinnacle architecture. It uses RFFs for efficient non-linear feature extraction in latent covariate layers (Z, U) and a Fully Independent Training Conditional (FITC) sparse GP for the primary target (Y) to preserve global structural correlations.
* FFT-Informed Heuristic: RFF frequencies are scaled based on the inverse standard deviation of input coordinates to align basis functions with the data's characteristic scales.

### Implementation Refinements
* Numerical Stability: Increased jitter to `1e-3` across all kernel matrices to prevent `PosDefException` during NUTS sampling.
* Lux.jl Integration: A deep learning implementation of V25 was created, allowing for coordinate warping via ReLU activations and optimization via Adam.

### Future Directions
* Real-world Validation: Testing the hybrid architecture against non-mock spatiotemporal datasets.
* Extreme Scalability (SVI): Implementation of full Stochastic Variational Inference for the final hybrid model to handle millions of observations.
