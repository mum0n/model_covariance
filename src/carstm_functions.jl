macro save_carstm_state(file_to_save_name_sym)
  quote
    try
      # Evaluate the input symbol (e.g., :state_filename) to its value (e.g., "carstm_state.jld2")
      local filename_val = $(esc(file_to_save_name_sym))
      @info "Saving CARSTM state to $(filename_val)..."
      # JLD2.@save expects variable names as symbols, not their values.
      # The variables themselves should be directly passed.
      JLD2.@save "$(filename_val)" areal_units mod chain pts y_sim y_binary time_idx weights trials cov_indices cov_indices_mat trials_sim class1_sim class2_sim weights_sim adj_matrix_numeric n_pts n_time area_method
      @info "CARSTM state saved successfully."
    catch e
      @error "Error saving CARSTM state: $e"
    end
  end
end

macro load_carstm_state(filename_sym)
  quote
    # Evaluate the input symbol (e.g., :state_filename) to its value (e.g., "carstm_state.jld2")
    local filename_val = $(esc(filename_sym))
    if !isfile(filename_val)
      @error "File $(filename_val) not found."
      return nothing
    end
    try
      @info "Loading CARSTM state from $(filename_val)..."
      # JLD2.@load expects variable names as symbols, not their values.
      # The variables themselves should be directly passed.
      JLD2.@load "$(filename_val)" areal_units mod chain pts y_sim y_binary time_idx weights trials cov_indices cov_indices_mat trials_sim class1_sim class2_sim weights_sim adj_matrix_numeric n_pts n_time area_method
      @info "CARSTM state loaded successfully."
      # Variables are loaded directly into the calling scope by JLD2.@load
      # No explicit return value from the macro itself, as it injects variables
    catch e
      @error "Error loading CARSTM state: $e"
      return nothing
    end
  end
end


function init_params_extract( res=NaN; load_from_file=false, override_means=false, fn_inits = "init_params.jl2"  )
  # Description: Extracts initial parameter values from a model result summary or loads them from a file.
  # Inputs:
  #   - res: Model result object (default: NaN).
  #   - load_from_file: Boolean, if true loads params from fn_inits.
  #   - override_means: Boolean, if true applies custom overrides for specific parameter patterns.
  #   - fn_inits: String, filename for storage.
  # Outputs:
  #   - A FillArray containing the extracted or loaded mean parameter values.

  if load_from_file
    init_params = load(fn_inits )
    return(init_params)
  end

  ressumm = summarize(res)
  vns = ressumm.nt.parameters
  means = ressumm.nt[2]  # means

  if  override_means
    u = findall(x-> occursin(r"^t_period\[", String(x)), vns ); vns[u]
    if length(u) > 0 
      means[u] = [ 1.0, 1.0, 5.0, 5.0]  # (sin, cos) X annual, 5-year (el nino)
    end

    u = findall(x-> occursin(r"^pca_sd\[", String(x)), vns ); vns[u]
    if length(u) > 0  
      means[u] = sigma_prior  # from basic pca
    end

    u = findall(x-> occursin(r"^v\[", String(x)), vns ); vns[u]
    if length(u) > 0 
      means[u] = v_prior  # from basic pca
    end
  end

  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function init_params_copy( res=NaN, res0=NaN; load_from_file=false, override_means=false, fn_inits = "init_params.jl2"  )
  # using spatial parts of res0 
  # Description: Copies parameter values from a reference result (res0) to a target result structure (res).
  # Inputs:
  #   - res: Target model result object.
  #   - res0: Reference model result object.
  #   - load_from_file: Boolean to load from fn_inits instead.
  #   - override_means: Boolean to apply custom pattern-based overrides.
  #   - fn_inits: String, filename for storage.
  # Outputs:
  #   - A FillArray containing the merged mean parameter values.
  if load_from_file
    init_params = load(fn_inits )
    return(init_params)
  end

  ressumm = summarize(res)
  vns = ressumm.nt.parameters
  means = ressumm.nt[2]  # means

  ressumm0 = summarize(res0)
  vns0 = ressumm0.nt.parameters
  means0 = ressumm0.nt[2]  # means

  if  override_means
    u = findall(x-> occursin(r"^t_period\[", String(x)), vns );  
    if length(u) > 0 
      means[u] = [ 1.0, 1.0, 5.0, 5.0, 10.0, 10.0][1:length(u)]  # (sin, cos) X annual, 5-year (el nino)
    end

    u = findall(x-> occursin(r"^pca_sd\[", String(x)), vns );  
    if length(u) > 0  
      u0 = findall(x-> occursin(r"^pca_sd\[", String(x)), vns0 );  
      if length(u0) > 0  && length(u) == length(u0)
        means[u] = means0[u0]  # from basic pca
      end
    end

    u = findall(x-> occursin(r"^v\[", String(x)), vns );  
    if length(u) > 0  
      u0 = findall(x-> occursin(r"^pca_sd\[", String(x)), vns0 );  
      if length(u0) > 0  && length(u) == length(u0)
        means[u] = means0[u0]  # from basic pca
      end
    end
  end
  
  init_params = FillArrays.Fill( means )
  jldsave( fn_inits; init_params )

  return(init_params)
end


function plot_kde_simple(pts; grid_res=600, sd_extension_factor=0.25, title="Spatial Intensity (KDE)")
    # Internal wrapper for estimate_local_kde_with_extrapolation
    # Description: Generates a simple 2D Heatmap of spatial intensity using Kernel Density Estimation.
    # Inputs:
    #   - pts: Vector of (x, y) coordinate tuples.
    #   - grid_res: Resolution of the output grid.
    #   - sd_extension_factor: Factor to extend the bandwidth standard deviation.
    #   - title: Title for the generated plot.
    # Outputs:
    #   - A Plots.Plot object (Heatmap with scatter overlay).
    # Using a dummy time_idx of 1s since we are plotting a static slice
    t_idx_dummy = ones(Int, length(pts))
    x_g, y_g, intensity = estimate_local_kde_with_extrapolation(pts, t_idx_dummy, 1; grid_res=grid_res, sd_extension_factor=sd_extension_factor)
    
    plt = Plots.heatmap(x_g, y_g, intensity', 
                  title=title, 
                  c=:viridis, 
                  aspect_ratio=:equal,
                  xlabel="X", ylabel="Y")
    Plots.scatter!(plt, [p[1] for p in pts], [p[2] for p in pts], 
                   markersize=2, markercolor=:white, markeralpha=0.5, label="Points")
    return plt
end


Turing.@model function pca_carstm( Y, ::Type{T}=Float64 ) where {T}
  # X, G, log_offset, y, z, auid, nData, nX, nG, nAU, node1, node2, scaling_factor 
  # Description: Turing model performing Latent PCA (Householder) followed by a spatial CARSTM (BYM2) on factor scores.
  # Inputs:
  #   - Y: Observation matrix (nData x nVar).
  #   - Implicit globals: nAU, nvar, nz, node1, node2, scaling_factor, sigma_prior.
  # Outputs:
  #   - Bayesian posterior estimates for PCA loadings and spatial components.
  # first pca (latent householder transform) then carstm bym2
  
  # pca_sd ~ Bijectors.ordered( MvLogNormal(MvNormal(ones(nz) )) )  
  pca_sd ~ Bijectors.ordered( arraydist( LogNormal.(sigma_prior, 1.0)) )  
  # minimum(pca_sd) < noise && Turing.@addlogprob! Inf  
  # maximum(pca_sd) > nvar && Turing.@addlogprob! Inf 
  pca_pdef_sd ~ LogNormal(0.0, 0.5)
  v ~ filldist(Normal(0.0, 1.0), nvh )
  Kmat, r, U = householder_transform(v, nvar, nz, ltri, pca_sd, pca_pdef_sd, noise)
  # soft priors for r 
  # new .. Gamma in stan is same as in Distributions
  r ~ filldist(Gamma(0.5, 0.5), nz)
  Turing.@addlogprob! sum(-log.(r) .* iz)
  Turing.@addlogprob! -0.5 * sum(pca_sd.^ 2) + (nvar-nz-1) * sum(log.(pca_sd)) 
  Turing.@addlogprob! sum(log.(pca_sd[hindex[:,1]].^ 2) .- pca_sd[hindex[:,2]].^ 2)
  Turing.@addlogprob! sum(log.(2.0 .* pca_sd))
   
  Y ~ filldist( MvNormal( Symmetric(Kmat)), nData )  # latent factors
  
  pcscores = Y' * U 

  # Fixed (covariate) effects (including intercept)
  f_beta ~ filldist( Normal(0.0, 1.0), nz) ;
  
  # icar (spatial effects)
  s_theta ~ filldist( Normal(0.0, 1.0), nAU, nz)  # unstructured (heterogeneous effect)
  s_phi ~ filldist( Normal(0.0, 1.0), nAU, nz) # spatial effects: stan goes from -Inf to Inf .. 
    
  s_sigma ~ filldist( LogNormal(0.0, 1.0), nz) ; 
  s_rho ~ filldist(Beta(0.5, 0.5), nz);
        
  eta ~ filldist( LogNormal(0.0, 1.0), nz ) # overall observation variance
 
  # spatial effects (without inverting covariance) 
  dphi = phi[node1] - phi[node2]
  dot_phi = dot( dphi, dphi )
  Turing.@addlogprob! -0.5 * dot_phi

  # soft sum-to-zero constraint on phi
  sum_phi_s = sum( dot_phi ) 
  sum_phi_s ~ Normal(0, 0.001 * nAU);      # soft sum-to-zero constraint on s_phi)

  convolved_re_s = icar_form( s_theta[:,z], s_phi[:,z], s_sigma[z], s_rho[z] )
  Turing.@addlogprob! -0.5 * dot_phi
  sum_phi_s ~ Normal(0, 0.001 * nAU_float);      # soft sum-to-zero constraint on s_phi)
  mu = f_beta[z] .+ convolved_re_s[auid] 
  pcscores[:,z] ~ MvNormal( mu, eta[z] *I )
   
  return
end
 


Turing.@model function carstm_pca( Y, ::Type{T}=Float64; nData=size(Y, 1), nvar=size(Y, 2), nz=2, nvh=Int(nvar*nz - nz * (nz-1) / 2), noise=1e-9, log_offset=0.0, hindex=(2,1) ) where {T}

    # Description: Turing model that estimates spatial/temporal effects per variable before a latent PCA reduction.
    # Inputs:
    #   - Y: Data matrix.
    #   - nz: Number of latent factors.
    #   - Implicit globals: nAU, nX, node1, node2, scaling_factor.
    # Outputs:
    #   - Bayesian posterior estimates for individual trends and latent PCA structure.
    # first carstm then pca .. as in msmi . incomplete ... too slow

    # Threads.@threads for f in 1:nvar
    for f in 1:nvar
        # est betas, sp, st eff for each sp

        # Fixed (covariate) effects 
        f_beta ~ filldist( Normal(0.0, 1.0), nX);
        f_effect = X * f_beta .+ log_offset

        # icar (spatial effects)
        beta_s ~ filldist( Normal(0.0, 1.0), nX); 
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
        mp_icar =  mp_pca * beta_s +  convolved_re_s[auid]  # mean process for bym2 / icar
        #  @. y ~ LogPoisson( mp_icar);


        # Fourier process (global, main effect)
        ncf = 4  # 2 for seasonal 2 for interannual ..
        t_period ~ filldist( LogNormal(0.0, 0.5), ncf ) 
        t_beta ~ Normal(0, 1)  # linear trend in time

        t_amp ~ MvNormal(Zeros(ncf), I) #  coefficients of harmonic components
 # ~ MvNormal(Zeros(ncf), I) #  coefficients of harmonic components
     #   t_error ~ LogNormal(0, 1)
    #end
    #    mp_fp = rand( MvNormal( mu_fp, t_error^2 * I ) )  
# = t_beta .* ti + sin.( (2pi / t_period) .* ti ) * betahs + cos.((2pi / t_period) .* ti ) * t_amp


        # space X time

    end

    # latent PCA with householder transform  upon the L (latent Y)   
    pca_pdef_sd ~ LogNormal(0.0, 0.5)

    # currently, Bijectors.ordered is broken, revert for better posteriors once it works again
    # sigma ~ Bijectors.ordered( MvLogNormal(MvNormal(ones(nz) )) )  
    sigma ~ filldist(LogNormal(0.0, 1.0), nz ) 
    v ~ filldist(Normal(0.0, 1.0), nvh )

    v_mat = zeros(T, nvar, nz)
    v_mat[ltri] .= v

    U = householder_to_eigenvector( v_mat, nvar, nz )
    
    W = zeros(T, nvar, nz)
    W += U * Diagonal(sigma)
    Kmat = W * W' + (pca_pdef_sd^2 + noise) * I(nvar)

   # soft priors for r 
    # favour reasonably small r .. new .. Gamma in stan is same as in Distributions
    r = sqrt.(mapslices(norm, v_mat[:,1:nz]; dims=1))
    r ~ filldist(Gamma(2.0, 2.0), nz)
    Turing.@addlogprob! sum(-log.(r) .* iz)

    minimum(sigma) < noise && Turing.@addlogprob! Inf && return
    
    Turing.@addlogprob! -0.5 * sum(sigma.^ 2) + (nvar-nz-1) * sum(log.(sigma)) 
    Turing.@addlogprob! sum(log.(sigma[hindex[:,1]].^ 2) .- sigma[hindex[:,2]].^ 2)
    Turing.@addlogprob! sum(log.(2.0 .* sigma))

    mp_pca = rand( (MvNormal( Symmetric(Kmat)), nData ) )

    # Y ~ filldist(MvNormal( Symmetric(Kmat)), nData )

    # y ~ ....

    return 
end
 


Turing.@model function carstm_temperature( Y, ::Type{T}=Float64; 
  nData=size(Y, 1), nvar=size(Y, 2), nz=2, nvh=Int(nvar*nz - nz * (nz-1) / 2), noise=1e-9 ) where {T}
  
    # Description: Turing model for temperature mapping using ICAR (spatial) and Fourier (temporal) processes.
    # Inputs:
    #   - Y: Temperature observation vector.
    #   - Implicit globals: X, nX, nAU, node1, node2, scaling_factor, ti, ncf, vcv.
    # Outputs:
    #   - Posterior distribution of spatial trends and periodic temperature fluctuations.

    # Fixed (covariate) effects 
    #f_beta ~ filldist( Normal(0.0, 1.0), nX);
    #f_effect = X * f_beta + log_offset

    # icar (spatial effects)
    beta_s ~ filldist( Normal(0.0, 1.0), nX); 
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


    return 
end
 

function libgeos_lattice_adjacency_matrix(rows::Int, cols::Int)
    """
    libgeos_lattice_adjacency_matrix(rows, cols)

    Description:
    Generates a sparse adjacency matrix for a regular 2D lattice using LibGEOS for spatial geometry operations.
    Constructs unit square polygons for each cell and identifies neighbors based on Queen contiguity
    (any shared boundary point or edge).

    Inputs:
    - rows (Int): Number of rows in the lattice grid.
    - cols (Int): Number of columns in the lattice grid.

    Output:
    - W (SparseMatrixCSC{Int, Int}): A binary sparse adjacency matrix of size (rows*cols) x (rows*cols).
    """
    # Create polygons for each cell in the lattice
    polygons = []
    for r in 1:rows, c in 1:cols
        # Define unit square coordinates as nested vectors for LibGEOS compatibility
        coords = [
            [Float64(c-1), Float64(r-1)],
            [Float64(c),   Float64(r-1)],
            [Float64(c),   Float64(r)],
            [Float64(c-1), Float64(r)],
            [Float64(c-1), Float64(r-1)]
        ]
        # Construct LinearRing and then Polygon
        ring = LibGEOS.LinearRing(coords)
        push!(polygons, LibGEOS.Polygon(ring))
    end

    n = length(polygons)
    W = spzeros(Int, n, n)

    # Queen contiguity check
    for i in 1:n
        poly_i = polygons[i]
        for j in (i+1):n
            if LibGEOS.intersects(poly_i, polygons[j])
                W[i, j] = W[j, i] = 1
            end
        end
    end
    return W
end


function summarize_array(samples::AbstractArray; alpha=0.05)
    # Description: Summarizes a sample array across the last dimension (samples).
    # Inputs:
    #   - samples: AbstractArray where the last dimension is the index of MCMC samples.
    #   - alpha: Significance level for credible intervals.
    # Outputs:
    #   - NamedTuple: (mean, median, lower, upper) with sample dimension dropped.

    # Assumes last dimension is samples
    dims = size(samples)
    n_dims = length(dims)
    
    m = mean(samples, dims=n_dims)
    med = median(samples, dims=n_dims)
    l = mapslices(x -> quantile(x, alpha/2), samples, dims=n_dims)
    u = mapslices(x -> quantile(x, 1 - alpha/2), samples, dims=n_dims)
    
    # Squeeze the sample dimension for easier use
    return (mean = dropdims(m, dims=n_dims), 
            median = dropdims(med, dims=n_dims), 
            lower = dropdims(l, dims=n_dims), 
            upper = dropdims(u, dims=n_dims))
end

# --- 1. Custom PC Priors ---
struct PCPriorSigma <: ContinuousUnivariateDistribution
    U::Float64
    alpha::Float64
    lambda::Float64
    function PCPriorSigma(U, alpha)
        return new(U, alpha, -log(alpha) / U)
    end
end

function Distributions.logpdf(d::PCPriorSigma, x::Real)
    x > 0 ? log(d.lambda) - d.lambda * x : -Inf
end

Distributions.rand(rng::AbstractRNG, d::PCPriorSigma) = rand(rng, Exponential(1 / d.lambda))
Distributions.minimum(d::PCPriorSigma) = 0.0
Distributions.maximum(d::PCPriorSigma) = Inf
Bijectors.bijector(d::PCPriorSigma) = Bijectors.exp

# --- 2. Precision Matrix Construction ---
function build_laplacian_precision(adj_matrix)
    # Description: Builds a standard Laplacian precision matrix (Degree - Adjacency).
    # Inputs:
    #   - adj_matrix: Sparse adjacency matrix.
    # Outputs:
    #   - Sparse precision matrix.

    D = Diagonal(vec(sum(adj_matrix, dims=2)))
    return D - adj_matrix
end

function scale_precision!(Q)
    vals = eigvals(Matrix(Q))
    scaling_factor = geomean(vals[vals .> 1e-6])
    if Q isa Symmetric; Q.data ./= scaling_factor; else; Q ./= scaling_factor; end
    return Q
end

function build_rw2_precision(n)
    # Description: Builds a second-order random walk (RW2) precision matrix for smoothing.
    # Inputs:
    #   - n: Number of categories or time points.
    # Outputs:
    #   - Sparse precision matrix of size n x n.
    D = spzeros(n - 2, n)
    for i in 1:(n - 2)
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    end
    return D' * D
end

function build_ar1_precision(n, rho, tau)
    # Description: Builds a first-order autoregressive (AR1) precision matrix.
    # Inputs:
    #   - n: Number of time points.
    #   - rho: Correlation coefficient.
    #   - tau: Precision scale.
    # Outputs:
    #   - Sparse precision matrix.
    T = promote_type(typeof(rho), typeof(tau))
    diag_vals = [one(T); fill(one(T) + rho^2, n - 2); one(T)]
    off_diag = fill(-rho, n - 1)
    Q = spdiagm(0 => diag_vals, 1 => off_diag, -1 => off_diag)
    return (tau / (one(T) - rho^2)) * Q
end

function build_cyclic_ar1_precision(n, rho, tau)
    # Description: Builds a cyclic AR1 precision matrix (wrapping last to first).
    # Inputs:
    #   - n: Number of time points.
    #   - rho: Correlation coefficient.
    #   - tau: Precision scale.
    # Outputs:
    #   - Sparse precision matrix.
    T = promote_type(typeof(rho), typeof(tau))
    Q = zeros(T, n, n)
    for i in 1:n
        Q[i, i] = one(T) + rho^2
        prev, nxt = (i == 1 ? n : i - 1), (i == n ? 1 : i + 1)
        Q[i, prev] = -rho
        Q[i, nxt] = -rho
    end
    return (tau / (one(T) - rho^2)) * Q
end

function logpdf_gmrf(x, Q)
    # Description: Calculates the log-probability of a Gaussian Markov Random Field.
    # Inputs:
    #   - x: Vector of values.
    #   - Q: Precision matrix.
    # Outputs:
    #   - Log-likelihood value.
    Q_stable = Matrix(Q) + I * 1e-5
    F = cholesky(Symmetric(Q_stable))
    return 0.5 * (logdet(F) - dot(x, Q, x) - length(x) * log(2 * pi))
end



function reconstruct_posteriors(model::DynamicPPL.Model, chain::MCMCChains.Chains, pts, area_idx, time_idx, W_sym; cov_indices=nothing, class1=nothing, class2=nothing)
    # Description: Aggregates and summarizes MCMC samples into denoised spatial, temporal, and prediction matrices.
    # Inputs:
    #   - model: Turing model object.
    #   - chain: MCMC sample chain.
    #   - pts: Observation points.
    #   - area_idx/time_idx: Mapping of observations to spatial/temporal units.
    #   - W_sym: Adjacency matrix.
    #   - cov_indices/class1/class2: Categorical indices for effects.
    # Outputs:
    #   - NamedTuple containing summarized spatial/temporal effects, categorical effects, 
    #     ST matrices (denoised/noisy), and predictions.

    N_obs = size(pts,1); N_areas = size(W_sym, 1); N_time = maximum(time_idx)
    family = detect_model_family(model)
    N_samples = size(chain, 1)
    names_ch = names(chain)

    # 1. Spatial Effects
    spatial_samples = zeros(N_areas, N_samples)
    if "s_eff[1]" in names_ch
        for i in 1:N_areas; spatial_samples[i, :] = vec(chain["s_eff[$i]"].data); end
    elseif all(x -> x in names_ch, ["sigma_sp", "phi_sp"])
        sig = vec(chain[:sigma_sp].data); phi = vec(chain[:phi_sp].data)
        u_icar = hcat([vec(chain["u_icar[$i]"].data) for i in 1:N_areas]...)
        u_iid = hcat([vec(chain["u_iid[$i]"].data) for i in 1:N_areas]...)
        spatial_samples = (sig .* (sqrt.(phi) .* u_icar .+ sqrt.(1 .- phi) .* u_iid))'
    end

    # 2. Temporal Effects (GMRF or RFF)
    temporal_samples = zeros(N_time, N_samples)
    if "f_time[1]" in names_ch
        for i in 1:N_time; temporal_samples[i, :] = vec(chain["f_time[$i]"].data); end
    else
        for i in 1:N_time
            val = zeros(N_samples)
            if "f_trend[$i]" in names_ch; val .+= vec(chain["f_trend[$i]"].data); end
            if "f_seas[$i]" in names_ch; val .+= vec(chain["f_seas[$i]"].data); end
            temporal_samples[i, :] = val
        end
    end

    # 3. Categorical/Class Effects
    beta_cov_summaries = []
    beta_cov_raw = []
    for k in 1:4
        k_names = filter(n -> startswith(string(n), "beta_cov[$k]["), names_ch)
        if !isempty(k_names)
            mat = hcat([vec(chain[n].data) for n in k_names]...) # N_samples x N_levels
            push!(beta_cov_raw, mat)
            # Summarize (Levels x 1 x Samples)
            push!(beta_cov_summaries, summarize_array(reshape(mat', size(mat,2), 1, N_samples)))
        end
    end

    b1_summary = nothing
    b1_raw = nothing
    if "b_class1[1]" in names_ch
        b1_raw = hcat([vec(chain["b_class1[$i]"].data) for i in 1:maximum(class1)]...)
        b1_summary = summarize_array(reshape(b1_raw', size(b1_raw,2), 1, N_samples))
    end

    b2_summary = nothing
    b2_raw = nothing
    if "b_class2[1]" in names_ch
        b2_raw = hcat([vec(chain["b_class2[$i]"].data) for i in 1:maximum(class2)]...)
        b2_summary = summarize_array(reshape(b2_raw', size(b2_raw,2), 1, N_samples))
    end

    # 4. Parameters for noise
    sig_y = "sigma_y" in names_ch ? vec(chain[:sigma_y].data) : zeros(N_samples)
    r_nb = "r_nb" in names_ch ? vec(chain[:r_nb].data) : fill(2.0, N_samples)
    phi_zi = "phi_zi" in names_ch ? vec(chain[:phi_zi].data) : zeros(N_samples)

    # 5. Core Reconstruction
    st_samples = zeros(N_areas, N_time, N_samples)
    st_noisy_samples = zeros(N_areas, N_time, N_samples)
    pred_samples = zeros(N_obs, N_samples)

    for s in 1:N_samples
        for a in 1:N_areas, t in 1:N_time
            eta = spatial_samples[a, s] + temporal_samples[t, s]
            st_samples[a, t, s] = eta
            if family == :gaussian; st_noisy_samples[a, t, s] = eta + randn() * sig_y[s]
            elseif family == :lognormal; st_noisy_samples[a, t, s] = rand(LogNormal(eta, sig_y[s]))
            elseif family == :binomial; st_noisy_samples[a, t, s] = rand(Binomial(1, logistic(eta)))
            elseif family == :poisson; st_noisy_samples[a, t, s] = rand(Poisson(exp(eta)))
            elseif family == :negbinomial; st_noisy_samples[a, t, s] = rand(NegativeBinomial2(exp(eta), r_nb[s]))
            elseif family == :zip; st_noisy_samples[a, t, s] = rand() < phi_zi[s] ? 0.0 : rand(Poisson(exp(eta)))
            elseif family == :zinb; st_noisy_samples[a, t, s] = rand() < phi_zi[s] ? 0.0 : rand(NegativeBinomial2(exp(eta), r_nb[s]))
            else st_noisy_samples[a, t, s] = eta
            end
        end

        for i in 1:N_obs
            eta_i = st_samples[area_idx[i], time_idx[i], s]
            if !isempty(beta_cov_raw)
                for k in 1:length(beta_cov_raw)
                    eta_i += beta_cov_raw[k][s, cov_indices[i, k]]
                end
            end
            if !isnothing(b1_raw); eta_i += b1_raw[s, class1[i]]; end
            if !isnothing(b2_raw); eta_i += b2_raw[s, class2[i]]; end
            
            pred_samples[i, s] = family == :binomial ? logistic(eta_i) : (family in [:negbinomial, :zinb, :zip, :poisson, :lognormal] ? exp(eta_i) : eta_i)
        end
    end

    return (spatial = summarize_array(reshape(spatial_samples, N_areas, 1, N_samples)),
            temporal = summarize_array(reshape(temporal_samples, N_time, 1, N_samples)),
            beta_cov = beta_cov_summaries,
            b_class1 = b1_summary,
            b_class2 = b2_summary,
            st_mat_denoised = summarize_array(st_samples),
            st_mat_noisy = summarize_array(st_noisy_samples),
            predictions = summarize_array(reshape(pred_samples, N_obs, 1, N_samples)),
            family = family)
end



function detect_model_family(model::DynamicPPL.Model)
    # Description: Infers the likelihood family (e.g. :poisson) from the Turing model function name.
    # Inputs:
    #   - model: Turing model object.
    #   - Outputs:
    #   - Symbol indicating the family.

    name = lowercase(string(model.f))
    if occursin("gaussian", name) return :gaussian end
    if occursin("poisson", name) return :poisson end
    if occursin("binomial", name) return :binomial end
    if occursin("negativebinomial", name) return :negbinomial end
    if occursin("lognormal", name) return :lognormal end
    return :gaussian # Fallback
end


function posterior_predictive_check(model::DynamicPPL.Model, stats, y_obs)
    # Description: Performs Posterior Predictive Checks (PPC), computing metrics like RMSE, Pearson R, and Kendall Tau.
    # Inputs:
    #   - model: Turing model object.
    #   - stats: summarized results from reconstruct_posteriors.
    #   - y_obs: Vector of ground-truth observations.
    # Outputs:
    #   - NamedTuple of metrics and Plots.Plot objects for density and scatter checks.

    # 1. Automatically detect family from model if not explicitly in stats
    family = haskey(stats, :family) ? stats.family : detect_model_family(model)
    # Ensure y_pred is a 1D vector for calculation compatibility
    y_pred = vec(stats.predictions.mean)

    # 2. Basic Metrics
    rmse = sqrt(mean((y_obs .- y_pred).^2))

    # Correlation metrics with p-values from HypothesisTests
    pearson_val = 0.0; pearson_p = 1.0
    kendall_val = 0.0; kendall_p = 1.0

    if length(unique(y_pred)) > 1 && length(unique(y_obs)) > 1
        try
            # Use explicit module referencing for reliability
            p_test = HypothesisTests.CorrelationTest(y_obs, y_pred)
            pearson_val = p_test.r
            pearson_p = HypothesisTests.pvalue(p_test)

            # Replaced Spearman with Kendall's Tau
            k_test = HypothesisTests.KendallTauTest(y_obs, y_pred)
            kendall_val = k_test.tau
            kendall_p = HypothesisTests.pvalue(k_test)
        catch e
            @warn "Hypothesis tests failed. Ensure HypothesisTests is correctly loaded."
        end
    end

    println("--- PPC for Family: $family ---")
    println("RMSE: ", round(rmse, digits=4))
    println("Pearson R: ", round(pearson_val, digits=4), " (p=", round(pearson_p, digits=4), ")")
    println("Kendall τ: ", round(kendall_val, digits=4), " (p=", round(kendall_p, digits=4), ")")

    # 3. Density Visualization
    plt_density = density(y_obs, label="Observed", lw=2, color=:black)
    density!(plt_density, y_pred, label="Predicted (Denoised)", lw=2, color=:blue, ls=:dash)
    title!(plt_density, "PPC: Posterior Predictive Density ($family)")
    xlabel!(plt_density, "Value")
    ylabel!(plt_density, "Density")

    # 4. Scatterplot of Observed vs. Predicted
    plt_scatter = Plots.scatter(y_obs, y_pred, alpha=0.3, label="Observed vs Predicted",
                          xlabel="Observed", ylabel="Predicted", title="PPC: Model Fit Scatter ($family)")
    Plots.plot!(plt_scatter, [minimum(y_obs), maximum(y_obs)], [minimum(y_obs), maximum(y_obs)],
          lc=:red, ls=:dash, label="Identity")

    return (rmse=rmse, pearson=(val=pearson_val, p=pearson_p), kendall=(val=kendall_val, p=kendall_p), plot_density=plt_density, plot_scatter=plt_scatter)
end

function plot_posterior_results(stats, pts, spatial_res, W; time_slice=nothing, effect=:spatial, cov_idx=1, show_pts=false)
    # Description: Comprehensive posterior visualization for CARSTM and Deep GP models.

    # 1. Handle Categorical/Class Bar Plots
    if effect == :beta_cov
        b_stats = stats.beta_cov[cov_idx]
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1], 
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="Covariate $cov_idx Effects", xlabel="Level", ylabel="Effect Size", legend=false)
    
    elseif effect == :b_class1 || effect == :b_class2
        b_stats = effect == :b_class1 ? stats.b_class1 : stats.b_class2
        if isnothing(b_stats); error("Effect $effect not found in stats"); end
        n_levels = size(b_stats.mean, 1)
        return StatsPlots.bar(1:n_levels, b_stats.mean[:,1], 
                  yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="$effect Levels", xlabel="Class Index", ylabel="Effect Size", legend=false)

    # 2. Handle Temporal Main Effects
    elseif effect == :temporal
        t_stats = stats.temporal
        n_times = length(t_stats.mean)
        return StatsPlots.plot(1:n_times, t_stats.mean, 
                   ribbon=(t_stats.mean .- t_stats.lower, t_stats.upper .- t_stats.mean), 
                   fillalpha=0.2, lw=2, title="Temporal Main Effect", xlabel="Time Index", ylabel="Effect (Latent Scale)", legend=false)

    # 3. Handle Spatial, ST, and Deep GP Mean Fields
    elseif effect in [:spatial, :st_predictions_denoised, :st_predictions_noisy, :residuals, :eta_gp, :hidden_layer]
        plt = StatsPlots.plot(aspect_ratio=:equal, title="$effect (T=$(time_slice))", legend=true)

        # Determine the values to map to colors
        values = if effect == :spatial
            stats.spatial.mean
        elseif effect == :eta_gp
            # Visualize the final latent field from Deep GP layers
            haskey(stats, :eta_gp) ? stats.eta_gp.mean : error("eta_gp not found in stats")
        elseif effect == :hidden_layer
            # Visualize the hidden warping (Layer 1)
            haskey(stats, :h1) ? stats.h1.mean : error("hidden layer h1 not found in stats")
        elseif effect == :st_predictions_denoised && !isnothing(time_slice)
            stats.st_mat_denoised.mean[:, time_slice]
        elseif effect == :st_predictions_noisy && !isnothing(time_slice)
            stats.st_mat_noisy.mean[:, time_slice]
        elseif effect == :residuals
            # Unit-level aggregate residual check using denoised mean field as base
            stats.st_mat_noisy.mean[:, isnothing(time_slice) ? 1 : time_slice]
        else
            error("Effect $effect requires specific keys in stats or time_slice index")
        end

        for i in 1:length(spatial_res.polygons)
            poly_coords = spatial_res.polygons[i]
            if length(poly_coords) > 2
                px = [pt[1] for pt in poly_coords if !isnan(pt[1])]
                py = [pt[2] for pt in poly_coords if !isnan(pt[2])]
                
                if !isempty(px)
                    if (px[1], py[1]) != (px[end], py[end])
                        push!(px, px[1]); push!(py, py[1])
                    end
                    
                    val = values[i]
                    StatsPlots.plot!(plt, px, py, 
                        seriestype=:shape, 
                        fill_z=val, 
                        c=:RdYlBu, 
                        linecolor=:black, 
                        linewidth=0.5, 
                        fillalpha=0.8,
                        legend=false
                    )
                end
            end
        end

        if show_pts
            StatsPlots.scatter!(plt, [p[1] for p in pts], [p[2] for p in pts], 
                markersize=1, markercolor=:gray, alpha=0.2, label="Observations")
        end

        StatsPlots.scatter!(plt, [c[1] for c in spatial_res.centroids], [c[2] for c in spatial_res.centroids], 
            markersize=2, markercolor=:white, markerstrokecolor=:black, alpha=0.5, label="Centroids")

        return plt
    else
        error("Effect $effect not recognized.")
    end
end


function plot_posterior_vs_prior(model::DynamicPPL.Model, chain::MCMCChains.Chains, param_sym::Symbol; n_prior_samples=1000, title="Posterior vs Prior")
    # Description: Overlays posterior and prior densities for a specific parameter to check learning/shrinkage.
    # Inputs:
    #   - model: Turing model object.
    #   - chain: MCMC sample chain.
    #   - param_sym: Symbol of the parameter to check.
    # Outputs:
    #   - A Plots.Plot object.

    # 1. Extract posterior samples using .data for AxisArray compatibility
    post_samples = vec(chain[param_sym].data)

    # 2. Automated Prior Sampling via Turing
    prior_chain = sample(model, Prior(), n_prior_samples, progress=false)
    prior_samples = vec(prior_chain[param_sym].data)

    # 3. Visualization
    plt = density(post_samples, label="Posterior: $param_sym", lw=3, color=:blue, fill=(0, 0.2, :blue))
    density!(plt, prior_samples, label="Prior (sampled)", lw=2, ls=:dash, color=:red)

    title!(plt, title)
    xlabel!(plt, "Value")
    ylabel!(plt, "Density")

    return plt
end



function calculate_st_intervals(stats; alpha=0.05)
    # Description: Calculates credible intervals for Spatio-Temporal matrices (Placeholder implementation).
    # Inputs:
    #   - stats: summarized results from reconstruct_posteriors.
    # Outputs:
    #   - NamedTuple: (mean, lower, upper).

    # This function assumes stats contains the required matrices
    # In a full Bayesian implementation, this would iterate over multiple samples,
    # but here we provide the structure based on the current stats object.
    
    N_areas, N_time = size(stats.st_mat_denoised)
    
    # If stats only contains point estimates, we return those.
    # For true credible intervals, we would need the full sample 3D-array.
    
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2
    
    println("Note: Credible intervals for ST effects currently return point estimates.")
    println("To calculate full Bayesian CIs, pass the full sample array from the chain.")
    
    return (mean = stats.st_mat_denoised, 
            lower = stats.st_mat_denoised, 
            upper = stats.st_mat_denoised)
end


# --- 1. MODEL UTILITIES ---

function NegativeBinomial2(μ, r)
    # Description: Alternative parametrization of Negative Binomial using mean (μ) and dispersion (r).
    # Inputs:
    #   - μ: Mean.
    #   - r: Size/dispersion parameter.
    # Outputs:
    #   - Distributions.NegativeBinomial object.

    p = r / (r + μ)
    return NegativeBinomial(r, p)
end

function calculate_waic_weighted_likelihood(model::DynamicPPL.Model, chain::MCMCChains.Chains, y_obs, time_idx, area_idx, cov_matrix_indices, adj_matrix; weights=ones(length(y_obs)), offset=zeros(length(y_obs)))
    N_obs = length(y_obs)
    N_areas = size(adj_matrix, 1)
    N_time = maximum(time_idx)
    N_samples = size(chain, 1)

    log_lik_mat = zeros(N_samples, N_obs)

    # Determine the model family
    family = detect_model_family(model)
    if family != :gaussian
        @warn "This custom WAIC function is currently optimized for Gaussian models. Results for other families might be incorrect." 
    end

    for s in 1:N_samples
        # Extract scalar parameters for sample s
        sigma_y_s = chain[:sigma_y].data[s]
        sigma_sp_s = chain[:sigma_sp].data[s]
        phi_sp_s = chain[:phi_sp].data[s]
        sigma_tm_s = chain[:sigma_tm].data[s]
        rho_tm_s = chain[:rho_tm].data[s]
        sigma_int_s = chain[:sigma_int].data[s]

        # Extract indexed parameters for sample s
        u_icar_s = [chain["u_icar[" * string(i) * "]"].data[s] for i in 1:N_areas]
        u_iid_s = [chain["u_iid[" * string(i) * "]"].data[s] for i in 1:N_areas]
        f_tm_raw_s = [chain["f_tm_raw[" * string(i) * "]"].data[s] for i in 1:N_time]
        st_int_raw_s = [chain["st_int_raw[" * string(i) * "]"].data[s] for i in 1:(N_areas * N_time)]
        
        # beta_cov parameters are more complex, ensure correct indexing if used
        beta_cov_s = [Float64[] for _ in 1:4] # Initialize as vector of empty vectors
        for k in 1:4
            push!(beta_cov_s[k], [chain["beta_cov[" * string(k) * "][" * string(j) * "]"].data[s] for j in 1:13]...)
        end

        # Reconstruct effects for sample s
        s_eff_s = sigma_sp_s .* (sqrt(phi_sp_s) .* u_icar_s .+ sqrt(1 - phi_sp_s) .* u_iid_s)
        f_time_s = f_tm_raw_s .* sigma_tm_s
        st_interaction_s = reshape(st_int_raw_s, N_areas, N_time) .* sigma_int_s

        for i in 1:N_obs
            a, t = area_idx[i], time_idx[i]
            
            # Reconstruct mu for current observation i and sample s
            mu_i = offset[i] + s_eff_s[a] + f_time_s[t] + st_interaction_s[a, t]
            for k in 1:4
                mu_i += beta_cov_s[k][cov_matrix_indices[i, k]]
            end

            # Calculate log-likelihood for this observation and sample
            if family == :gaussian
                dist = Normal(mu_i, sigma_y_s)
            # Add other families here if needed
            else
                @error "Unsupported family for custom WAIC calculation: $family"
                return Inf
            end
            log_lik_mat[s, i] = weights[i] * logpdf(dist, y_obs[i])
        end
    end

    # WAIC calculation from log_lik_mat
    lppd = sum(logsumexp(log_lik_mat[:, i]) - log(N_samples) for i in 1:N_obs)
    p_waic = sum(var(log_lik_mat, dims=1))
    
    return -2 * (lppd - p_waic)
end


function calculate_waic(model::DynamicPPL.Model, chain::MCMCChains.Chains)
    # Description: Computes the Watanabe-Akaike Information Criterion (WAIC) for model comparison.
    # Inputs:
    #   - model/chain: Turing model and associated MCMC samples.
    # Outputs:
    #   - Float64 value (WAIC).

    try
        # Extract pointwise log-likelihoods using the official API
        pointwise_ll = Turing.pointwise_loglikelihoods(model, chain)

        # Get keys safely - handles LazyBundles and standard Dicts
        ks = collect(keys(pointwise_ll))

        if isempty(ks)
            return Inf
        end

        # Initialize the likelihood matrix with the first parameter key
        # Casting to Float64 ensures stability for sum and logsumexp
        log_lik_mat = Float64.(copy(pointwise_ll[ks[1]]))

        # Accumulate log-likelihoods if multiple variables are observed
        for i in 2:length(ks)
            log_lik_mat .+= pointwise_ll[ks[i]]
        end

        n_samples, n_obs = size(log_lik_mat)

        # lppd: log pointwise predictive density
        lppd = sum(logsumexp(log_lik_mat[:, i]) - log(n_samples) for i in 1:n_obs)

        # p_waic: effective number of parameters
        p_waic = sum(var(log_lik_mat, dims=1))

        return -2 * (lppd - p_waic)
    catch e
        @warn "WAIC calculation failed: $e"
        return Inf
    end
end

function get_rff_deep2D_basis(X, m, lengthscale)
    # Description: Generates Random Fourier Feature (RFF) basis for 2D inputs (Spatial/Temporal).
    # Inputs:
    #   - X: Input matrix (N x D).
    #   - m: Number of features.
    #   - lengthscale: Gaussian kernel lengthscale.
    # Outputs:
    #   - N x m feature matrix.
    N, D = size(X)
    Random.seed!(42)
    Omega_samples = randn(m, D) ./ lengthscale
    Phi_phases = rand(m) .* 2π
    return sqrt(2/m) .* cos.(X * Omega_samples' .+ Phi_phases')
end


function get_rff_trend_basis(t, m, lengthscale, ::Type{T}=Float64) where {T}
    N = length(t)
    # Generate random parameters for RFFs.
    # Using a seed ensures consistency within the AD pass.
    Random.seed!(42)
    Omega_samples_float = randn(m)
    Phi_phases_float = rand(m)

    Omega_samples = Omega_samples_float ./ lengthscale
    Phi_phases = Phi_phases_float .* convert(T, 2π)

    Z = zeros(T, N, m)
    for j in 1:m
        Z[:, j] = convert.(T, sqrt(2/m)) .* cos.(Omega_samples[j] .* t .+ Phi_phases[j])
    end
    return Z
end


function get_rff_seasonal_basis(t, m, freq, lengthscale)
    # Description: Generates RFF-style basis for periodic seasonal components.
    # Inputs:
    #   - t: Time vector.
    #   - m: Number of harmonics.
    #   - freq: Base frequency.
    #   - lengthscale: Smoothness scale.
    # Outputs:
    #   - N x (2*m) feature matrix.
    N = length(t)
    Z = zeros(N, 2*m)
    for j in 1:m
        omega_j = 2π * j * freq
        Z[:, 2j-1] = cos.(omega_j .* t)
        Z[:, 2j] = sin.(omega_j .* t)
    end
    return Z
end


function prepare_model_inputs(y, pts, area_idx, time_idx, W_sym, n_cat; m_trend=10, m_seas=5)
    """
    Consolidates static precomputations for the CARSTM model suite with an emphasis on numerical conditioning.
    """
    # 1. Standardize Time to [0, 1] range to prevent large Omega*t products
    N_time = maximum(time_idx)
    t_vec = collect(1:N_time) ./ N_time

    # 2. Spatial Scaling (BYM2)
    D_sp = Diagonal(vec(sum(W_sym, dims=2)))
    Q_sp_raw = D_sp - W_sym
    eigs_sp = filter(x -> x > 1e-6, eigvals(Matrix(Q_sp_raw)))
    scaling_sp_const = exp(mean(log.(eigs_sp)))
    Q_spatial_scaled = sparse(Q_sp_raw ./ scaling_sp_const)

    # 3. RW2 Scaling for categorical covariates
    D_rw2_mat = spzeros(Float64, n_cat - 2, n_cat)
    for i in 1:(n_cat - 2)
        D_rw2_mat[i, i] = 1.0
        D_rw2_mat[i, i+1] = -2.0
        D_rw2_mat[i, i+2] = 1.0
    end
    Q_rw2_raw = D_rw2_mat' * D_rw2_mat
    eigs_rw2 = filter(x -> x > 1e-6, eigvals(Matrix(Q_rw2_raw)))
    scaling_rw2_const = exp(mean(log.(eigs_rw2)))
    Q_rw2_scaled = sparse(Q_rw2_raw ./ scaling_rw2_const)

    # 4. Stable RFF Generation
    Random.seed!(42)
    Om_tr = randn(Float64, m_trend) ./ 0.5
    Ph_tr = rand(Float64, m_trend) .* 2π
    Z_trend = sqrt(2/m_trend) .* cos.(t_vec * Om_tr' .+ Ph_tr')

    Z_seas = zeros(Float64, N_time, 2 * m_seas)
    for j in 1:m_seas
        om_j = 2π * j
        Z_seas[:, 2j-1] = cos.(om_j .* t_vec)
        Z_seas[:, 2j] = sin.(om_j .* t_vec)
    end

    # 5. Templates and Indices
    n_areas = size(W_sym, 1)
    Q_ar1_template = spdiagm(0 => ones(N_time), 1 => fill(-1.0, N_time-1), -1 => fill(-1.0, N_time-1))
    interaction_idx = (time_idx .- 1) .* n_areas .+ area_idx
    
    N_obs = length(y)
    cov_mapping = zeros(Int, N_obs, 4)
    for k in 1:4; cov_mapping[:, k] .= mod1.(1:N_obs, n_cat); end

    return (
        y = y, pts_raw = pts, area_idx = area_idx, time_idx = time_idx,
        Q_sp = Q_spatial_scaled, Q_rw2 = Q_rw2_scaled, Q_ar1_template = Q_ar1_template,
        Z_trend = Z_trend, Z_seas = Z_seas,
        interaction_idx = interaction_idx, cov_indices = cov_mapping,
        n_cats = n_cat, scaling_sp_const = scaling_sp_const, scaling_rw2_const = scaling_rw2_const,
        weights = ones(N_obs), offset = zeros(N_obs)
    )
end



@model function model_v1_gaussian(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v1 Optimized: Foundational Gaussian Spatiotemporal model.
    # Decomposes the response into spatial (BYM2), temporal (AR1), and interaction effects.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Spatial Effect (BYM2) ---
    # Combines ICAR (structured) and IID (unstructured) components.
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    # Models temporal autocorrelation using a first-order autoregressive process.
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction (Type IV) ---
    # Captures localized deviations that vary over both space and time.
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Covariates (RW2 Smoothing) ---
    # Applies second-order random walk smoothing across categorical levels.
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + s_eff[modinputs.area_idx[i]] + f_time[modinputs.time_idx[i]] + st_interaction[modinputs.area_idx[i], modinputs.time_idx[i]]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


@model function model_v2_rff_gaussian(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v2 Optimized: Gaussian model replacing AR1 with Random Fourier Features (RFF).
    # Captures smooth non-linear trends and seasonality alongside spatial clustering.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0); sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    w_trend ~ MvNormal(zeros(size(modinputs.Z_trend, 2)), I); sigma_trend ~ Exponential(1.0)
    w_seas ~ MvNormal(zeros(size(modinputs.Z_seas, 2)), I); sigma_seas ~ Exponential(1.0)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Basis (RFF Trend & Seasonality) ---
    # Projects time into a high-dimensional space for non-linear trend/periodic effects.
    f_trend = modinputs.Z_trend * (w_trend .* sigma_trend)
    f_seas = modinputs.Z_seas * (w_seas .* sigma_seas)

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihood ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + f_trend[t] + f_seas[t] + s_eff[a] + st_interaction[a, t]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


@model function model_v3_lognormal(modinputs, ::Type{T}=Float64; offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v3 Optimized: LogNormal Spatiotemporal model for positive skewed data.
    # Employs a log-link to model the median of the distribution.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0); sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. LogNormal Likelihood ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(LogNormal(mu, sigma_y), y[i])
    end
end


@model function model_v4_binomial(modinputs, ::Type{T}=Float64; trials=ones(Int, length(modinputs.y)), offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v4 Optimized: Binomial Spatiotemporal model with Logit link.
    # Suitable for binary outcomes or proportion data across areas/time.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Binomial Likelihood (Logit Link) ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        eta = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4; eta += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(BinomialLogit(trials[i], eta), y[i])
    end
end


@model function model_v5_poisson(modinputs, ::Type{T}=Float64; use_zi=false, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v5 Optimized: Poisson Spatiotemporal model with optional Zero-Inflation.
    # Uses a log-link to ensure non-negative intensity (mu).

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)
    phi_zi ~ use_zi ? Beta(1, 1) : Dirac(0.0)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Poisson Likelihood (Log Link) ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        eta = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4; eta += beta_cov[k][modinputs.cov_indices[i, k]]; end
        mu = exp(eta)
        if use_zi
            Turing.@addlogprob! weights[i] * (y[i] == 0 ? log(phi_zi + (1 - phi_zi) * exp(-mu)) : log(1 - phi_zi) + logpdf(Poisson(mu), y[i]))
        else
            Turing.@addlogprob! weights[i] * logpdf(Poisson(mu), y[i])
        end
    end
end


@model function model_v6_negativebinomial(modinputs, ::Type{T}=Float64; use_zi=false, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v6 Optimized: Negative Binomial Spatiotemporal model.
    # Suitable for over-dispersed counts, with optional zero-inflation.

    y = modinputs.y
    N_obs, N_areas, N_time = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx)

    # --- 1. Priors ---
    sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_rw2 ~ filldist(Exponential(1.0), 4)
    r_nb ~ Exponential(1.0); phi_zi ~ use_zi ? Beta(1, 1) : Dirac(0.0)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Negative Binomial Likelihood ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        eta = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4; eta += beta_cov[k][modinputs.cov_indices[i, k]]; end
        mu = exp(eta); p_nb = r_nb / (r_nb + mu)
        if use_zi
            Turing.@addlogprob! weights[i] * (y[i] == 0 ? log(phi_zi + (1 - phi_zi) * pdf(NegativeBinomial(r_nb, p_nb), 0)) : log(1 - phi_zi) + logpdf(NegativeBinomial(r_nb, p_nb), y[i]))
        else
            Turing.@addlogprob! weights[i] * logpdf(NegativeBinomial(r_nb, p_nb), y[i])
        end
    end
end


@model function model_v7_deep_gp_binomial(modinputs, ::Type{T}=Float64; trials=ones(Int, length(modinputs.y)), m1=10, m2=5, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v7 Optimized: Deep Gaussian Process (GP) Spatiotemporal model with Binomial likelihood.
    # Uses Random Fourier Features (RFF) to approximate non-stationary spatio-temporal interactions.

    y = modinputs.y
    N_obs = length(y); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 1. Deep GP Priors ---
    # Lengthscales control the smoothness of the warping (Layer 1) and response (Layer 2).
    lengthscale1 ~ Gamma(2, 1); w1 ~ MvNormal(zeros(m1), I)
    lengthscale2 ~ Gamma(2, 1); w2 ~ MvNormal(zeros(m2), I)

    # --- 2. Feature Matrix Construction ---
    # Input features: Spatial X, Spatial Y, and Time.
    X = hcat([p[1] for p in modinputs.pts_raw], [p[2] for p in modinputs.pts_raw], Float64.(modinputs.time_idx))

    # --- 3. Layer 1 (Hidden Warp) ---
    # Projects (x,y,t) into a hidden feature space to capture complex interactions.
    Random.seed!(42); Om1 = randn(m1, 3) ./ lengthscale1; Ph1 = rand(m1) .* convert(T, 2π)
    h1 = (convert(T, sqrt(2/m1)) .* cos.(X * Om1' .+ Ph1')) * w1

    # --- 4. Layer 2 (Latent Response) ---
    # Maps the warped features to the latent linear predictor.
    Random.seed!(43); Om2 = randn(m2, 1) ./ lengthscale2; Ph2 = rand(m2) .* convert(T, 2π)
    eta_gp = (convert(T, sqrt(2/m2)) .* cos.(reshape(h1, :, 1) * Om2' .+ Ph2')) * w2

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Likelihood (Binomial Logit Link) ---
    for i in 1:N_obs
        eta = offset[i] + eta_gp[i]
        for k in 1:4; eta += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(BinomialLogit(trials[i], eta), y[i])
    end
end


@model function model_v8_deep_gp_gaussian(modinputs, ::Type{T}=Float64; m1=10, m2=5, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v8 Optimized: Refined 2-Layer Deep GP Spatiotemporal model for Gaussian data.
    # Combines deep non-linear interaction with robust modinputsuted covariate smoothing.

    y = modinputs.y
    N_obs = length(y); sigma_y ~ Exponential(1.0); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 1. Deep GP Priors ---
    l1 ~ Gamma(2, 1); w1 ~ MvNormal(zeros(m1), I)
    l2 ~ Gamma(2, 1); w2 ~ MvNormal(zeros(m2), I)

    # --- 2. Deep GP Layer 1 (Input Transformation) ---
    X = hcat([p[1] for p in modinputs.pts_raw], [p[2] for p in modinputs.pts_raw], Float64.(modinputs.time_idx))
    Random.seed!(42); Om1 = randn(m1, 3) ./ l1; Ph1 = rand(m1) .* convert(T, 2π)
    h1 = (convert(T, sqrt(2/m1)) .* cos.(X * Om1' .+ Ph1')) * w1

    # --- 3. Deep GP Layer 2 (Latent Mean Field) ---
    Random.seed!(43); Om2 = randn(m2, 1) ./ l2; Ph2 = rand(m2) .* convert(T, 2π)
    eta_gp = (convert(T, sqrt(2/m2)) .* cos.(reshape(h1, :, 1) * Om2' .+ Ph2')) * w2

    # --- 4. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 5. Gaussian Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + eta_gp[i]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


@model function model_v9_continuous_gaussian(continuous_covs, modinputs, ::Type{T}=Float64; m_feat=5, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v9 Optimized: Spatiotemporal model integrating continuous covariates via RFF-Matern Kernels.
    # Merges traditional BYM2 spatial effects with flexible non-linear covariate trends.

    y = modinputs.y
    N_obs, N_areas, N_time, N_covs = length(y), size(modinputs.Q_sp, 1), maximum(modinputs.time_idx), size(continuous_covs, 2)

    # --- 1. Priors ---
    sigma_y ~ Exponential(1.0); sigma_sp ~ Exponential(1.0); phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0); rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5); sigma_cov ~ filldist(Exponential(1.0), N_covs); lengthscale_cov ~ filldist(Gamma(2, 1), N_covs)

    # --- 2. Spatial Effect (BYM2) ---
    u_icar ~ MvNormal(zeros(N_areas), I); Turing.@addlogprob! -0.5 * dot(u_icar, modinputs.Q_sp * u_icar)
    u_iid ~ MvNormal(zeros(N_areas), I); s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # --- 3. Temporal Effect (AR1) ---
    Q_ar1 = (1.0 / (1.0 - rho_tm^2)) .* (modinputs.Q_ar1_template + (rho_tm^2) * I)
    f_tm_raw ~ MvNormal(zeros(N_time), I); Turing.@addlogprob! -0.5 * dot(f_tm_raw, Q_ar1 * f_tm_raw)
    f_time = f_tm_raw .* sigma_tm

    # --- 4. Space-Time Interaction ---
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I); st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # --- 5. Continuous Covariates (RFF Approximation) ---
    # Approximates a Matern kernel for each continuous covariate using Random Fourier Features.
    W_cov_raw ~ MvNormal(zeros(N_covs * m_feat), I); W_mat = reshape(W_cov_raw, m_feat, N_covs)
    f_cov_total = zeros(T, N_obs)
    for k in 1:N_covs
        Random.seed!(42 + k); Om = randn(m_feat) ./ lengthscale_cov[k]; Ph = rand(m_feat) .* convert(T, 2π)
        Z_k = convert(T, sqrt(2/m_feat)) .* cos.(continuous_covs[:, k] * Om' .+ Ph')
        f_cov_total .+= Z_k * (W_mat[:, k] .* sigma_cov[k])
    end

    # --- 6. Gaussian Likelihood ---
    for i in 1:N_obs
        a, t = modinputs.area_idx[i], modinputs.time_idx[i]
        mu = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t] + f_cov_total[i]
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end


@model function model_v10_deep_gp_3layer_gaussian(modinputs, ::Type{T}=Float64; m1=10, m2=5, m3=3, offset=modinputs.offset, weights=modinputs.weights) where {T}
    # Model v10 Optimized: 3-Layer Deep GP with Gaussian likelihood.
    # Hierarchical composition of GPs for capturing extremely complex spatio-temporal dynamics.

    y = modinputs.y
    N_obs = length(y); sigma_y ~ Exponential(1.0); sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # --- 1. 3-Layer Deep GP Priors ---
    l1 ~ Gamma(2, 1); w1 ~ MvNormal(zeros(m1), I)
    l2 ~ Gamma(2, 1); w2 ~ MvNormal(zeros(m2), I)
    l3 ~ Gamma(2, 1); w3 ~ MvNormal(zeros(m3), I)

    # --- 2. Layer 1 (Input Transformation) ---
    X = hcat([p[1] for p in modinputs.pts_raw], [p[2] for p in modinputs.pts_raw], Float64.(modinputs.time_idx))
    Random.seed!(42); Om1 = randn(m1, 3) ./ l1; Ph1 = rand(m1) .* convert(T, 2π)
    h1 = (convert(T, sqrt(2/m1)) .* cos.(X * Om1' .+ Ph1')) * w1

    # --- 3. Layer 2 (Non-linear Manifold Transformation) ---
    Random.seed!(43); Om2 = randn(m2, 1) ./ l2; Ph2 = rand(m2) .* convert(T, 2π)
    h2 = (convert(T, sqrt(2/m2)) .* cos.(reshape(h1, :, 1) * Om2' .+ Ph2')) * w2

    # --- 4. Layer 3 (Response Surface) ---
    Random.seed!(44); Om3 = randn(m3, 1) ./ l3; Ph3 = rand(m3) .* convert(T, 2π)
    eta_gp = (convert(T, sqrt(2/m3)) .* cos.(reshape(h2, :, 1) * Om3' .+ Ph3')) * w3

    # --- 5. Categorical Smoothing (RW2) ---
    beta_cov = [Vector{T}(undef, modinputs.n_cats) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(modinputs.n_cats), I)
        Turing.@addlogprob! -0.5 * dot(beta_cov[k], (modinputs.Q_rw2 ./ sigma_rw2[k]^2) * beta_cov[k])
    end

    # --- 6. Gaussian Likelihood ---
    for i in 1:N_obs
        mu = offset[i] + eta_gp[i]
        for k in 1:4; mu += beta_cov[k][modinputs.cov_indices[i, k]]; end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end



function model_results_summary(filename)
    # Description: Comprehensive post-processing using robust state loading and ST interval calculation.
    # Inputs:
    #   - filename: String path to the .jld2 result file.
    
    if !isfile(filename)
        error("File $filename not found.")
    end

    # 1. Load state using the optimized macro
    @load_carstm_state filename

    # 2. Basic Diagnostics
    sum_stats = summarystats(chain)
    min_ess = minimum(sum_stats[:, :ess_bulk])
    max_rhat = maximum(sum_stats[:, :rhat])

    # 3. Reconstruct Posteriors (Denoised/Noisy ST Fields)
    stats = reconstruct_posteriors(
        mod, 
        chain, 
        pts, 
        areal_units.assignments, 
        time_idx, 
        adj_matrix_numeric
    )

    # 4. Spatio-Temporal Credible Intervals
    # Extract alpha quantiles for the ST mean field
    st_mean = stats.st_mat_denoised.mean
    st_low = stats.st_mat_denoised.lower
    st_high = stats.st_mat_denoised.upper

    # 5. Accuracy Metrics
    y_pred = vec(stats.predictions.mean)
    rmse = sqrt(mean((y_sim .- y_pred).^2))
    mae = mean(abs.(y_sim .- y_pred))
    
    # Coverage Probability
    low_bound = vec(stats.predictions.lower)
    high_bound = vec(stats.predictions.upper)
    coverage = mean((y_sim .>= low_bound) .& (y_sim .<= high_bound))

    # 6. Visualization
    ppc = posterior_predictive_check(mod, stats, y_sim)
    
    # Spatial Plot based on centroids
    plt_sp = plot_posterior_results(stats, pts, areal_units.centroids, adj_matrix_numeric; effect=:spatial)

    return (
        summarystats = sum_stats,
        min_ess = min_ess,
        max_rhat = max_rhat,
        rmse = rmse,
        mae = mae,
        coverage_prob = coverage,
        st_intervals = (mean=st_mean, lower=st_low, upper=st_high),
        ppc = ppc,
        plt_spatial = plt_sp,
        sigma_sp = haskey(chain, :sigma_sp) ? density(chain[:sigma_sp], title="Spatial Sigma Posterior") : nothing
    )
end


function prepare_model_inputs_direct(D, W, X, log_offset, y, time_idx, nX, nAU, node1, node2; n_cat=13)
    """
    Synopsis: Adapts raw data to the standardized prepare_model_inputs format for CARSTM modeling.
    """
    # Create a vector of tuples for coordinates (placeholder since this is areal data)
    # We'll use the centroids of a dummy grid or just zeroed tuples if specific coords aren't needed
    # for the discrete model components.
    pts_dummy = [(0.0, 0.0) for _ in 1:length(y)]

    # The area_idx is the spatial unit identifier (1 to 56)
    area_idx = repeat(1:nAU, 10)

    # Standardize time indices
    t_idx = Int.(time_idx)

    # Use the existing robust prepare_model_inputs logic to compute scalings and bases
    # Note: We pass n_cat which is used for the RW2 categorical smoothing priors
    mod_data = prepare_model_inputs(
        y, 
        pts_dummy, 
        area_idx, 
        t_idx, 
        W, 
        n_cat
    )

    # Override/Update with Lip Cancer specific inputs
    # We merge the log_offset (expected cases) and the actual covariate matrix X
    return merge(mod_data, (
        X = X, 
        offset = log_offset, 
        n_fixed = nX,
        node1 = node1,
        node2 = node2
    ))
    # Example usage:
    # lip_inputs = prepare_model_inputs_direct(D, W, Xnew, log_offset_new, ynew, ti, nX, nAU, node1, node2)
    # display(lip_inputs)
end


function assign_spatial_units_inferred(adjacency_matrix, nAU; iterations=50, learning_rate=0.1, buffer_dist=0.5, input_polygons = nothing)
    """
    Synopsis: Manually constructs a areal_units object for areal data like the Lip Cancer dataset.
              Centroid locations are spatially inferred from connectivity using a rudimentary force-directed layout.
    Inputs:
    - adjacency_matrix: The adjacency matrix (W) of the areal units.
    - nAU: Number of areal units
    - iterations: Number of iterations for the force-directed layout.
    - learning_rate: Step size for moving centroids in the layout algorithm.
    - buffer_dist: Distance to buffer the convex hull when polygons are inferred.
    - input_polygons: Optional. A vector of LibGEOS Polygons. If provided, centroids and hull are derived from these.
    """

    local final_centroids
    local adjacency_edges_output
    local polys_output
    local hull_coords_output
    local g_final # The final graph that will be in the result

    if input_polygons !== nothing && !isempty(input_polygons)
        # Case 1: Polygons are provided
        # 1. Extract centroids from input_polygons
        final_centroids_geoms = [LibGEOS.centroid(p) for p in input_polygons]
        final_centroids = map(final_centroids_geoms) do g_pt
            seq = LibGEOS.getCoordSeq(g_pt)
            (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        end

        # 2. Determine hull by dissolving all internal edges
        united_geom = LibGEOS.unaryunion(input_polygons)
        hull_coords_output = get_coords_from_geom(united_geom)

        # 3. Determine adjacency from input_polygons (using LibGEOS.touches)
        adjacency_edges_output = []
        for i in 1:nAU
            g1 = input_polygons[i]
            for j in (i+1):nAU
                g2 = input_polygons[j]
                if LibGEOS.touches(g1, g2)
                    push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                else
                    # Fallback robust check, similar to get_voronoi_polygons_and_edges
                    g1_buffered = LibGEOS.buffer(g1, 1e-6)
                    if LibGEOS.intersects(g1_buffered, g2)
                        inter = LibGEOS.intersection(g1_buffered, g2)
                        if !LibGEOS.isEmpty(inter) && (LibGEOS.area(inter) > 1e-9 || LibGEOS.geomTypeId(inter) in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_MULTILINESTRING])
                            push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                        end
                    end
                end
            end
        end

        polys_output = [get_coords_from_geom(p) for p in input_polygons]

        # Build graph from the determined adjacency edges and ensure connectivity
        g_final = SimpleGraph(nAU)
        centroid_map = Dict(c => i for (i, c) in enumerate(final_centroids))
        for (c1, c2) in adjacency_edges_output
            u_idx = get(centroid_map, c1, 0)
            v_idx = get(centroid_map, c2, 0)
            if u_idx > 0 && v_idx > 0 && !has_edge(g_final, u_idx, v_idx)
                add_edge!(g_final, u_idx, v_idx)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids) # Ensure connectivity if necessary

    else
        # Case 2: Polygons are not provided, infer centroids and use tessellation
        # 1. Build initial graph from adjacency_matrix for force-directed layout
        g_initial_for_layout = SimpleGraph(adjacency_matrix)

        # 2. Infer initial centroids using force-directed layout
        side = ceil(Int, sqrt(nAU))
        initial_centroids_fd = [(Float64(i % side), Float64(i ÷ side)) for i in 0:(nAU-1)]
        centroids_vec = [SVector{2, Float64}(c) for c in initial_centroids_fd]

        for iter in 1:iterations
            new_centroids_vec = copy(centroids_vec)
            for i in 1:nAU
                neighbors_i = Graphs.neighbors(g_initial_for_layout, i)
                if !isempty(neighbors_i)
                    avg_neighbor_pos = sum(centroids_vec[n] for n in neighbors_i) / length(neighbors_i)
                    new_centroids_vec[i] = centroids_vec[i] + learning_rate * (avg_neighbor_pos - centroids_vec[i])
                end
            end
            centroids_vec = new_centroids_vec
        end
        # Centroids after force-directed layout
        forced_layout_centroids = [(p[1], p[2]) for p in centroids_vec]

        # 3. Determine hull_geom from inferred centroids for clipping
        hull_geom = expand_hull(forced_layout_centroids, buffer_dist)
        hull_coords_output = get_coords_from_geom(hull_geom)

        # 4. Use tessellation to determine polygon coordinates and initial adjacency (based on forced_layout_centroids)
        polys_coords_raw, _ = get_voronoi_polygons_and_edges(forced_layout_centroids, hull_geom) # Discard initial edges as they refer to old centroids

        # 5. RECOMPUTE CENTROIDS from the generated (clipped) polygons and prepare for adjacency
        final_centroids = Vector{Tuple{Float64, Float64}}(undef, length(polys_coords_raw))
        lg_polygons_for_adjacency = Vector{Union{LibGEOS.Polygon, Nothing}}(undef, length(polys_coords_raw)) # Allow Nothing
        polys_output = polys_coords_raw # Keep the raw coordinates for output

        for (idx, poly_coord_list) in enumerate(polys_coords_raw)
            if !isempty(poly_coord_list) && length(poly_coord_list) >= 3 # Ensure it's a valid polygon
                # Make sure the polygon is closed for LibGEOS
                if poly_coord_list[1] != poly_coord_list[end]
                    push!(poly_coord_list, poly_coord_list[1])
                end
                lg_poly = LibGEOS.Polygon([ [Float64[p[1], p[2]] for p in poly_coord_list] ])
                centroid_geom = LibGEOS.centroid(lg_poly)
                seq = LibGEOS.getCoordSeq(centroid_geom)
                final_centroids[idx] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
                lg_polygons_for_adjacency[idx] = lg_poly
            else
                @warn "Invalid or empty polygon encountered in Voronoi tessellation at index $idx. Using original centroid as fallback, and skipping polygon for adjacency checks."
                final_centroids[idx] = forced_layout_centroids[idx]
                lg_polygons_for_adjacency[idx] = nothing # Mark as invalid for adjacency checks
            end
        end

        # 6. Re-build adjacency based on the newly derived centroids and polygons
        adjacency_edges_output = []
        if !isempty(lg_polygons_for_adjacency)
            for i in 1:length(lg_polygons_for_adjacency)
                g1 = lg_polygons_for_adjacency[i]
                if g1 === nothing continue end # Skip if polygon is invalid
                for j in (i+1):length(lg_polygons_for_adjacency)
                    g2 = lg_polygons_for_adjacency[j]
                    if g2 === nothing continue end # Skip if polygon is invalid
                    # Check for adjacency using LibGEOS predicates
                    if LibGEOS.touches(g1, g2)
                        push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                    else
                        # Fallback: Robust check using a tiny buffer for floating-point misalignments
                        g1_buffered = LibGEOS.buffer(g1, 1e-6)
                        if LibGEOS.intersects(g1_buffered, g2)
                            inter = LibGEOS.intersection(g1_buffered, g2)
                            # Check if intersection is a line or has significant area
                            if !LibGEOS.isEmpty(inter) && (LibGEOS.area(inter) > 1e-9 || LibGEOS.geomTypeId(inter) in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_MULTILINESTRING])
                                push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                            end
                        end
                    end
                end
            end
        end

        # 7. Build final graph from the re-derived adjacency edges and ensure connectivity
        g_final = SimpleGraph(nAU) # nAU is the count of regions
        centroid_map = Dict(c => i for (i, c) in enumerate(final_centroids)) # Map new centroids to indices
        for (c1, c2) in adjacency_edges_output
            u_idx = get(centroid_map, c1, 0)
            v_idx = get(centroid_map, c2, 0)
            if u_idx > 0 && v_idx > 0 && !has_edge(g_final, u_idx, v_idx)
                add_edge!(g_final, u_idx, v_idx)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids)
    end

    return (
        centroids = final_centroids,
        adjacency_edges = adjacency_edges_output,
        graph = g_final,
        polygons = polys_output,
        hull_coords = hull_coords_output
    )
    
    """
    # Generate the spatial metadata for the Lip Cancer dataset
    areal_units = assign_spatial_units_inferred(W, nAU)
    println("Spatial metadata created for Scottish Lip Cancer dataset.")
    println("Number of units: ", length(areal_units.centroids))
    println("Graph connectivity: ", is_connected(areal_units.graph))
    
    # Quick test run: model 2 using explicit unpacking of required fields
    plt = plot_spatial_graph(lip_inputs.pts_raw, areal_units; title="Lip Cancer Spatial Graph", domain_boundary=lip_inputs.pts_raw)
    display(plt)
    
    println("First few centroids from areal_units: ", areal_units.centroids[1:min(5, length(areal_units.centroids))])
    """
end