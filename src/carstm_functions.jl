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


function plot_kde_simple(pts; grid_res=600, sd_extension_factor=1.0, title="Spatial Intensity (KDE)")
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

    Threads.@threads for f in 1:nvar
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


function plot_model_fit(chain, y_sim )
    # Description: Plots Observed vs Predicted values to visualize overall fit quality.
    # Inputs:
    #   - chain: MCMC sample chain containing 'mu' parameters.
    #   - y_sim: Vector of observed values.
    # Outputs:
    #   - A Plots.Plot object (Scatter with Identity line).
    # Simplified fit check: Mean of y_sim vs Mean of posterior predictions
    # Note: Requires reconstruct_posteriors to have run
    # Here we just calculate a quick RMSE based on the denoised mu
    
    # Reconstruct inside for convenience if not passed
    N_obs = length(y_sim)
    mu_post = zeros(N_obs)
    
    # Check if we have a mu parameter in the chain
    if "mu[1]" in names(chain)
        mu_post = [mean(chain["mu[$i]"]) for i in 1:N_obs]
    else
        # Fallback to reconstructing if possible (placeholder logic)
        println("Parameter 'mu' not found in chain for direct plotting.")
        return nothing
    end

    plt = scatter(y_sim, mu_post, alpha=0.3, label="Observed vs Predicted",
                  xlabel="Observed", ylabel="Predicted", title="Model Fit Check")
    plot!(plt, [minimum(y_sim), maximum(y_sim)], [minimum(y_sim), maximum(y_sim)], 
          lc=:red, ls=:dash, label="Identity")
    
    display(plt)
    return plt
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



function plot_posterior_results(stats, pts, centroids, W; time_slice=nothing, effect=:spatial, cov_idx=1)
    # Description: Visualizes specific model effects (spatial maps or categorical levels) as heatmaps or bar charts.
    # Inputs:
    #   - stats: NamedTuple from reconstruct_posteriors.
    #   - centroids/W: Spatial structure definitions.
    #   - time_slice: Index for ST slices.
    #   - effect: :spatial, :st_predictions_denoised, :beta_cov, etc.
    # Outputs:
    #   - A Plots.Plot object.

    grid_res = 100; x_g = range(0, 10, length=grid_res); y_g = range(0, 10, length=grid_res)
    find_idx(xi, yi) = argmin([sqrt((xi-p[1])^2 + (yi-p[2])^2) for p in centroids])

    if effect in [:spatial, :st_predictions_denoised, :st_predictions_noisy]
        z = if effect == :spatial
            [stats.spatial.mean[find_idx(xi, yi)] for xi in x_g, yi in y_g]
        elseif effect == :st_predictions_denoised && !isnothing(time_slice)
            [stats.st_mat_denoised.mean[find_idx(xi, yi), time_slice] for xi in x_g, yi in y_g]
        elseif effect == :st_predictions_noisy && !isnothing(time_slice)
            [stats.st_mat_noisy.mean[find_idx(xi, yi), time_slice] for xi in x_g, yi in y_g]
        else
            error("Effect $effect not supported or missing time_slice")
        end
        plt = heatmap(x_g, y_g, z', title="$effect (T=$(time_slice))", color=:RdYlBu, aspect_ratio=:equal)
        return plt
    elseif effect == :beta_cov
        # Plotting categorical levels for a specific covariate
        b_stats = stats.beta_cov[cov_idx]
        n_levels = size(b_stats.mean, 1)
        plt = bar(1:n_levels, b_stats.mean[:,1], yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="Covariate $cov_idx Effects", xlabel="Level", ylabel="Effect Size", legend=false)
        return plt
    elseif effect == :b_class1 || effect == :b_class2
        b_stats = effect == :b_class1 ? stats.b_class1 : stats.b_class2
        if isnothing(b_stats); error("Effect $effect not found in stats"); end
        n_levels = size(b_stats.mean, 1)
        plt = bar(1:n_levels, b_stats.mean[:,1], yerror=(b_stats.mean[:,1] .- b_stats.lower[:,1], b_stats.upper[:,1] .- b_stats.mean[:,1]),
                  title="$effect Levels", xlabel="Class Index", ylabel="Effect Size", legend=false)
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

function get_rff_trend_basis(t, m, lengthscale)
    # Description: Generates RFF basis for 1D temporal trends.
    # Inputs:
    #   - t: Time vector.
    #   - m: Number of features.
    #   - lengthscale: Trend smoothness scale.
    # Outputs:
    #   - N x m feature matrix.
    N = length(t)
    Random.seed!(42)
    Omega_samples = randn(m) ./ lengthscale
    Phi_phases = rand(m) .* 2π
    Z = zeros(N, m)
    for j in 1:m
        Z[:, j] = sqrt(2/m) .* cos.(Omega_samples[j] .* t .+ Phi_phases[j])
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




### Model V1 Audit: Advanced CARSTM (GMRF + AR1 + Interaction)
@model function model_v1_gaussian(y, time_idx, area_idx, cov_matrix_indices, adj_matrix; offset=zeros(length(y)), weights=ones(length(y)))
    N_obs, N_areas, N_time = length(y), size(adj_matrix, 1), maximum(time_idx)

    # 1. Precision Matrices
    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))
    Q_rw2 = scale_precision!(build_rw2_precision(13)) # Smoothing for categorical levels

    # 2. Priors
    sigma_y ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2) # AR1 correlation parameter
    sigma_int ~ Exponential(0.5) # Interaction scale
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # 3. Spatial Effect (BYM2)
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # 4. Temporal Effect (AR1)
    Q_ar1 = build_ar1_precision(N_time, rho_tm, 1.0)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_tm_raw, Q_ar1)
    f_time = f_tm_raw .* sigma_tm

    # 5. Space-Time Interaction (Type IV approximation)
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # 6. Smooth Categorical Covariates (RW2)
    beta_cov = [Vector{Real}(undef, 13) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(13), I)
        Turing.@addlogprob! logpdf_gmrf(beta_cov[k], (1/sigma_rw2[k]^2) * Q_rw2)
    end

    for i in 1:N_obs
        a, t = area_idx[i], time_idx[i]
        mu = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4
            mu += beta_cov[k][cov_matrix_indices[i, k]]
        end
        # Weighted Likelihood
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end
### Model V2 Audit: RFF CARSTM (Integrated Covariates & Interaction)
@model function model_v2_carstm_rff(y, time_idx, area_idx, cov_matrix_indices, adj_matrix; m_trend=10, m_seas=5, offset=zeros(length(y)), weights=ones(length(y)))
    N_obs, N_areas, N_time = length(y), size(adj_matrix, 1), maximum(time_idx)
    t_vec = collect(1:N_time) ./ N_time

    # 1. Temporal basis functions
    Z_trend = get_rff_trend_basis(t_vec, m_trend, 0.5)
    Z_seas = get_rff_seasonal_basis(t_vec, m_seas, 1.0/N_time, 0.1)

    # 2. Precision Matrices
    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))
    Q_rw2 = scale_precision!(build_rw2_precision(13))

    # 3. Priors
    sigma_y ~ Exponential(1.0)
    w_trend ~ MvNormal(zeros(m_trend), I)
    w_seas ~ MvNormal(zeros(size(Z_seas, 2)), I)
    sigma_trend ~ Exponential(1.0)
    sigma_seas ~ Exponential(1.0)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)
    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_int ~ Exponential(0.5)

    # 4. Categorical Covariates (RW2 Prior)
    beta_cov = [Vector{Real}(undef, 13) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(13), I)
        Turing.@addlogprob! logpdf_gmrf(beta_cov[k], (1/sigma_rw2[k]^2) * Q_rw2)
    end

    # 5. Spatial BYM2 components
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # 6. Interaction Effect
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # 7. Reconstruct effects
    f_trend = Z_trend * (w_trend .* sigma_trend)
    f_seas = Z_seas * (w_seas .* sigma_seas)

    for i in 1:N_obs
        a, t = area_idx[i], time_idx[i]
        mu = offset[i] + f_trend[t] + f_seas[t] + s_eff[a] + st_interaction[a, t]
        for k in 1:4
            mu += beta_cov[k][cov_matrix_indices[i, k]]
        end
        # Weighted Likelihood
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end
### Model V3 Audit: LogNormal CARSTM (Weighted)
@model function model_v3_lognormal(y, time_idx, area_idx, cov_matrix_indices, adj_matrix; offset=zeros(length(y)), weights=ones(length(y)))
    N_obs, N_areas, N_time = length(y), size(adj_matrix, 1), maximum(time_idx)
    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))
    Q_rw2 = scale_precision!(build_rw2_precision(13))

    # Priors
    sigma_y ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # Spatial Effect (BYM2)
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # Temporal Effect (AR1)
    Q_ar1 = build_ar1_precision(N_time, rho_tm, 1.0)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_tm_raw, Q_ar1)
    f_time = f_tm_raw .* sigma_tm

    # Interaction
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # Smooth Categorical Covariates (RW2)
    beta_cov = [Vector{Real}(undef, 13) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(13), I)
        Turing.@addlogprob! logpdf_gmrf(beta_cov[k], (1/sigma_rw2[k]^2) * Q_rw2)
    end

    for i in 1:N_obs
        a, t = area_idx[i], time_idx[i]
        mu = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4
            mu += beta_cov[k][cov_matrix_indices[i, k]]
        end
        # Weighted Log-Likelihood
        Turing.@addlogprob! weights[i] * logpdf(LogNormal(mu, sigma_y), y[i])
    end
end
### Model V4 Audit: Binomial CARSTM (Weighted)
@model function model_v4_binomial(y, time_idx, area_idx, cov_matrix_indices, adj_matrix, n_trials, class1, class2; offset=zeros(length(y)), weights=ones(length(y)))
    N_obs, N_areas, N_time = length(y), size(adj_matrix, 1), maximum(time_idx)

    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))
    Q_rw2 = scale_precision!(build_rw2_precision(13))

    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # Spatial BYM2
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # Temporal AR1
    Q_ar1 = build_ar1_precision(N_time, rho_tm, 1.0)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_tm_raw, Q_ar1)
    f_time = f_tm_raw .* sigma_tm

    # Interaction
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # Smooth Categorical Covariates (RW2)
    beta_cov = [Vector{Real}(undef, 13) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(13), I)
        Turing.@addlogprob! logpdf_gmrf(beta_cov[k], (1/sigma_rw2[k]^2) * Q_rw2)
    end

    for i in 1:N_obs
        a, t = area_idx[i], time_idx[i]
        eta = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4
            eta += beta_cov[k][cov_matrix_indices[i, k]]
        end
        # Weighted Likelihood
        Turing.@addlogprob! weights[i] * logpdf(BinomialLogit(n_trials[i], eta), y[i])
    end
end
### Model V5: (ZI)P CARSTM
@model function model_v5_poisson(y, time_idx, area_idx, cov_matrix_indices, adj_matrix, class1, class2; weights=ones(length(y)), offset=zeros(length(y)), use_zi=true)
    N_obs, N_areas, N_time = length(y), size(adj_matrix, 1), maximum(time_idx)
    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))
    Q_rw2 = scale_precision!(build_rw2_precision(13))

    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # Spatial
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # Temporal
    Q_ar1 = build_ar1_precision(N_time, rho_tm, 1.0)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_tm_raw, Q_ar1)
    f_time = f_tm_raw .* sigma_tm

    # Interaction
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # Smooth Categorical Covariates (RW2)
    beta_cov = [Vector{Real}(undef, 13) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(13), I)
        Turing.@addlogprob! logpdf_gmrf(beta_cov[k], (1/sigma_rw2[k]^2) * Q_rw2)
    end

    if use_zi; phi_zi ~ Beta(1, 1); end

    for i in 1:N_obs
        a, t = area_idx[i], time_idx[i]
        eta = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4
            eta += beta_cov[k][cov_matrix_indices[i, k]]
        end
        lambda_i = exp(eta)
        dist_poi = Poisson(lambda_i)

        if use_zi
            ll = y[i] == 0 ? log(phi_zi + (1 - phi_zi) * exp(-lambda_i)) : log(1 - phi_zi) + logpdf(dist_poi, y[i])
        else
            ll = logpdf(dist_poi, y[i])
        end
        Turing.@addlogprob! weights[i] * ll
    end
end
### Model V6: (ZI)NB CARSTM
@model function model_v6_negativebinomial(y, time_idx, area_idx, cov_matrix_indices, adj_matrix, class1, class2; weights=ones(length(y)), offset=zeros(length(y)), use_zi=true)
    N_obs, N_areas, N_time = length(y), size(adj_matrix, 1), maximum(time_idx)
    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))
    Q_rw2 = scale_precision!(build_rw2_precision(13))

    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5)
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # Spatial
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # Temporal
    Q_ar1 = build_ar1_precision(N_time, rho_tm, 1.0)
    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_tm_raw, Q_ar1)
    f_time = f_tm_raw .* sigma_tm

    # Interaction
    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # Smooth Categorical Covariates (RW2)
    beta_cov = [Vector{Real}(undef, 13) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(13), I)
        Turing.@addlogprob! logpdf_gmrf(beta_cov[k], (1/sigma_rw2[k]^2) * Q_rw2)
    end

    if use_zi; phi_zi ~ Beta(1, 1); end
    r_nb ~ Exponential(1.0)

    for i in 1:N_obs
        a, t = area_idx[i], time_idx[i]
        eta = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t]
        for k in 1:4
            eta += beta_cov[k][cov_matrix_indices[i, k]]
        end
        mu = exp(eta)
        dist_nb = NegativeBinomial2(mu, r_nb)

        if use_zi
            ll = y[i] == 0 ? log(phi_zi + (1 - phi_zi) * pdf(dist_nb, 0)) : log(1 - phi_zi) + logpdf(dist_nb, y[i])
        else
            ll = logpdf(dist_nb, y[i])
        end
        Turing.@addlogprob! weights[i] * ll
    end
end
### Model V7: Deep Gaussian Process (RFF) CARSTM (Binomial)
@model function model_v7_deep_gaussianprocess_binomial(y, trials, time_idx, area_idx, pts_raw, cov_matrix_indices, adj_matrix, class1, class2; weights=ones(length(y)), offset=zeros(length(y)), use_zi=false)
    N_obs = length(y)
    m_layer1, m_layer2 = 10, 5

    # Precision for categorical covariates
    Q_rw2 = scale_precision!(build_rw2_precision(13))
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # Deep GP Layers
    X = hcat([p[1] for p in pts_raw], [p[2] for p in pts_raw], Float64.(time_idx))
    lengthscale1 ~ Gamma(2, 1)
    basis1 = get_rff_deep2D_basis(X, m_layer1, lengthscale1)
    w1 ~ MvNormal(zeros(m_layer1), I)
    hidden_layer = basis1 * w1

    lengthscale2 ~ Gamma(2, 1)
    basis2 = get_rff_deep2D_basis(reshape(hidden_layer, :, 1), m_layer2, lengthscale2)
    w2 ~ MvNormal(zeros(m_layer2), I)
    eta_gp = basis2 * w2

    # Smooth Categorical Covariates (RW2)
    beta_cov = [Vector{Real}(undef, 13) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(13), I)
        Turing.@addlogprob! logpdf_gmrf(beta_cov[k], (1/sigma_rw2[k]^2) * Q_rw2)
    end

    if use_zi; phi_zi ~ Beta(1, 1); end

    for i in 1:N_obs
        eta = offset[i] + eta_gp[i]
        for k in 1:4
            eta += beta_cov[k][cov_matrix_indices[i, k]]
        end
        
        dist_bin = BinomialLogit(trials[i], eta)
        if use_zi
            ll = y[i] == 0 ? log(phi_zi + (1 - phi_zi) * pdf(dist_bin, 0)) : log(1 - phi_zi) + logpdf(dist_bin, y[i])
        else
            ll = logpdf(dist_bin, y[i])
        end
        Turing.@addlogprob! weights[i] * ll
    end
end
### Model V8: Deep Gaussian Process (RFF) CARSTM (Gaussian)
@model function model_v8_deep_gaussianprocess_gaussian(y, time_idx, area_idx, pts_raw, cov_matrix_indices, adj_matrix, class1, class2; weights=ones(length(y)), offset=zeros(length(y)))
    N_obs = length(y)
    m_layer1, m_layer2 = 10, 5

    # Precision for categorical covariates
    Q_rw2 = scale_precision!(build_rw2_precision(13))
    sigma_rw2 ~ filldist(Exponential(1.0), 4)

    # Deep GP Layers via Random Fourier Features
    X = hcat([p[1] for p in pts_raw], [p[2] for p in pts_raw], Float64.(time_idx))
    lengthscale1 ~ Gamma(2, 1)
    basis1 = get_rff_deep2D_basis(X, m_layer1, lengthscale1)
    w1 ~ MvNormal(zeros(m_layer1), I)
    hidden_layer = basis1 * w1

    lengthscale2 ~ Gamma(2, 1)
    basis2 = get_rff_deep2D_basis(reshape(hidden_layer, :, 1), m_layer2, lengthscale2)
    w2 ~ MvNormal(zeros(m_layer2), I)
    eta_gp = basis2 * w2

    # Smooth Categorical Covariates (RW2)
    beta_cov = [Vector{Real}(undef, 13) for _ in 1:4]
    for k in 1:4
        beta_cov[k] ~ MvNormal(zeros(13), I)
        Turing.@addlogprob! logpdf_gmrf(beta_cov[k], (1/sigma_rw2[k]^2) * Q_rw2)
    end

    sigma_y ~ Exponential(1.0)

    for i in 1:N_obs
        mu = offset[i] + eta_gp[i]
        for k in 1:4
            mu += beta_cov[k][cov_matrix_indices[i, k]]
        end
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end
### Model V9 Audit: Continuous Covariate CARSTM (RFF-Matern)
@model function model_v9_continuous_gaussian(y, time_idx, area_idx, continuous_covs, adj_matrix; m_feat=10, offset=zeros(length(y)), weights=ones(length(y)))
    N_obs, N_areas, N_time = length(y), size(adj_matrix, 1), maximum(time_idx)
    N_covs = size(continuous_covs, 2)

    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))

    sigma_y ~ Exponential(1.0)
    sigma_sp ~ Exponential(1.0)
    phi_sp ~ Beta(1, 1)
    sigma_tm ~ Exponential(1.0)
    rho_tm ~ Beta(2, 2)
    sigma_int ~ Exponential(0.5)

    sigma_cov ~ filldist(Exponential(1.0), N_covs)
    lengthscale_cov ~ filldist(Gamma(2, 1), N_covs)

    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    f_tm_raw ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_tm_raw, build_ar1_precision(N_time, rho_tm, 1.0))
    f_time = f_tm_raw .* sigma_tm

    st_int_raw ~ MvNormal(zeros(N_areas * N_time), I)
    st_interaction = reshape(st_int_raw .* sigma_int, N_areas, N_time)

    # Fix: Sample a matrix of weights to avoid name collision in the loop
    W_cov ~ MvNormal(zeros(N_covs * m_feat), I)
    W_mat = reshape(W_cov, m_feat, N_covs)

    f_cov_total = zeros(N_obs)
    for k in 1:N_covs
        Z_k = get_rff_trend_basis(continuous_covs[:, k], m_feat, lengthscale_cov[k])
        f_cov_total += Z_k * (W_mat[:, k] .* sigma_cov[k])
    end

    for i in 1:N_obs
        a, t = area_idx[i], time_idx[i]
        mu = offset[i] + s_eff[a] + f_time[t] + st_interaction[a, t] + f_cov_total[i]
        # Applying weights to the Gaussian log-likelihood
        Turing.@addlogprob! weights[i] * logpdf(Normal(mu, sigma_y), y[i])
    end
end