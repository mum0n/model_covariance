
 

function init_params_extract( res=NaN; load_from_file=false, override_means=false, fn_inits = "init_params.jl2"  )
 
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


Turing.@model function pca_carstm( Y, ::Type{T}=Float64 ) where {T}
  # X, G, log_offset, y, z, auid, nData, nX, nG, nAU, node1, node2, scaling_factor 
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
 

function lattice_adjacency_matrix(rows, cols)
    geoms = []
    for r in 1:rows, c in 1:cols
        # Create a unit square for each cell
        poly = ArchGDAL.createpolygon([
            (Float64(c-1), Float64(r-1)), (Float64(c), Float64(r-1)),
            (Float64(c), Float64(r)), (Float64(c-1), Float64(r)),
            (Float64(c-1), Float64(r-1))
        ])
        push!(geoms, poly)
    end
    
    n = length(geoms)
    W = spzeros(Int, n, n)
    for i in 1:n
        prep_i = ArchGDAL.preparegeom(geoms[i])
        for j in (i+1):n
            # Queen contiguity (any shared point)
            if ArchGDAL.intersects(prep_i, geoms[j])
                W[i, j] = W[j, i] = 1
            end
        end
    end
    return W
end




function generate_sim_data(n_pts, n_time; rndseed=42 )
    n_total = n_pts * n_time
    Random.seed!(rndseed)
    pts = [(rand() * 10, rand() * 10) for _ in 1:n_pts]
    time_idx = repeat(1:n_time, inner=n_pts)
    weights = ones(n_total)
    trials = ones(Int, n_total)
    cov_indices = rand(1:3, n_total)
    spatial_effect = [sin(p[1]/2) + cos(p[2]/2) for p in pts]
    spatial_effect_long = repeat(spatial_effect, n_time)
    temporal_effect = sin.(time_idx)
    y_sim  = spatial_effect_long + temporal_effect + randn(n_total) * 0.5
    y_binary = y_sim  .> (mean(y_sim) + 0.5)
    return (pts=pts, y_sim=y_sim, y_binary=y_binary, time_idx=time_idx,
            weights=weights, trials=trials, cov_indices=cov_indices)
end

# --- Intensity Estimation Helpers ---

function estimate_intensity_kde_optimized(pts; grid_res=50)
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    x_grid = range(0, 10, length=grid_res)
    y_grid = range(0, 10, length=grid_res)
    bw = 0.9 * min(std(xs), (quantile(xs, 0.75)-quantile(xs, 0.25))/1.34) * length(pts)^(-0.2)
    intensity = zeros(grid_res, grid_res)
    for (i, x) in enumerate(x_grid), (j, y) in enumerate(y_grid)
        intensity[i,j] = sum(exp.(-0.5 * (((x .- xs).^2 .+ (y .- ys).^2) ./ bw^2))) / (2π * bw^2)
    end
    return x_grid, y_grid, intensity
end

function estimate_intensity_gp(pts; grid_res=50, l=2.0)
    x_grid = range(0, 10, length=grid_res)
    y_grid = range(0, 10, length=grid_res)
    intensity = zeros(grid_res, grid_res)
    for (i, x) in enumerate(x_grid), (j, y) in enumerate(y_grid)
        p_g = (x, y)
        dists = [sqrt(sum((p_g .- p).^2)) for p in pts]
        val = sum((1 .+ sqrt(3) .* dists ./ l) .* exp.(-sqrt(3) .* dists ./ l))
        intensity[i,j] = val
    end
    return x_grid, y_grid, intensity
end

function compute_hybrid_density_cvt(pts, x_g, y_g, dens, n_seeds; iters=15)
    centroids = pts[randperm(length(pts))[1:n_seeds]]
    assignments = zeros(Int, length(pts))
    for _ in 1:iters
        for i in 1:length(pts)
            assignments[i] = argmin([sum((pts[i] .- s).^2) for s in centroids])
        end
        for j in 1:n_seeds
            idx = findall(==(j), assignments)
            if !isempty(idx)
                sub_pts = pts[idx]
                weights = [dens[argmin(abs.(x_g .- p[1])), argmin(abs.(y_g .- p[2]))] for p in sub_pts]
                sum_w = sum(weights)
                centroids[j] = (sum([p[1]*w for (p,w) in zip(sub_pts, weights)])/sum_w,
                                sum([p[2]*w for (p,w) in zip(sub_pts, weights)])/sum_w)
            end
        end
    end
    return centroids, assignments
end

function assign_spatial_units(pts, area_method, n_time;
                             dist_threshold=nothing, time_idx=nothing,
                             boundary_hull=nothing, buffer=nothing)
    n = length(pts)
    n_arealunits = max(3, floor(Int, sqrt(n)))

    # Internal Hull Computation with Dynamic Buffer
    if isnothing(boundary_hull)
        tri_full = triangulate(pts)
        hull = convex_hull(tri_full)
        hull_indices = DelaunayTriangulation.get_vertices(hull)
        raw_hull = pts[hull_indices]

        if isnothing(buffer)
            dists = Float64[]
            for i in 1:n, j in (i+1):n
                push!(dists, sqrt(sum((pts[i] .- pts[j]).^2)))
            end
            buffer = median(dists) * 0.25
        end

        # Apply buffer by expanding from hull centroid
        cx = mean(p[1] for p in raw_hull)
        cy = mean(p[2] for p in raw_hull)
        boundary_hull = map(raw_hull) do p
            dx = p[1] - cx; dy = p[2] - cy
            mag = sqrt(dx^2 + dy^2)
            (p[1] + (dx/mag)*buffer, p[2] + (dy/mag)*buffer)
        end
        push!(boundary_hull, boundary_hull[1]) # Close loop
    end

    if area_method == :triangulation
        centroids = pts[randperm(n)[1:n_arealunits]]
        assignments = [argmin([sum((p .- s).^2) for s in centroids]) for p in pts]
    elseif area_method == :cvt || area_method == :granular_cvt || area_method == :poisson_cvt || area_method == :gp_cvt
        if area_method == :cvt || area_method == :granular_cvt
            centroids, assignments = compute_hybrid_density_cvt(pts, 1:10, 1:10, ones(10,10), n_arealunits)
        elseif area_method == :poisson_cvt
            x_g, y_g, dens = estimate_intensity_kde_optimized(pts)
            centroids, assignments = compute_hybrid_density_cvt(pts, x_g, y_g, dens, n_arealunits)
        elseif area_method == :gp_cvt
            x_g, y_g, dens = estimate_intensity_gp(pts)
            centroids, assignments = compute_hybrid_density_cvt(pts, x_g, y_g, dens, n_arealunits)
        end
    else
        error("Method $area_method not supported")
    end

    area_idx = repeat(assignments, n_time)
    W = spzeros(n_arealunits, n_arealunits)
    tri = triangulate(centroids)
    for edge in each_edge(tri)
        u, v = edge
        if u > 0 && v > 0 && u <= n_arealunits && v <= n_arealunits
            W[u, v] = 1.0; W[v, u] = 1.0
        end
    end

    return centroids, assignments, Symmetric(W), area_idx, boundary_hull
end

function plot_spatial_graph(pts, assignments, centroids, W; title="Spatial Partition", show_boundaries=true, boundary_hull=nothing)
    p = scatter([pt[1] for pt in pts], [pt[2] for pt in pts],
                marker_z=assignments, color=:viridis, alpha=0.4,
                label="Data Points", title=title, aspect_ratio=:equal, markersize=3)

    if show_boundaries && !isnothing(boundary_hull)
        min_x = minimum(pt[1] for pt in boundary_hull)
        max_x = maximum(pt[1] for pt in boundary_hull)
        min_y = minimum(pt[2] for pt in boundary_hull)
        max_y = maximum(pt[2] for pt in boundary_hull)
        bbox = (min_x, max_x, min_y, max_y)

        tri = triangulate(centroids)
        vorn = voronoi(tri)
        plot!(p, [c[1] for c in boundary_hull], [c[2] for c in boundary_hull],
              color=:blue, lw=2, label="Global Boundary", fillalpha=0.1, fillrange=0)

        for i in each_polygon_index(vorn)
            coords = get_polygon_coordinates(vorn, i, bbox)
            plot!(p, [c[1] for c in coords], [c[2] for c in coords],
                  color=:black, alpha=0.3, label="", lw=0.8)
        end
    end

    n_c = length(centroids)
    for i in 1:n_c, j in (i+1):n_c
        if W[i,j] > 0
            plot!(p, [centroids[i][1], centroids[j][1]], [centroids[i][2], centroids[j][2]],
                  color=:red, alpha=0.6, lw=1.5, label="")
        end
    end
    scatter!(p, [c[1] for c in centroids], [c[2] for c in centroids],
             marker=:star, color=:red, markersize=8, label="Centroids")
    return p
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
    D = spzeros(n - 2, n)
    for i in 1:(n - 2)
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    end
    return D' * D
end

function build_ar1_precision(n, rho, tau)
    T = promote_type(typeof(rho), typeof(tau))
    diag_vals = [one(T); fill(one(T) + rho^2, n - 2); one(T)]
    off_diag = fill(-rho, n - 1)
    Q = spdiagm(0 => diag_vals, 1 => off_diag, -1 => off_diag)
    return (tau / (one(T) - rho^2)) * Q
end

function build_cyclic_ar1_precision(n, rho, tau)
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
    Q_stable = Matrix(Q) + I * 1e-5
    F = cholesky(Symmetric(Q_stable))
    return 0.5 * (logdet(F) - dot(x, Q, x) - length(x) * log(2 * pi))
end


function plot_model_fit(chain, y_sim, time_idx, area_idx)
    # Extract latent parameter means
    n_areas = maximum(area_idx)
    n_time = maximum(time_idx)

    # Convert entire chain to a matrix for easier indexing
    vals = chain.value
    p_names = names(chain, :parameters)

    println("Detected Parameter Names (first 10): ", p_names[1:min(10, end)])

    # Helper to get index safely
    get_idx(name) = findfirst(==(Symbol(name)), p_names)

    # Helper to get mean/std of a parameter by name with a safety fallback
    function get_p_val(name, func)
        idx = get_idx(name)
        if idx === nothing
            # Try alternative naming common in some MCMC versions (e.g. dot instead of brackets)
            alt_name = replace(name, "[" => ".", "]" => "")
            idx = get_idx(alt_name)
        end
        return idx !== nothing ? func(vals[:, idx, :]) : 0.0
    end

    u_icar_mean = [get_p_val("u_icar[" * string(i) * "]", mean) for i in 1:n_areas]
    u_iid_mean = [get_p_val("u_iid[" * string(i) * "]", mean) for i in 1:n_areas]

    phi_sp_mean = get_p_val("phi_sp", mean)
    sigma_sp_mean = get_p_val("sigma_sp", mean)

    s_mean = sigma_sp_mean .* (sqrt(phi_sp_mean) .* u_icar_mean .+ sqrt(1 - phi_sp_mean) .* u_iid_mean)

    # Extract f_time
    t_mean = [get_p_val("f_time[" * string(i) * "]", mean) for i in 1:n_time]
    t_std = [get_p_val("f_time[" * string(i) * "]", std) for i in 1:n_time]

    # Reconstruct the linear predictor
    y_hat = t_mean[time_idx] .+ s_mean[area_idx]

    p1 = scatter(y_sim, y_hat, alpha=0.3, label="Obs vs Pred",
                 xlabel="Simulated Y", ylabel="Predicted Y", title="Global Fit Overview")
    plot!(line=(0,1), color=:red, label="Identity")

    # Plot temporal trend
    p2 = plot(t_mean, ribbon=t_std, label="Post. Mean f_time",
              xlabel="Time", title="Temporal Component Recovery")

    display(plot(p1, p2, layout=(1,2), size=(900, 400)))
    return y_hat
end

# New function to plot spatial effects for a given time slice
function plot_spatial_time_slice(chain, pts, time_slice, area_idx, time_idx; title_prefix="Spatial Effect")
    # Extract latent parameter means from the chain
    n_areas = maximum(area_idx)
    p_names = names(chain, :parameters)
    vals = chain.value

    # Helper to get mean of a parameter by name
    function get_p_val_mean(name)
        idx = findfirst(==(Symbol(name)), p_names)
        if idx === nothing
            alt_name = replace(name, "[" => ".", "]" => "")
            idx = findfirst(==(Symbol(alt_name)), p_names)
        end
        return idx !== nothing ? mean(vals[:, idx, :]) : 0.0
    end

    u_icar_mean = [get_p_val_mean("u_icar[" * string(i) * "]") for i in 1:n_areas]
    u_iid_mean = [get_p_val_mean("u_iid[" * string(i) * "]") for i in 1:n_areas]
    phi_sp_mean = get_p_val_mean("phi_sp")
    sigma_sp_mean = get_p_val_mean("sigma_sp")

    # Calculate the mean spatial effect for each areal unit
    s_mean = sigma_sp_mean .* (sqrt(phi_sp_mean) .* u_icar_mean .+ sqrt(1 - phi_sp_mean) .* u_iid_mean)

    # Retrieve global variables for centroids, area_assignments, W_sym (from FoV6J_mGrv8Q)
    global centroids, area_assignments, W_sym

    # Create a grid for interpolation
    grid_res = 100 # Resolution of the grid
    x_grid = range(0, 10, length=grid_res)
    y_grid = range(0, 10, length=grid_res)
    interpolated_s = zeros(grid_res, grid_res)

    # Simple nearest-neighbor interpolation from centroids to grid
    for i in 1:grid_res, j in 1:grid_res
        gp = (x_grid[i], y_grid[j])
        closest_centroid_idx = argmin([sum((gp .- c).^2) for c in centroids])
        interpolated_s[i, j] = s_mean[closest_centroid_idx]
    end

    # Create the heatmap/isopleth plot
    p = heatmap(x_grid, y_grid, interpolated_s', # Transpose for correct orientation
                color = :viridis,
                title = "$(title_prefix) (Isopleth, Time Slice: $(time_slice))",
                xlabel = "X-coordinate",
                ylabel = "Y-coordinate",
                aspect_ratio = :equal,
                colorbar_title = "Spatial Effect Value")

    # Overlay centroids
    scatter!(p, [c[1] for c in centroids], [c[2] for c in centroids],
             marker = :star,
             marker_z = s_mean, # Color centroids by their spatial effect
             color = :red, markersize = 8, label = "Centroids",
             colorbar_title = "") # Prevent double colorbar title

    # Overlay polygon boundaries
    tri = triangulate(centroids)
    vorn = voronoi(tri)
    for i in each_polygon_index(vorn)
        # Ensure the coordinates are within the plot limits (0,10) for x and y
        coords = get_polygon_coordinates(vorn, i, (0.0, 10.0, 0.0, 10.0))
        plot!(p, [c[1] for c in coords], [c[2] for c in coords],
              color=:black, alpha=0.5, label="", lw=1.0)
    end

    # display(p)
    return p
end




# -------------


@model function model_v1_carstm_basic(y, time_idx, area_idx, cov_matrix_indices, adj_matrix)
    N_obs, N_areas, N_time, N_groups = length(y), size(adj_matrix, 1), maximum(time_idx), 13

    # 1. Spatial and RW2 structures
    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))
    Q_rw2 = scale_precision!(build_rw2_precision(N_groups))

    # 2. Priors (Surrogates for PC-priors)
    λ = -log(0.01) / 1.0
    σ_y ~ Exponential(1/λ)

    σ_ar1_t ~ Exponential(1/λ)
    ρ_ar1_t ~ Uniform(0, 1)

    σ_cyc ~ Exponential(1/λ)
    ρ_cyc ~ Uniform(0, 1)

    σ_rw2 ~ filldist(Exponential(1/λ), 4)

    σ_sp ~ Exponential(1/λ)
    ϕ_sp ~ Beta(1, 1)

    σ_st ~ Exponential(1/λ)
    ϕ_st ~ Beta(1, 1)
    ρ_st ~ Uniform(0, 1)

    # 3. Latent Effects
    f_time ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_time, build_ar1_precision(N_time, ρ_ar1_t, 1/σ_ar1_t^2))

    f_cyc ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_cyc, build_cyclic_ar1_precision(N_time, ρ_cyc, 1/σ_cyc^2))

    β = [Vector{Real}(undef, N_groups) for _ in 1:4]
    for k in 1:4
        β[k] ~ MvNormal(zeros(N_groups), I)
        Turing.@addlogprob! logpdf_gmrf(β[k], (1/σ_rw2[k]^2) * Q_rw2)
    end

    # BYM2 Spatial
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)

    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = σ_sp .* (sqrt(ϕ_sp) .* u_icar .+ sqrt(1 - ϕ_sp) .* u_iid)

    # Spatio-Temporal Interaction
    st_ic = Vector{Vector{Real}}(undef, N_time)
    st_ii = Vector{Vector{Real}}(undef, N_time)
    st_effs = Vector{Vector{Real}}(undef, N_time)

    for t in 1:N_time
        st_ic[t] ~ MvNormal(zeros(N_areas), I)
        Turing.@addlogprob! logpdf_gmrf(st_ic[t], Q_spatial)
        st_ii[t] ~ MvNormal(zeros(N_areas), I)
        st_effs[t] = σ_st .* (sqrt(ϕ_st) .* st_ic[t] .+ sqrt(1 - ϕ_st) .* st_ii[t])
        if t > 1
            st_effs[t] = st_effs[t] .+ (ρ_st .* st_effs[t-1])
        end
    end

    # 4. Predictor
    mu = f_time[time_idx] .+ f_cyc[time_idx] .+ s_eff[area_idx]

    for k in 1:4
        mu = mu .+ β[k][cov_matrix_indices[:, k]]
    end

    for i in 1:N_obs
        mu[i] += st_effs[time_idx[i]][area_idx[i]]
    end

    y ~ MvNormal(mu, σ_y^2 * I)
end


# --------------



# Helper to generate RFF basis for a Squared Exponential (Trend) kernel
function get_rff_trend_basis(t, m, lengthscale)
    # t: time vector, m: number of features
    N = length(t)
    Random.seed!(123)
    ω = randn(m) ./ lengthscale
    φ = rand(m) .* 2π

    # Projection matrix (N x m)
    Z = sqrt(2/m) .* cos.(t * ω' .+ φ')
    return Z
end

# Helper to generate RFF basis for a Periodic (Seasonal) kernel
function get_rff_seasonal_basis(t, m, period, lengthscale)
    N = length(t)
    Random.seed!(456)
    # For periodic kernels, we can use a simpler harmonic set or sample
    # frequencies concentrated around 2π/period
    ω = (2π / period) .* (1:m)

    Z_cos = cos.(t * ω')
    Z_sin = sin.(t * ω')

    return hcat(Z_cos, Z_sin) ./ sqrt(m)
end

@model function model_v2_carstm_rff(y, time_idx, area_idx, cov_matrix_indices, adj_matrix; m_trend=10, m_seas=5)
    N_obs = length(y)
    N_areas = size(adj_matrix, 1)
    N_time = maximum(time_idx)
    N_groups = 13

    # --- 1. Precompute Basis Matrices ---
    # We use a normalized time vector [0, 1]
    t_vec = collect(1:N_time) ./ N_time
    Z_trend = get_rff_trend_basis(t_vec, m_trend, 0.5)
    Z_seas = get_rff_seasonal_basis(t_vec, m_seas, 1.0/N_time, 0.1)

    # --- 2. Spatial & RW2 Structures ---
    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))
    Q_rw2 = scale_precision!(build_rw2_precision(N_groups))

    # --- 3. Priors ---
    λ = -log(0.01) / 1.0
    σ_y ~ Exponential(1/λ)

    # RFF Coefficients weights
    w_trend ~ MvNormal(zeros(m_trend), I)
    w_seas ~ MvNormal(zeros(size(Z_seas, 2)), I)

    σ_trend ~ Exponential(1/λ)
    σ_seas ~ Exponential(1/λ)

    # RW2 Covariates
    σ_rw2 ~ filldist(Exponential(1/λ), 4)
    β = [Vector{Real}(undef, N_groups) for _ in 1:4]
    for k in 1:4
        β[k] ~ MvNormal(zeros(N_groups), I)
        Turing.@addlogprob! logpdf_gmrf(β[k], (1/σ_rw2[k]^2) * Q_rw2)
    end

    # BYM2 Spatial
    σ_sp ~ Exponential(1/λ)
    ϕ_sp ~ Beta(1, 1)
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = σ_sp .* (sqrt(ϕ_sp) .* u_icar .+ sqrt(1 - ϕ_sp) .* u_iid)

    # --- 4. Predictor Assembly ---
    f_trend = Z_trend * w_trend .* σ_trend
    f_seas = Z_seas * w_seas .* σ_seas

    mu = f_trend[time_idx] .+ f_seas[time_idx] .+ s_eff[area_idx]

    for k in 1:4
        mu = mu .+ β[k][cov_matrix_indices[:, k]]
    end

    y ~ MvNormal(mu, σ_y^2 * I)
end



# -----------------


@model function model_v3_binomial(y, trials, time_idx, area_idx, cov_matrix_indices, adj_matrix, class1_idx, class2_idx)
    N_obs = length(y)
    N_areas = size(adj_matrix, 1)
    N_time = maximum(time_idx)
    N_groups = 13

    # 1. Spatial and RW2 structures
    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))
    Q_rw2 = scale_precision!(build_rw2_precision(N_groups))

    # 2. Priors
    λ = -log(0.01) / 1.0

    # Fixed Effects
    β_class1 ~ MvNormal(zeros(13), 10.0 * I)
    β_class2 ~ MvNormal(zeros(2), 10.0 * I)

    σ_ar1_t ~ Exponential(1/λ)
    ρ_ar1_t ~ Uniform(0, 1)

    σ_cyc ~ Exponential(1/λ)
    ρ_cyc ~ Uniform(0, 1)

    σ_rw2 ~ filldist(Exponential(1/λ), 4)

    σ_sp ~ Exponential(1/λ)
    ϕ_sp ~ Beta(1, 1)

    # 3. Latent Effects
    f_time ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_time, build_ar1_precision(N_time, ρ_ar1_t, 1/σ_ar1_t^2))

    f_cyc ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_cyc, build_cyclic_ar1_precision(N_time, ρ_cyc, 1/σ_cyc^2))

    β_rw = [Vector{Real}(undef, N_groups) for _ in 1:4]
    for k in 1:4
        β_rw[k] ~ MvNormal(zeros(N_groups), I)
        Turing.@addlogprob! logpdf_gmrf(β_rw[k], (1/σ_rw2[k]^2) * Q_rw2)
    end

    # BYM2 Spatial
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = σ_sp .* (sqrt(ϕ_sp) .* u_icar .+ sqrt(1 - ϕ_sp) .* u_iid)

    # 4. Predictor Assembly
    mu = f_time[time_idx] .+ f_cyc[time_idx] .+ s_eff[area_idx] .+
         β_class1[class1_idx] .+ β_class2[class2_idx]

    for k in 1:4
        mu = mu .+ β_rw[k][cov_matrix_indices[:, k]]
    end

    # Refactored likelihood to avoid tilde_observe!! error
    for i in 1:N_obs
        y[i] ~ BinomialLogit(trials[i], mu[i])
    end
end


# -----------------



@model function model_v4_weighted_binary(y, trials, weights, time_idx, area_idx, cov_matrix_indices, adj_matrix, class1_idx, class2_idx)
    N_obs = length(y)
    N_areas = size(adj_matrix, 1)
    N_time = maximum(time_idx)
    N_groups = 13

    # 1. Spatial and RW2 structures
    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))
    Q_rw2 = scale_precision!(build_rw2_precision(N_groups))

    # 2. Priors
    λ = -log(0.01) / 1.0
    beta_class1 ~ MvNormal(zeros(13), 10.0 * I)
    beta_class2 ~ MvNormal(zeros(2), 10.0 * I)
    sigma_ar1_t ~ Exponential(1/λ)
    rho_ar1_t ~ Uniform(0, 1)
    sigma_cyc ~ Exponential(1/λ)
    rho_cyc ~ Uniform(0, 1)
    sigma_rw2 ~ filldist(Exponential(1/λ), 4)
    sigma_sp ~ Exponential(1/λ)
    phi_sp ~ Beta(1, 1)

    # 3. Latent Effects
    f_time ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_time, build_ar1_precision(N_time, rho_ar1_t, 1/sigma_ar1_t^2))

    f_cyc ~ MvNormal(zeros(N_time), I)
    Turing.@addlogprob! logpdf_gmrf(f_cyc, build_cyclic_ar1_precision(N_time, rho_cyc, 1/sigma_cyc^2))

    beta_rw = [Vector{Real}(undef, N_groups) for _ in 1:4]
    for k in 1:4
        beta_rw[k] ~ MvNormal(zeros(N_groups), I)
        Turing.@addlogprob! logpdf_gmrf(beta_rw[k], (1/sigma_rw2[k]^2) * Q_rw2)
    end

    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)
    u_iid ~ MvNormal(zeros(N_areas), I)
    s_eff = sigma_sp .* (sqrt(phi_sp) .* u_icar .+ sqrt(1 - phi_sp) .* u_iid)

    # 4. Predictor
    mu = f_time[time_idx] .+ f_cyc[time_idx] .+ s_eff[area_idx] .+
         beta_class1[class1_idx] .+ beta_class2[class2_idx]

    for k in 1:4
        mu = mu .+ beta_rw[k][cov_matrix_indices[:, k]]
    end

    # 5. Weighted Likelihood
    for i in 1:N_obs
        dist = BinomialLogit(trials[i], mu[i])
        Turing.@addlogprob! weights[i] * logpdf(dist, y[i])
    end
end



# -----------


# Demonstration of the two methods
@model function weight_comparison(y, n, weights)
    μ ~ Normal(0, 1)

    # Method A: Log-Likelihood Weighting (V4/V5 Style)
    # Correct for Importance/Survey weights
    for i in 1:length(y)
        Turing.@addlogprob! weights[i] * logpdf(BinomialLogit(n[i], μ), y[i])
    end

    # Method B: Trial Weighting (Alternative)
    # This is essentially changing the sample size
    # for i in 1:length(y)
    #     y[i] ~ BinomialLogit(round(Int, n[i] * weights[i]), μ)
    # end
end



# Helper to create an RFF mapping for multi-dimensional input
function get_deep_rff_basis(X, m, lengthscale)
    # X is (N_obs x D) matrix
    N, D = size(X)
    Random.seed!(789)

    # Spectral frequencies for RBF kernel
    Ω = randn(m, D) ./ lengthscale
    Φ = rand(m) .* 2π

    # Feature map: sqrt(2/m) * cos(X*Ω' + Φ)
    Z = sqrt(2/m) .* cos.(X * Ω' .+ Φ')
    return Z
end

@model function model_v5_deep_gp_carstm(y, trials, weights, time_idx, area_idx, pts, cov_indices, adj_matrix, class1_idx, class2_idx)
    N_obs = length(y)
    N_areas = size(adj_matrix, 1)
    N_time = maximum(time_idx)

    # Hyperparameters
    λ = -log(0.01) / 1.0

    # categorical Fixed Effects (similar to V3/V4)
    beta_class1 ~ MvNormal(zeros(13), 10.0 * I)
    beta_class2 ~ MvNormal(zeros(2), 10.0 * I)

    # --- Deep GP Layer 1: Latent Manifold ---
    t_norm = time_idx ./ N_time
    s_coords = reduce(hcat, [collect(pts[i]) for i in area_idx])'
    X_input = hcat(t_norm, s_coords)

    m_layer1 = 10
    Z1 = get_deep_rff_basis(X_input, m_layer1, 0.5)
    w1 ~ MvNormal(zeros(m_layer1), I)
    hidden_latent = Z1 * w1

    # --- Deep GP Layer 2: Non-linear Predictor ---
    m_layer2 = 15
    Random.seed!(101)
    Ω2 = randn(m_layer2, 1)
    Φ2 = rand(m_layer2) * 2π

    w2 ~ MvNormal(zeros(m_layer2), I)
    σ_deep ~ Exponential(1/λ)

    f_deep = σ_deep .* (sqrt(2/m_layer2) .* cos.(hidden_latent .* Ω2' .+ Φ2')) * w2

    # --- Additive Structural Components ---
    Q_spatial = scale_precision!(build_laplacian_precision(adj_matrix))
    σ_sp ~ Exponential(1/λ)
    u_icar ~ MvNormal(zeros(N_areas), I)
    Turing.@addlogprob! logpdf_gmrf(u_icar, Q_spatial)

    # --- Predictor Assembly ---
    # Combine Deep GP, ICAR spatial, and Fixed Effects
    mu = f_deep .+ (σ_sp .* u_icar[area_idx]) .+ beta_class1[class1_idx] .+ beta_class2[class2_idx]

    # Weighted Likelihood
    for i in 1:N_obs
        dist = BinomialLogit(trials[i], mu[i])
        Turing.@addlogprob! weights[i] * logpdf(dist, y[i])
    end
end


function calculate_waic(model, samples)
    # 1. Standardize input to a Matrix [Parameters x Iterations]
    sample_matrix = if samples isa MCMCChains.Chains
        Array(samples)'
    else
        samples
    end
    
    n_params, n_iters = size(sample_matrix)
    
    # 2. Robust observation extraction
    # Check for different possible names in the model arguments
    y_data = if haskey(model.args, :y)
        model.args.y
    elseif haskey(model.args, :y_sim)
        model.args.y_sim
    elseif haskey(model.args, :y_binary)
        model.args.y_binary
    else
        # Fallback to the first argument if keys don't match
        model.args[1]
    end
    n_obs = length(y_data)
    
    # 3. Compute Pointwise Log-Likelihood Matrix [Iterations x Observations]
    ll_matrix = zeros(n_iters, n_obs)
    
    for i in 1:n_iters
        theta = sample_matrix[:, i]
        try
            # Distribute total log-likelihood across observations as a proxy
            l = Turing.loglikelihood(model, theta)
            ll_matrix[i, :] .= l / n_obs
        catch
            ll_matrix[i, :] .= -1e10 # Use a large negative number for failure
        end
    end
    ll_matrix[.!isfinite.(ll_matrix)] .= -1e10

    # 4. Numerically Stable WAIC Implementation
    # lppd = sum_{i=1}^{N_obs} log( (1/S) * sum_{s=1}^S exp(log_lik_{s,i}) )
    # We use the log-sum-exp trick to avoid overflow/Inf
    lppd = 0.0
    for j in 1:n_obs
        col = ll_matrix[:, j]
        max_ll = maximum(col)
        # log(mean(exp(x))) = max_x + log(mean(exp(x - max_x)))
        lppd += max_ll + log(mean(exp.(col .- max_ll)))
    end
    
    # p_waic = sum_{i=1}^{N_obs} var_{s=1}^S (log_lik_{s,i})
    p_waic = sum(var(ll_matrix, dims=1))
    
    waic_val = -2 * (lppd - p_waic)
    
    return waic_val
end


# Helper to calculate RMSE
function calculate_rmse(y_true, y_pred)
    return sqrt(mean((y_true .- y_pred).^2))
end





