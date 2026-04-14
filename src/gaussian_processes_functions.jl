
# Squared-exponential covariance function
sqexp_cov_fn(D, phi, eps=1e-3) = exp.(-D^2 / phi) + LinearAlgebra.I * eps

# Exponential covariance function
exp_cov_fn(D, phi) = exp.(-D / phi)



# generic kernel functions 

# lenscale = -1 / log(ρ)
# σ_ar1^2 / (1 - ρ^2) = marginal variance
# kernel_ar1(σ, ρ) = σ^2 * with_lengthscale(Matern12Kernel(), -1/log(ρ)) 
# the softplus should not be necessary ... 

kernel_ar1(σ, ρ) = σ^2 / softplus(1 - ρ^2) * with_lengthscale(Matern12Kernel(), softplus(-1 / log(ρ)) )

# RW2 is equivalent to a Spline kernel or an Integrated Wiener Process
# For simplicity, we often use a high-order Matern or a custom structure

kernel_rw2(σ) = σ^2 * Matern52Kernel() # Matern32 is a common smooth approximation for RW2


Turing.@model function gaussian_process_basic(; Y, D, cov_fn=exp_cov_fn, nData=length(Y) )
    mu ~ Normal(0.0, 1.0); # mean process
    sig2 ~ LogNormal(0, 1) # "nugget" variance
    phi ~ LogNormal(0, 1) # phi is ~ lengthscale along Xstar (range parameter)
    # sigma = cov_fn(D, phi) + sig2 * LinearAlgebra.I(nData) # Realized covariance function + nugget variance
    vcov = cov_fn(D, phi) + sig2 .* LinearAlgebra.I(nData ) .+ eps() * 1000.0
    Y ~ MvNormal(mu * ones(nData), Symmetric(vcov) )     # likelihood
end


Turing.@model function gaussian_process_covars(; Y, X, D, cov_fn=exp_cov_fn, nData=length(Y), nF=size(X,2) )
    # model matrix for fixed effects (X)
    beta ~ filldist( Normal(0.0, 1.0), nF); 
    sig2 ~ LogNormal(0, 1) # "nugget" variance
    phi ~ LogNormal(0, 1) # phi is ~ lengthscale along Xstar (range parameter)
    # sigma = cov_fn(D, phi) + sig2 * LinearAlgebra.I(nData) # Realized covariance function + nugget variance
    mu = X * beta # mean process
    vcov = cov_fn(D, phi) + sig2 .* LinearAlgebra.I(nData ) .+ eps() .* 1000.0
    Y ~ MvNormal(mu, Symmetric(vcov) )     # likelihood
end



Turing.@model function gaussian_process_ar1( ::Type{T}=Float64; Y, X, D, ar1, cov_fn=exp_cov_fn, 
    nData=length(Y), nF=size(X,2), nT=maximum(ar1)-minimum(ar1)+1 ) where {T} 
 
    rho ~ truncated(Normal(0,1), -1, 1)
    ar1_process_error ~ LogNormal(0, 1) 
    var_ar1 =  ar1_process_error^2 / (1 - rho^2)
    vcv = ar1_covariance_local(nT, rho, var_ar1, T)  # -- covariance by time
    ymean_ar1 ~ MvNormal(Symmetric(vcv) );  # -- means by time 
     
    # # mean process model matrix  
    beta ~ filldist( Normal(0.0, 1.0), nF); 
 
    sig2 ~ LogNormal(0, 1) # "nugget" variance
    phi ~ LogNormal(0, 1) # phi is ~ lengthscale along Xstar (range parameter)
    # sigma = cov_fn(D, phi) + sig2 * LinearAlgebra.I(nData) # Realized covariance function + nugget variance
    # vcov = cov_fn(D, phi) + sig2 .* LinearAlgebra.I(nData )
    
    Y ~ MvNormal( X * beta .+ ymean_ar1[ar1[1:nData]], Symmetric(cov_fn(D, phi) .+ sig2 * I(nData) .+ eps() ) )     # likelihood
end

 


function gp_predictions(; Y, D, mu, sig2, phi, cov_fn=exp_cov_fn, nN=length(Xnew), nP=size(res, 1) ) 
    ynew = Vector{Float64}()
    # Threads.@threads -- to add 
    for i in sample(1:size(res,1), nP, replace=true)
        K = cov_fn(D, phi[i])
        Koo_inv = inv(K[(nN+1):end, (nN+1):end])
        Knn = K[1:nN, 1:nN]
        Kno = K[1:nN, (nN+1):end]
        C = Kno * Koo_inv
        mvn = MvNormal( 
            C * (Y .- mu[i]) .+ mu[i], 
            Matrix(LinearAlgebra.Symmetric(Knn - C * Kno')) + sig2[i] * LinearAlgebra.I 
        ) 
        ynew = vcat(ynew, [rand(mvn) ] )
    end
    ynew = stack(ynew, dims=1)  # rehape to matrix   
    return ynew
end



function ar1_covariance( n, rho, var,  ::Type{T}=Float64 )  where {T} 
    vcv = zeros( T, n, n) .+ I(n) 
    Threads.@threads for r in 1:n
    for c in 1:n
        if r >= c 
            vcv[r,c] = var * rho^(r-c) 
        end
    end
    end
    return vcv
end



function ar1_covariance_local( n, rho, var,  ::Type{T}=Float64 )  where {T} 
    vcv = zeros( T, n, n) .+ I(n) 
    Threads.@threads for r in 1:n
    for c in 1:n
        d = r-c
        if d == 0 | d == 1
            vcv[r,c] = var * rho^d  
        end
    end
    end
    return vcv
end


Turing.@model function ar1_gp(  ::Type{T}=Float64; Y, ar1,  nData=length(Y), nT=Integer(maximum(ar1)-minimum(ar1)+1) ) where {T} 
    Ymean = mean(Y)
    rho ~ truncated(Normal(0,1), -1, 1)
    ar1_process_error ~ LogNormal(0, 1) 
    var_ar1 =  ar1_process_error^2 / (1 - rho^2)
    vcv = ar1_covariance(nT, rho, var_ar1, T)  # -- covariance by time
    ymean_ar1 ~ MvNormal(Symmetric(vcv) );  # -- means by time 
    observation_error ~ LogNormal(0, 1) 
    Y ~ MvNormal( ymean_ar1[ar1[1:nData]] .+ Ymean, observation_error )     # likelihood
end
 


Turing.@model function ar1_gp_local(  ::Type{T}=Float64; Y, ar1,  nData=length(Y), nT=Integer(maximum(ar1)-minimum(ar1)+1) ) where {T} 
    Ymean = mean(Y)
    rho ~ truncated(Normal(0,1), -1, 1)
    ar1_process_error ~ LogNormal(0, 1) 
    var_ar1 =  ar1_process_error^2 / (1 - rho^2)
    vcv = ar1_covariance_local(nT, rho, var_ar1, T)  # -- covariance by time
    ymean_ar1 ~ MvNormal(Symmetric(vcv) );  # -- means by time 
    observation_error ~ LogNormal(0, 1) 
    Y ~ MvNormal( ymean_ar1[ar1[1:nData]] .+ Ymean, observation_error )     # likelihood
end
 

Turing.@model function ar1_recursive(; Y, ar1,  nData=length(Y), nT=Integer(maximum(ar1)-minimum(ar1)+1) )
    Ymean = mean(Y)
    alpha_ar1 ~ Normal(0,1)
    rho ~ truncated(Normal(0,1), -1, 1)
    ar1_process_error ~ LogNormal(0, 1) 
    ymean_ar1 = tzeros(nT);  # -- means by time 
    ymean_ar1[1] ~ Normal(Ymean, ar1_process_error) 
    for t in 2:nT
        ymean_ar1[t] ~ Normal(alpha_ar1 + rho * ymean_ar1[t-1], ar1_process_error );
    end
    observation_error ~ LogNormal(0, 1) 
    Y ~ MvNormal( ymean_ar1[ar1[1:nData]] .+ Ymean, observation_error )     # likelihood
end
 


Turing.@model function SparseFinite_example( Yobs, Xobs, Xinducing )
    nInducing = length(Xinducing)
 
    # m ~ filldist( Normal(0, 100), nInducing ) 
    # A ~ filldist( Normal(), nInducing, nInducing ) 
    # S = PDMat(Cholesky(LowerTriangular(A)))
 
    kernel_var ~ Gamma(0.5, 1.0)
    kernel_scale ~ Gamma(2.0, 1.0)
    lambda = 0.001
    
    fkernel = kernel_var * Matern52Kernel() ∘ ScaleTransform(kernel_scale) # ∘ ARDTransform(α)
         
    fgp = atomic(GP(fkernel), GPC())
    fobs = fgp( Xobs, lambda )
    finducing = fgp( Xinducing, lambda ) 
    fsparse = SparseFiniteGP(fobs, finducing)
    Turing.@addlogprob! -Stheno.elbo( fsparse, Yobs ) 
    # GPpost = posterior(fsparse, Yobs)
 
    # m ~ GPpost(Xinducing, lambda) 
  
    # Yobs ~ GPpost(Xobs, lambda)
      
end



Turing.@model function SparseVariationalApproximation_example( Yobs, Xobs, Xinducing, lambda = 0.001)
    nInducing = length(Xinducing)

    m ~ filldist( Normal(0, 100), nInducing ) 
  
    # variance process
    # Efficiently constructs S as A*Aᵀ
    A ~ filldist( Normal(), nInducing, nInducing ) 
    S = PDMat(Cholesky(LowerTriangular(A)))
 
    kernel_var ~ Gamma(0.5, 1.0)
    kernel_scale ~ Gamma(2.0, 1.0)
    
    fkernel = kernel_var * Matern52Kernel() ∘ ScaleTransform(kernel_scale) # ∘ ARDTransform(α)

    fgp = atomic( GP(fkernel), GPC()) 

    finducing = fgp(Xinducing, lambda) # aka "prior" in AbstractGPs
    fsparse = SparseVariationalApproximation(finducing, MvNormal(m, S))
    
    # Turing.@addlogprob! -Stheno.elbo(fsparse, fobs, Yobs )  # failing here, 

    fposterior = posterior(fsparse, finducing, Yobs)
    
    o = fposterior(Xobs)
    Yobs ~ MvNormal( mean(o), Symmetric(cov(o)) + I*lambda )

end



function variational_inference_solution(m; max_iters=100, nsamps=max_iters,  nelbo=3 )

    # Fit via ADVI. minor speed benefit vs NUTS
    _, indices = Bijectors.bijector(m, Val(true));
    vars = keys(indices)

    q0 = Variational.meanfield(m)     # initialize variational distribution (optional)
    advi = ADVI(nelbo, max_iters)    # num_elbo_samples, max_iters
    msol = Turing.vi(m, advi, q0) #, optimizer=Flux.ADAM(1e-1));
    msamples = DataFrame( rand(msol, nsamps )', :auto ) 

    # vectorize variable names ... needs more conditions if 2-D or higher ..
    vns = []
    for (i, sym) in enumerate(vars) 
        j = union(indices[sym]...)  # <= array of ranges
        nj = sum(length.(j)) 
        if  nj > 1
            offset = 1
            for r in j
                push!(vns, "$(sym)[$offset]")
                offset += 1
            end
        else
            push!(vns, "$(sym)") 
        end
    end
    
    vns = Symbol.(vns)

    msamples = rename(msamples, vns)

    mmean = combine( msamples, [ n => (x -> mean(x)) => n for n in names(msamples)  ] )
    mstd  = combine( msamples, [ n => (x -> std(x)) => n for n in names(msamples)  ] )

    out = (
        msol = msol,
        msamples = msamples, 
        mmean = mmean,
        mstd = mstd
    )
    
    return out
 
end


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
