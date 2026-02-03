

function posterior_samples( ps; sym=:none )
    # rename function for convenience, # works for scalar vars too
    if sym != :none
        ps = Turing.group(ps, sym)
        ps = Array(ps)
        return ps
    end
end
    
function posterior_summary(ps; sym=:none, stat=:all, dims=:none, perm=:none) 
    if sym != :none
        ps = Turing.group(ps, sym) # works for scalar vars too
    end
    ps = summarize(ps)
    
    if stat == :all
        return ps
    end
    ps= ps[:,stat]

    if dims != :none 
        ps = reshape(ps, dims)
    end

    if perm != :none
        ps = permutedims(ps, perm)
    end

    ps = Array(ps) #redunant?
    return ps
end 


function example_data(; N=1000,  cov2lev = ("1"=>1, "2"=>1.25, "3"=>2, "4"=>1.5), alpha=0.1 )
    # make random data for analysis
    # NOTE: utility in terms of creating model matrix using schemas, etc
    
    xvar = vec(randn(N)*3.0  )
    
    df = DataFrame(
        xvar = xvar,
        covar1 = string.(rand( [1, 2, 3, 4], N)  ),  # factorial
        covar2 = vec(randn(N)),
        covar3 = vec(trunc.( Int, randn(N)*3 ) )
    )

    cov2 = replace(df.covar1, cov2lev[1], cov2lev[2], cov2lev[3], cov2lev[4] ) 
    df.yvar = sin.(vec(xvar)) + df.covar2 .* alpha .+ rand.(Normal.(cov2, 0.1))
    schm = StatsModels.schema(df, Dict(:covar1 => EffectsCoding()) )
    dm = StatsModels.apply_schema(StatsModels.term.((:xvar, :covar1, :covar2)), schm ) 
    modelcols = StatsModels.modelcols(StatsModels.MatrixTerm( dm ), df)
    coefnames = StatsModels.coefnames( dm )   # coef names of design matrix 
    termnames = StatsModels.termnames( dm)  # coef names of model data
    
    # alternative access:
    # fm = StatsModels.@formula( yvar ~ 1 + xvar + covar1 + covar2)
    # resp = response(fm, df)  # response
    # cols = modelcols(z, df)
    # o = reduce(hcat, cols)
    
    return df, modelcols, coefnames, termnames, cov2lev
end


function example_nonlinear_data(Xlatent = -7:0.1:7, Xobs = -7:0.5:7  ) 
    
    # function that describes latent process
    Y(x) = (x + 4) * (x + 1) * (x - 1) * (x - 3)  

    # Latent data ("truth")
    # Xlatent = -7:0.1:7
    Ylatent = Y.(Xlatent)

    # "Observations" with noise
    Yobs = Y.(Xobs) .+ rand(Uniform(-100, 100), size(Xobs,1))
    
    return Xlatent, Ylatent, Xobs, Yobs
end


function featurize_poly(Xin, degree=1)
    # add higher order polynomials
    return repeat(Xin, 1, degree + 1) .^ (0:degree)'
end
 

function linear_regression(X, y, Xstar)
    beta = (X' * X) \ (X' * y)
    return Xstar * beta
end;


function ridge_regression(X, y, Xstar, lambda)
    beta = (X' * X + lambda * I) \ (X' * y)
    return Xstar * beta
end


function kernel_ridge_regression(X, y, Xstar, lambda, k)
    K = kernelmatrix(k, X)
    kstar = kernelmatrix(k, Xstar, X)
    return kstar * ((K + lambda * I) \ y)
end;
 

function gp_kernel_ridge_regression_cholesky( v, s, r, Xobs, Xinducing, Yobs )
    # Gaussian Process Kernel Ridge Regression
    k = ( v * SqExponentialKernel() ) ∘ ScaleTransform(s)
    ko = kernelmatrix(k, Xobs)
    kp = kernelmatrix(k, Xinducing, Xobs)
    L = cholesky(ko + r * I)
    Yp = kp * ( L.U \ (L.L \ Yobs) )  # mean process  
    return Yp 
end


function gp_kernel_ridge_regression( v, s, r, Xobs, Xinducing, Yobs )
    # need to check algebra
    # Gaussian Process Kernel Ridge Regression
    # k = ( v * SqExponentialKernel() ) ∘ ScaleTransform(s)
    # ko = kernelmatrix(k, Xobs) + r * I  
    # kp = kernelmatrix(k, Xinducing, Xobs) 
    # Yp = rand( MvNormal(  ko\Yobs, ko ) ) # faster
    return Yp
end


function quantiles(X, q; dims, drop=false)
    Q = mapslices(x -> quantile(x, q), X, dims=dims)
    out = drop ? dropdims(Q, dims=dims) : Q
    return out
end


function summarize_samples(S; dims=1)
  smean = mean(S, dims=dims)
  slb = quantiles(S, 0.025, dims=dims)
  sub = quantiles(S, 0.975, dims=dims)
  ssd = std(S, dims=dims)
  smed = median(S, dims=dims)
  return smean, slb, sub, ssd, smed
end
    
# Squared-exponential covariance function
sqexp_cov_fn(D, phi, eps=1e-3) = exp.(-D^2 / phi) + LinearAlgebra.I * eps

# Exponential covariance function
exp_cov_fn(D, phi) = exp.(-D / phi)


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
