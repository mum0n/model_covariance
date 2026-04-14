

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
    


@model function model_recursive(y, xi, zi, ui, nx, nz, nu)
    # input is y, xindex zindex, uindex and numbers of xi total, zi total, ui total
    σ_obs ~ Exponential(1.0)
    σ_ar1 ~ Exponential(1.0)
    ρ ~ Uniform(-1, 1) # AR1 coefficient
  
    σ_rw2 ~ Exponential(1.0)  # Random walk step size (smoothness)
  
    # AR1(x) - latent
    x = Vector{Real}(undef, nx)
    x[1] ~ Normal(0, σ_ar1 / sqrt(1 - ρ^2))
    for g in 2:nx
      x[g] ~ Normal(ρ * x[g-1], σ_ar1)
    end

    # RW2(z) - latent
    z = Vector{Real}(undef, nz)
    z[1] ~ Normal(0, 10)
    z[2] ~ Normal(0, 10)
    for g in 3:nz
      z[g] ~ Normal( 2*z[g-1] - z[g-2], σ_rw2 )
    end
  
    # Fixed Effect: Categorical 
    β_u ~ filldist(Normal(0, 10), nu) 
  
    # Likelihood
    mu = Vector{Real}(undef, length(y))
    for i in 1:length(y)
      mu[i] = x[xi[i]] + z[zi[i]] + β_u[ui[i]]
    end

    y ~ MvNormal(mu, σ_obs * I)

    return mu
end
 
