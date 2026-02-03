---
title: "CARSTM in Julia for snow crab"
header: "CARSTM Julia Snow crab"

keyword: |
	Keywords - Guassian Process / CAR 
abstract: |
	Compare Guassian Process / CARSTM with snow crab data.

metadata-files:
  - _metadata.yml
 
params:
  todo: [nothing,add,more,here]
---


<!-- Quarto formatted: To create/render document:

make quarto FN=snowcrab_carstm_julia.md DOCTYPE=html PARAMS="-P todo:[nothing,add,more,here]" --directory=~/projects/model_covariance/docs
 
-->

<!-- To include a common file:
{{< include _common.qmd >}}  
-->

<!-- To force landscape (eg. in presentations), surround the full document or pages with the following fencing:
::: {.landscape}
  ... document ...
:::
-->

NOTES:: car method works nicely
 
need to add method for fixed effects and AR1 .. then CARSTM should be complete


# Space-time model with snow crab

 
## Setup data and environment

First save a copy of rdata to a local directory ("outdir")

```R
# create data
yrs = 1999:2024

homedir = Sys.getenv()[["HOME"]]
scriptsdir = file.path( homedir, "projects", "model_covariance", "scripts" ) 
outdir = file.path( homedir, "projects", "model_covariance", "data", "snowcrab" ) 

source( file.path( scriptsdir, "snow_crab_survey_data.R" ) ) 

```

Prepare julia environment and import the rdata files

```julia

using DrWatson
 
# rootdir = joinpath("\\", "home", "jae" ) # windows
rootdir = joinpath("/", "home", "jae" )  # linux

project_directory = joinpath( rootdir, "projects", "model_covariance"  )

quickactivate(project_directory)

include( scriptsdir( "startup.jl" ));     # env and data

include( srcdir( "simple_linear_regression.jl") );
include( srcdir( "regression_functions.jl" ));   # support functions  
include( srcdir( "car_functions.jl"  ) );  
include( srcdir( "snowcrab_functions.jl"  ) )  ;
   
# Set a seed for reproducibility.
 
Random.seed!( Xoshiro(1234) )
 

GPmethod="GPvfe"  # default

kerneltype="squared_exponential"

M, nb, sp = snow_crab_survey_data( 2000:2010 );


# indexes for identify preds and obs
ip = findall(M.tag .== "predictions" );
io = findall(M.tag .== "observations" );
nData = length(io) 
nPred = length(ip)


# space, time labels  
tuid = M.year[io] ;
auid = M.space[io];

stuid = [join(i) for i=zip(auid, fill("-", nData), tuid)];


# adjacency_matrix
node1, node2, scaling_factor = nodes(nb);
W = nb_to_adjacency_matrix(nb) ;
D = diagm(vec( sum(W, dims=2) )) ;
nAU = length(nb)


# independent variables:  
y = floor.(Int, M.totno[io]);
pa = floor.(Int, M.pa[io]);
wt = M.meansize[io];
log_offset = log.(M.data_offset[io]);


# defined good habitat aprori == 1
good = findall(x -> x==1, pa);
# good_wt = findall(x -> x>5, y) -- no obs > 5

# minimum( y[findall(x -> x==1, pa)]  ) # detection limit == 1
# maximum( y[findall(x -> x==0, pa)]  ) # detection limit == 1

YG = log.(y) .- log_offset 
YG = YG .- mean( YG[good])  # centered

# fixed effects 
# https://repsychling.github.io/contrasts-and-formula/

X, Xschema, Xcoefnames, nX = model_matrix_fixed_effects(
  M[io,:], 
  @formula(totno ~ 1 + year ), 
  contrasts=Dict( :year => StatsModels.EffectsCoding()  )
)

# covariates to be smoothed as GP
nInducing = 13
nUnique = 100
Gvars=["z", "t", "pca1", "pca2"]

  G0, G, Gp, Gr, nG, G_means, G_sds, Gpp = get_gp_covariates( 
    M=M, 
    Gvars=Gvars, 
    nUnique = nUnique,  
    nInducing = nInducing
  )

# Base.delete_method.(methods(myfunction))

# M = nothing
# sp =nothing

# debug:

# DEBUG = Ref{Any}()

# add this inside of a function to track vars
# Main.DEBUG[] = y,p,t


```

## Model 1: CAR in space with (linear) covariates   

This is the simplest spatial form: 
  - spatial random effects (CAR) - icar form
  - year as fixed effect 
  - linear covariates 
  - GP covars - none

```julia

m = turing_car(D, W, X, log_offset, y, auid )
msol = sample(m, Turing.MH(), 10)   

```

## Model 2: CAR in space (precison) with (linear) covariates   

This uses a precision form of Model 1, for the spatial random effects
 
```julia

m = turing_car_prec(D, W, X, log_offset, y, auid )
msol = sample(m, Turing.MH(), 10)   

```

## Model 3: ICAR in space with (linear) covariates  and fixed effects

This uses an ICAR form of Model 1, for the spatial random effects. 
Testing different sampling approaches and MAP, VI approaches for viability.

Turing's samplers:

-  SMC: number of particles.
-  PG: number of particles, number of iterations.
-  HMC: leapfrog step size, leapfrog step numbers.
-  Gibbs: component sampler 1, component sampler 2, ...
-  HMCDA: total leapfrog length, target accept ratio.
-  NUTS: number of adaptation steps (optional), target accept ratio.


```julia

# simple spatial icar test .. using nodes .. year as fixed effect covariates (-- no GP covars)

m = turing_glm_icar( family="poisson", good=good,
  Y=y, X=X, log_offset=log_offset, auid=auid, nAU=nAU,
  node1=node1, node2=node2, scaling_factor=scaling_factor
) 

rand(m)  # check a sample

msol = sample(m, Turing.MH(), 10)   
  
msol = optimize(m, MLE() )  
  # Optim.Options(iterations=5_000, allow_f_increases=true): 
  # does not converge 
 
msol = optimize(m, MAP() )  
  # 6yr test:  ~ 20 sec; lp of -12885.31; 
  # full data: ~ 5 min: -29629.15
  # 0.004562825353062191, 7.221872596866684, 0.9534728688028509

# ~ 14 min; 
using Turing.Variational
res = variational_inference_solution(m, max_iters=100 )
pm = res.mmean
msol = res.msol


n_samples, n_adapts, n_chains = 1000, 1000, 4
target_acceptance, max_depth, init_ϵ = 0.65, 10, 0.001   
turing_sampler = Turing.NUTS(n_adapts, target_acceptance; max_depth=max_depth, init_ϵ=init_ϵ)

msol = sample( m, turing_sampler,  n_samples ) # to see progress
# msol = sample( m, turing_sampler, MCMCThreads(), n_samples, n_chains  ) # to see progress
 
showall( summarize(msol) )
  parameters      mean       std   naive_se      mcse         ess      rhat   ess_per_sec 

     beta[1]    7.2529    0.0610     0.0019    0.0046    193.0744    1.0009        0.0197
     beta[2]    0.0490    0.0222     0.0007    0.0009    797.9459    1.0013        0.0816
     beta[3]   -0.1927    0.0199     0.0006    0.0006    976.6058    1.0003        0.0998
     beta[4]   -0.1364    0.0197     0.0006    0.0008    896.5258    0.9991        0.0916
    theta[1]    0.0682    1.0456     0.0331    0.0351   1004.3178    1.0014        0.1026
     sum_phi    0.0406    0.7393     0.0234    0.0252   1039.1352    1.0014        0.1062
       sigma    0.7490    0.0408     0.0013    0.0030    155.5465    1.0116        0.0159
         rho    0.9552    0.0583     0.0018    0.0048    141.4854    1.0058        0.0145

```


## Model 4: GP only (no space, no fixed effects)


SEE: https://github.com/STOR-i/GaussianProcesses.jl/blob/master/notebooks/Regression.ipynb eqs 2-4

https://betanalpha.github.io/assets/case_studies/gaussian_processes.html

```julia
 
testing = false
if testing
      
    GPmethod="textbook"
    GPmethod="cholesky_meanprocess" 
    GPmethod="cholesky" 
    GPmethod="GPexact" 
    GPmethod="GPsparse" 
    GPmethod="GPvfe" 

    Random.seed!( Xoshiro(1234) )

    kernel_var = rand( filldist( Gamma(0.5, 1.0) , nG ))
    kernel_scale =rand( filldist( Gamma(0.5, 1.0), nG ) ) 
    l2reg =rand(  filldist(Gamma(1.0, 0.001), nG ) )  
 
   #  fkernel = kernel_var * SqExponentialKernel()  ∘ ScaleTransform(kernel_scale)
   i=1
    ys = sample_gaussian_process( GPmethod=GPmethod, 
      kvar=kernel_var[i], kscale=kernel_scale[i],
      Yobs=YG[good], Xobs=G[good,i], Xinducing=Gp[:,i], lambda=l2reg[i]
    )

 
end

# intercept only with offets
X, Xschema, Xcoefnames, nX = model_matrix_fixed_effects(
  M[io,:], 
  @formula(totno ~ 1   ), 
  contrasts=Dict( :year => StatsModels.EffectsCoding()  )
)


# poisson of numerical abundance
m = turing_glm_icar(  ; family="poisson", GPmethod=GPmethod, 
  good=good, X=X,  
  Y=y, YG=YG, G=G, Gp=Gp, nInducing=nInducing, log_offset=log_offset ) 

rand(m)


# poisson of numerical abundance
m = test_gp2( y, YG, G, Gp, nInducing, good, 1 ) 

rand(m)

n_samples, n_adapts, n_chains = 10, 10, 1
target_acceptance, max_depth, init_ϵ = 0.65, 10, 0.01   
  
turing_sampler = Turing.HMC(init_ϵ, 10 )  

turing_sampler = Turing.NUTS(n_adapts, target_acceptance; max_depth=max_depth, init_ϵ=init_ϵ)

msol = sample(m, turing_sampler, 10)
 
modelruntime(msol)

n_sample=10
MS = turing_glm_icar_summary( 
  "mcmc", 
  msol=msol, model=m, Y=y, YG=YG, 
  family="poisson", n_sample=n_sample, good=good, kerneltype=kerneltype,
  X=X, G=G, Gp=Gp, nInducing=nInducing )  # no offset means at standard rate (log_offset = 0) 


--- turingindex might not be working for mcmc outputs ..

i=1; plot( Gpp[:,i], MS.Gymu[:,i], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=2; plot( Gpp[:,i], MS.Gymu[:,i], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=3; plot( Gpp[:,i], MS.Gymu[:,i], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=4; plot( Gpp[:,i], MS.Gymu[:,i], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )

fixed_effects = DataFrame(
    parameter=String.(names(MS.fixed_effects)), 
    mean=vec(mean(MS.fixed_effects, dims=1)), 
    sd=vec(mapslices(std, MS.fixed_effects, dims=1))
)


# -----
# using optimizers: (fast)
turing_sampler = MLE() 

msol = optimize(m, turing_sampler, NelderMead(), Optim.Options(iterations=100) )   
 
n_sample=1
MS = turing_glm_icar_summary( 
  "optim",  GPmethod=GPmethod, 
  msol=msol, model=m, Y=y,  YG=YG, kerneltype=kerneltype,
  family="poisson", n_sample=n_sample, good=good,
  X=X, G=G, Gp=Gp, nInducing=nInducing )  # no offset means at standard rate (log_offset = 0)

# Gymu = ( msol.values[ turingindex( m, :Gymu, (:, nG) ) ] )

i=1; plot( Gpp[:,i], MS.Gymu[:,i], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=2; plot( Gpp[:,i], MS.Gymu[:,i], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=3; plot( Gpp[:,i], MS.Gymu[:,i], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=4; plot( Gpp[:,i], MS.Gymu[:,i], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )


# using variational inference:
 
using Turing.Variational
res = variational_inference_solution(m, max_iters=100 )
pm = res.mmean
msol = res.msol

n_sample=100
 
MS = turing_glm_icar_summary( 
  "variational_inference", GPmethod=GPmethod,
  msol=msol, model=m, Y=y,  YG=YG, kerneltype=kerneltype,
  family="poisson", n_sample=n_sample, good=good,
  X=X, G=G, Gp=Gp, nInducing=nInducing )  # no offset means at standard rate (log_offset = 0)


# for vi only:
_, sym2range = bijector(m, Val(true));
plot_variational_marginals(res, sym2range)

 


```



## Model 5: adding GP on covars 

-- mean priors

o = mapreduce(DynamicPPL.tovec ∘ mean, vcat, values(extract_priors(msol)))


-- Predictions on whole dataset
posteriors = sample(model_Linear(data.y, data.x, data.participant, 10), NUTS(), 200)

pred = predict(model_Linear(fill(missing, 100), data.x, data.participant, 10), posteriors)

   # NOTE: Return the values so we can use `generated_quantities` to extract it.
model ...
    return (; μ_fixed, μ_random)
end


# Then once we have a chain, we can do the following to extract the values.
results = generated_quantities(model, chain)


```julia

# product of separate kernels for all GP  snowcrab HURDLE model
# see: https://mc-stan.org/docs/2_20/stan-users-guide/zero-inflated-section.html

X, Xschema, Xcoefnames, nX = model_matrix_fixed_effects(
  M[io,:], 
  @formula(totno ~ 1 + year ), 
  contrasts=Dict( :year => StatsModels.EffectsCoding()  )
)

# poisson of positive valued (> 0) numerical abundance
m = turing_glm_icar_optimized( family="poisson", 
  GPmethod=GPmethod, kerneltype=kerneltype,
  good=good,  # good == positive valued data only
  Y=y, YG=YG, X=X, G=G, Gp=Gp, nInducing=nInducing, log_offset=log_offset, 
  auid=auid, nAU=nAU, node1=node1, node2=node2, scaling_factor=scaling_factor ) 


 
# presence - absence  -- all data, no offset
m = turing_glm_icar_optimized( family="binomial", 
  Y=pa, YG=YG, X=X, G=G, Gp=Gp, nInducing=nInducing, auid=auid, nAU=nAU, node1=node1, node2=node2, scaling_factor=scaling_factor ) 


# mean weight all data. .. no offset
m = turing_glm_icar_optimized( family="gaussian", 
  Y=wt, YG=YG, X=X, G=G, Gp=Gp, nInducing=nInducing, auid=auid, nAU=nAU, node1=node1, node2=node2, scaling_factor=scaling_factor ) 


rand(m)  # check a sample


# using variational inference:
using Turing.Variational
res = variational_inference_solution(m, max_iters=10 )
pm = res.mmean
msol = res.msol
 

msol_fn = joinpath(project_directory, "outputs", string("msol_turing_variationalinference", ".hdf5" ) )
@save msol_fn msol
# @load msol_fn msol
print( "\n\n", "Model object file: \n",  msol_fn, "\n\n" )
 

n_sample=100

MS = turing_glm_icar_summary( 
  "variational_inference", 
  msol=msol, model=m, Y=y, YG=YG,
  family="poisson", n_sample=n_sample, good=good,
  X=X, G=G, Gp=Gp, nInducing=nInducing,  kerneltype=kerneltype,
  scaling_factor=scaling_factor, nAU=nAU, auid=auid )  # no offset means at standard rate (log_offset = 0)



n_samples, n_adapts, n_chains = 10, 10, 1
target_acceptance, max_depth, init_ϵ = 0.65, 10, 0.01   

# Morris uses 0.97 for target_acceptance, stan default is 0.95; such high acceptance rate does not work well -- divergent chains

# if on windows and threads are still not working, use single processor mode:
# msol = mapreduce(c -> sample(m, turing_sampler, n_samples), chainscat, 1:n_chains)

using ForwardDiff; adtype = ADTypes.AutoForwardDiff()  #  ~ 67 sec
using ReverseDiff; adtype = ADTypes.AutoReverseDiff()

using Enzyme; adtype = ADTypes.AutoEnzyme()   #  crashing sec;
using Enzyme; adtype = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

using Zygote; adtype = ADTypes.AutoZygote()     # 130 sec
 

# using optimizers: (fast)

turing_sampler = MLE()
# turing_sampler = MAP()

niterations = 100
niterations = 1000 
niterations = 5000 

optim_options = Optim.Options(iterations=niterations, allow_f_increases=true)

# ignore gradients  -- might need to as -- cholesky can fail on non-PD matrices
optimizer = NelderMead()
# optimizer = SimulatedAnnealing()
 
# using gradients 
# optimizer = BFGS()
# optimizer = Newton()
# optimizer = AcceleratedGradientDescent()
# using Flux; optimizer = Flux.Adam() 

# many to choose from: https://julianlsolvers.github.io/Optim.jl/stable/user/config/ 
autodiff = :forward
autodiff = :reverse
autodiff = :Enzyme
autodiff = :Zygote
 
msol = optimize(m, turing_sampler, optimizer, optim_options) # ; autodiff=autodiff )


msol_fn = joinpath(project_directory, "outputs", string("msol_turing_mle", ".hdf5" ) )
@save msol_fn msol
# @load msol_fn msol
print( "\n\n", "Model object file: \n",  msol_fn, "\n\n" )


 
n_sample=1  # ... modes ... must bring in SD  
 
MS = turing_glm_icar_summary( 
  "optim", 
  msol=msol, model=m, Y=y, YG=YG, GPmethod=GPmethod,
  family="poisson", n_sample=n_sample, good=good,
  X=X, G=G, Gp=Gp, nInducing=nInducing,  kerneltype=kerneltype,
  scaling_factor=scaling_factor, nAU=nAU, auid=auid )   # no offset means at standard rate (log_offset = 0)
 

plot(MS.fixed_effects[2:11])


i=1; plot( Gpp[:,i], MS.Gymu[:,i,1], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=2; plot( Gpp[:,i], MS.Gymu[:,i,1], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=3; plot( Gpp[:,i], MS.Gymu[:,i,1], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=4; plot( Gpp[:,i], MS.Gymu[:,i,1], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )


# --------------

using ForwardDiff; adtype = ADTypes.AutoForwardDiff()  #  ~ 67 sec

using Enzyme; adtype = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

n_samples, n_adapts, n_chains = 10, 10, 1
target_acceptance, max_depth, init_ϵ = 0.65, 10, 0.01   

# testing : most are really slow, except MH
turing_sampler = Turing.MH()

# Turing samplers:
# turing_sampler = Turing.SMC()  # does not work with named arguments in models
turing_sampler = Turing.NUTS(; adtype=adtype )
turing_sampler = Turing.NUTS(n_adapts, target_acceptance; max_depth=max_depth, init_ϵ=init_ϵ, adtype=adtype )
turing_sampler = Turing.HMC(init_ϵ, 10, adtype=adtype)
turing_sampler = Turing.HMCDA( n_adapts, 0.65, 0.3; init_ϵ=init_ϵ, adtype=adtype)
# turing_sampler = Turing.Gibbs( Turing.HMC(0.2, 3, :v1), Turing.SMC(20, :v2) ) # an example only, SMC 

n_samples=5000
n_samples=100

# msol = sample( m, turing_sampler, n_samples, init_params = msol.values.array) # Sample with the MAP or MLE estimate as the starting point.
 
msol = sample(m, turing_sampler, n_samples)

modelruntime(msol)
  
msol_fn = joinpath(project_directory, "outputs", string("msol_turing_mh", ".hdf5" ) )
@save msol_fn msol
# @load msol_fn msol
print( "\n\n", "Model object file: \n",  msol_fn, "\n\n" )


n_sample=10

MS = turing_glm_icar_summary( 
  "mcmc", 
  msol=msol, model=m, Y=y, YG=YG,
  family="poisson", n_sample=n_sample, good=good,
  X=X, G=G, Gp=Gp, nInducing=nInducing,  kerneltype=kerneltype,
  scaling_factor=scaling_factor, nAU=nAU, auid=auid )  # no offset means at standard rate (log_offset = 0)


Xcoefnames
String.(names(msol))

plot(MS.fixed_effects[2:11])


i=1; plot( Gpp[:,i], MS.Gymu[:,i,1], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=2; plot( Gpp[:,i], MS.Gymu[:,i,1], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=3; plot( Gpp[:,i], MS.Gymu[:,i,1], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )
i=4; plot( Gpp[:,i], MS.Gymu[:,i,1], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )

# --------------

 
# using variational inference:

using ForwardDiff; adtype = ADTypes.AutoForwardDiff()  #  ~ 67 sec
using ReverseDiff; adtype = ADTypes.AutoReverseDiff()

using Enzyme; adtype = ADTypes.AutoEnzyme()   #  crashing sec;
using Enzyme; adtype = ADTypes.AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

using Zygote; adtype = ADTypes.AutoZygote()     # 130 sec
 

# ignore gradients  -- might need to as -- cholesky can fail on non-PD matrices
optimizer = NelderMead()
optimizer = SimulatedAnnealing()

# using gradients 
optimizer = BFGS()
optimizer = Newton()
optimizer = AcceleratedGradientDescent()
optimizer = SimulatedAnnealing()
using Flux; optimizer = Flux.Adam() 

 

# poisson of positive valued (> 0) numerical abundance
m = turing_glm_icar( family="poisson", 
  good=good,  # good == positive valued data only
  Y=y, YG=YG, X=X, G=G, Gp=Gp, nInducing=nInducing, log_offset=log_offset, 
  auid=auid, nAU=nAU, node1=node1, node2=node2, scaling_factor=scaling_factor ) 

# using variational inference:
using Turing.Variational
res = variational_inference_solution(m, max_iters=100 )
pm = res.mmean
msol = res.msol

msol_fn = joinpath(project_directory, "outputs", string("msol_turing_variationalinference", ".hdf5" ) )
@save msol_fn msol
# @load msol_fn msol
print( "\n\n", "Model object file: \n",  msol_fn, "\n\n" )

n_sample=100

MS = turing_glm_icar_summary( 
  "variational_inference", 
  msol=msol, model=m, Y=y, YG=YG,
  family="poisson", n_sample=n_sample, good=good,
  X=X, G=G, Gp=Gp, nInducing=nInducing,  kerneltype=kerneltype,
  scaling_factor=scaling_factor, nAU=nAU, auid=auid )  # no offset means at standard rate (log_offset = 0)


# for vi only:
_, sym2range = bijector(m, Val(true));
plot_variational_marginals(res, sym2range)




Xyear = modelcols( apply_schema(term( effectname ), Xschema ), M[io,:] )  # model matrix (contrasts)



plot(M[io,:year], mean(year_effect, dims=2), seriestype=:scatter)
for i in 1:n_sample
  plot!( M[io,:year], year_effect[:,i], seriestype=:scatter) 
end

i=4; plot( Gpp[:,i], Gymu[:,i], xlab=Gvars[i], ylab="Effect", seriestype=:path, label=Gvars[i] )

 
uu = turingindex( m, :beta  ) 
vv = turingindex( m, "varnames"  ) 
ww = turingindex( m ) 



 msol[turingindex( m, :Gymu),:]
 


# final run:
n_samples, n_adapts, n_chains = 1000, 1000, 4
target_acceptance, max_depth, init_ϵ = 0.65, 10, 0.001   
turing_sampler = Turing.NUTS(n_adapts, target_acceptance; max_depth=max_depth, init_ϵ=init_ϵ)
msol = sample( m, turing_sampler, MCMCThreads(), n_samples, n_chains  ) # to see progress
# msol = sample( m, turing_sampler, n_samples, init_params = omap.values.array) # Sample with the MAP estimate as the starting point.

chain_reloaded = deserialize("/Program_Julia/chain.jls")

for i in 1:20
 
    println("Start$(i)")
    chains = sample(model, NUTS(), MCMCThreads(), 1000, 2; progress = true, save_state = true, resume_from = chain_reloaded)
   
    plot(chains)
    savefig("/Program_Julia/trial/chain_$(i).pdf")

    plot_fit(chains, i)
    savefig("/Program_Julia/trial/fit_$(i).pdf")
    
    serialize("/Program_Julia/chain_new.jls", chains)
    chain_reloaded = deserialize("/Program_Julia/chain_new.jls")
    println("End$(i)")

end


--

last_state = chain.info.samplerstate;
# Continue sampling.
chain_continuation = sample(
    model, alg, 500;
    # NOTE: At the moment we have to use `resume_from` because Turing.jl
    # is slightly lagging behind AbstractMCMC.jl, but soon we will use
    # `initial_state` instead, which is consistent with the rest of the
    # ecosystem.
    resume_from=last_state,
    # initial_state=last_state,
);
range(chain_continuation)
1:1:500

# Can only concatenate chains if the iterations are consistent.
       # So we have to update the iterations of the second chain.
       chain_continuation = setrange(
           chain_continuation,
           range(chain_first)[end] .+ range(chain_continuation)
       );

chain_combined = vcat(chain_first, chain_continuation);

range(chain_combined)



---


    parameters      mean       std   naive_se      mcse          ess      rhat   ess_per_sec 
       beta[1]    0.2189    0.9864     0.3119    0.1836     -13.6352    0.9177       -0.2413
       sum_phi   -0.0747    0.5817     0.1839    0.2148    -449.3469    0.8968       -7.9521
         sigma    0.6706    0.4297     0.1359    0.2152       6.8018    1.1394        0.1204
           rho    0.2819    0.2682     0.0848    0.0954       5.6222    1.1564        0.0995
    kernel_var    1.0288    0.5491     0.1736    0.0600      85.5647    0.9276        1.5142
  kernel_scale    0.1519    0.1194     0.0377    0.0406     -40.4384    0.8944       -0.7156
        lambda    0.0018    0.0012     0.0004    0.0008       8.8172    1.0525        0.1560
           eta    0.2150    1.1391     0.3602    0.3007     -59.0233    0.9895       -1.0445
 

# this function needs to be updated for GP
p = turing_glm_icar_predict( msol, Xp; Gp=Gp, scaling_factor=scaling_factor, n_sample=10, nAU=nAU )

 
 

```

## Model 6: adding spatiotemporal random effects


```{julia} 

Turing.@model function ar1_gp( ::Type{T}=Float64; Y, ar1,  nData=length(Y), nT=Integer(maximum(ar1)-minimum(ar1)+1) ) where {T} 
    Ymean = mean(Y)
    rho ~ truncated(Normal(0,1), -1, 1)
    ar1_process_error ~ LogNormal(0, 1) 
    var_ar1 =  ar1_process_error^2 / (1 - rho^2)

    # -- covariance by time
    covt = zeros(n, n) .+ I(n) 
    for r in 1:nT
    for c in 1:nT
        if r >= c 
            covt[r,c] = var_ar1 * rho^(r-c) 
        end
    end
    end

    ymean_ar1 ~ MvNormal(Symmetric(covt) );  # -- means by time 
    observation_error ~ LogNormal(0, 1) 
    Y ~ MvNormal( ymean_ar1[ar1[1:nData]] .+ Ymean, observation_error )     # likelihood
end



# if grouped spatial locations

  groups_unique = unique(sort(groups))
  gi = Vector{Vector{Int64}}()
  for g in groups_unique
      msol =  findall(x -> x==g, groups) 
      push!(gi, msol)
  end

  scaling_factor = scaling_factor_bym2(node1, node2, groups)  # same function (overloaded)
  
 ```



 
