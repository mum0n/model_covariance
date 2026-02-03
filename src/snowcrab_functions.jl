
function snow_crab_survey_data(yrs)

  fndat1 = datadir( "snowcrab", "snowcrab_data.rdz" )
  fndat2 = datadir( "snowcrab", "snowcrab_nb.rdz" )
  fndat3 = datadir( "snowcrab", "snowcrab_sppoly.rdz" )

  # load and unwrap containers
  M  = load( fndat1, convert=true)["M"]
  nb = load( fndat2, convert=true)["nb"]["nbs"]
  sp = load( fndat3, convert=true)["sppoly"]

  M = M[∈( yrs ).(M.year), :] 

  return M, nb, sp
 
end


function get_gp_covariates( ; M, Gvars=["z", "t", "pca1", "pca2"], nUnique = 100,  nInducing = 9
    )
  
    # G0 are originating data
    G0 = Matrix( M[io, Gvars] );
    nG = length(Gvars)
  
    # limit bounds 
    k = []
    k = findall(x->x=="z", Gvars)
    if !isempty(k) 
      l = []
      l = findall(x -> x>300,  vec(G0[:,k]))
      if !isempty(l) 
        G0[l, k] .= 300.0  # cap
      end
    end
  
    k = []
    k = findall(x->x=="t", Gvars)
    if !isempty(k) 
      l = []
      l = findall(x -> x>10,  vec(G0[:,k]))
      if !isempty(l) 
        G0[l, k] .= 10.0  # cap
      end
    end
  
    k = []
    k = findall(x->x=="pca1", Gvars)
    if !isempty(k) 
      l = []
      l = findall(x -> x>5, vec(G0[:,k]))
      if !isempty(l) 
        G0[l, k] .= 5.0  # cap
      end
    end
  
    k = []
    k = findall(x->x=="pca2", Gvars)
    if !isempty(k) 
      l = []
      l = findall(x -> x>5,  vec(G0[:,k]))
      if !isempty(l) 
        G0[l, k] .= 5.0  # cap
      end
    end
  
  
    # G are standardized .. centered to mean of 0 and scaled to SD (zscores)
    G = zeros(size(G0))
    G_means = mean(G0, dims=1)
    G_sds = std(G0, dims=1)
    for i in 1:nG 
      G[:,i] = ( G0[:,i] .- G_means[i] ) ./ G_sds[i] ;
    end
   
    # inducing_points for GP, for prediction  
    Gr = zeros(size(G))  # recoded   
    Gp = zeros(nInducing, nG)   # inducing points
    qu = LinRange(0.0, 1.0, nInducing+1)[2:nInducing+1]
    for i in 1:nG
        Gp[:,i] = quantile(vec(G[:,i]), qu)
        Gr[:,i] = Gp[ searchsortedfirst.(Ref(Gp[:,i]), G[:,i]), i ] 
    end
  
    Gpp = zeros(size(Gp))
    for i in 1:nG 
      Gpp[:,i] = G_sds[i] .*  Gp[:,i] .+ G_means[i]  
    end
   
    return (
      G0, G, Gp, Gr, nG, G_means, G_sds, Gpp
    )
  
end
  

function model_matrix_fixed_effects(data, formula_fixed_effects; contrasts )

  Xmf = ModelFrame( formula_fixed_effects, data, contrasts=contrasts)

  Xcoefnames = coefnames(Xmf)

  Xschema = Xmf.schema
  X = modelmatrix( Xmf )
  nX = size(X, 2)

  return X, Xschema, Xcoefnames, nX
end


Turing.@model function snowcrab_full_model(; Ymodel="poisson", Y, nData=length(Y), 
    X=nothing, G=nothing, log_offset=nothing, 
    auid=nothing, nAU=nothing, node1=nothing, node2=nothing, scaling_factor=nothing, 
    good= nothing )
    
    # needs to be checked ... not used as it is very slow
    
    # note column 1 = poisson and column 2 is binomial

    beta ~ filldist( Normal(0.0, 1.0), nX, 2);
    theta ~ filldist( Normal(0.0, 1.0), nAU, 2)  # unstructured (heterogeneous effect)
    phi ~ filldist( Normal(0.0, 1.0), nAU, 2) # spatial effects: stan goes from -Inf to Inf .. 
    
    # icar likelihood
    dphi = phi[node1,:] - phi[node2,:]
    Turing.@addlogprob! (-0.5 * dot( dphi, dphi ))
    
    # soft sum-to-zero constraint on phi
    sum_phi = sum(phi, dims=1) 
    sum_phi[1] ~ Normal(0, 0.001 * nAU);      
    sum_phi[2] ~ Normal(0, 0.001 * nAU);      

    # respect columns as poisson and binomial
    sigma ~ filldist( Exponential(1.0), 1, 2); 
    rho ~ filldist( Beta(0.5, 0.5), 1, 2);
 
    convolved_re = sigma .*( sqrt.(2 .- rho) .* theta .+ sqrt.(rho ./ scaling_factor) .* phi )

    lambda =  X[good,:] * beta[:,1] + convolved_re[auid[good],1] + log_offset[good]  #non GP components
    pr_habitat = X * beta[:,2] + convolved_re[auid,2]  

    kernel_var ~ filldist( Gamma(2.0, 0.5), 1,2)
    kernel_scale ~ filldist(Gamma(2.0, 0.1), 1,2) # even more left shifted with mode ~0.1 .. ~ magnitudes of ~ 0.1 seems to be stable
    l2reg ~ filldist( Gamma(1.0, 0.0001), 1,2)   # = 1e-4  # L2 regularization factor for ridge regression

    # k = prod( [ ( kernel_var[i] * SqExponentialKernel() ) ∘ ScaleTransform(kernel_scale[i]) for i in 1:nG ] )
    # km = kernelmatrix( k, G, obsdim=1 ) + l2reg*I # add regularization

    km = kernelmatrix( ( kernel_var[1] * SqExponentialKernel() ) ∘ ScaleTransform(kernel_scale[1]), 
        G[good,:], obsdim=1 ) + l2reg[1]*I # makes PD and cholesky, add regularization

    km_hab = kernelmatrix( ( kernel_var[2] * SqExponentialKernel() ) ∘ ScaleTransform(kernel_scale[2]), 
        G, obsdim=1 ) + l2reg[2]*I # makes PD and cholesky, add regularization
 
    # lambda = max.(zero(eltype(lambda)), lambda) # a method to truncate safely
    lambda += sample_from_kernelmatrix(km)
    pr_habitat += sample_from_kernelmatrix(km_hab)

    # Hurdle process
    # @. pa ~ Bernoulli( logistic(pr_habitat[auid]) )   
    # @. y[good] ~ truncated( Poisson(  exp(lambda[good]) ),  2, nothing  ) ; # 1 or less is considered poor habitat
 
    pa ~ arraydist( @. Bernoulli( logistic.(pr_habitat) ) ) 
    y[good] ~ arraydist( @. LogPoisson( lambda ) )   ; # good = findall(x -> x==1, pa)

 
    #=
    # dynamics
    K ~ filldist( truncated(Normal( PM.K[1], PM.K[2]), PM.K[3], PM.K[4]), nAU)

    r ~  truncated(Normal( PM.r[1], PM.r[2], PM.r[3]), PM.r[4])   # (mu, sd)
    
    bpsd ~  truncated(Normal( PM.bpsd[1], PM.bpsd[2]), PM.bpsd[3], PM.bpsd[4] )  ;  # slightly informative .. center of mass between (0,1)
    bosd ~  truncated(Normal( PM.bosd[1], PM.bosd[2]), PM.bosd[3], PM.bosd[4] )  ;  # slightly informative .. center of mass between (0,1)
    q ~ truncated(Normal( PM.q[1], PM.q[2]), PM.q[3], PM.q[4] )    
    qc ~ truncated(Normal(PM.qc[1], PM.qc[2]), PM.qc[3], PM.qc[4]  ) 

    m = tzeros( PM.nM, nAU )

    @.  m[1,:] ~ truncated(Normal( PM.m0[1], PM.m0[2]), PM.m0[3], PM.m0[4] )  ; # starting b prior to first catch event

    for i in 2: PM.nT
        for j in 1:nAU
            m[i,j] ~ truncated(Normal( m[i-1,j] + r * m[i-1,j] * ( 1.0 - m[i-1,j] ) -  PM.removed[i-1,j]/K[j], bpsd), PM.mlim[1], PM.mlim[2])  ;
        end
    end
 
    if any( x -> x < 0.0 || x >1.0, m)
        Turing.@addlogprob! -Inf
        return nothing
    end
    
    # likelihood
    # observation model: Y = q X + qc ; X = (Y - qc) / q
    # m = abundance (prefishery) 
    PM.S[i] ~ Normal( q * ( m[i] - PM.removed[i]/K )+ qc, bosd )  ; # fall survey
=#

    return nothing
end
 
 
