
# Spatiotemporal modelling -- SES analysis using a bym2 (space) X ar1 (time) in julia

Recreate SES analysis using MCMC instead of INLA

## Data

The data are created in:
    
    Deprivation_Index/inst/scripts/01_prepare_census_COMe_data.R

Here we load the data and save all rolled into one file.

```{r}
    # This is mostly copied from Deprivation_Index/inst/scripts/01_prepare_census_COMe_data.R

    # identify key directory locations
    # network_root_dir = file.path( "S:", "S & E Unit - System files" )
    local_root_dir = file.path( "C:", "NATHALIE - LOCAL" )
    root.directory = local_root_dir  # use local as default
    workspace_environment = file.path(root.directory, "NATHprojects", "Projects", "Deprivation_Index", "Programs", "Deprivation_Index", "inst", "scripts", "00_define_workspace.R")
    source( workspace_environment )

    #this loads "census"
    census = read_write_fast(file=file.path( data.directory, "CENSUS_transf.rdz" )  )

    # operate on indat .. keep census untouched
    indat = census

    # remove Eskasoni from analyses  .. better to censor it after analysis ...
    remove.Eskasoni = FALSE
    if (remove.Eskasoni) indat = indat[ which(indat$COMe != 1713) , ]
 
    single.year = FALSE  # set year selection below
    if (single.year) indat = indat[ which(indat$year == "2001") , ]
  
    # aus= areal units,
    # tus = temporal units
    aus = indat$COMe
    tus = indat$year
    row.names(indat) = paste( aus, tus, sep="_")

    # select variables for analysis
    varnames = c(
        "sdw_asin", "loneP_asin",  "alone_asin", "move1yr_asin", "FAMsiz_log10",
        "imm5yr_asin", "vminor_asin", "refugee_asin", "nonoffLG_asin", "aborLG_asin",
        "noEDUC_asin", "noHD_asin", "noPSD2564_asin", "no_edu65_asin",
        "emplRATE_asin", "LAB_unempl_asin", "lab_PT_asin",
        "INCavg_log10",  "INCmed_log10", "INCempAVG_log10","INCgt_asin","spend30pct_asin", 
        "limat2064_asin", "limat65_asin",
        "hous_repair_asin","hous_rent_asin", "hous_value_log10", "subhousing_asin", "hous_crowd_asin"
    )
    indat = indat[, varnames]
    

normalize_from_quantiles = function(X, vns=colnames(X), prob_limits=0.95 ) {

  for (j in 1:length(vns)) {
    x = X[,j]    
		pr = ecdf(x) ( x )
		
    pr_one_tail = (1 - prob_limits) / 2 
    prlb = pr_one_tail
    prub = 1 - pr_one_tail

    u = which( pr < prlb )
    if (length(u) > 0 ) pr[u] = prlb
    
    u = which( pr > prub )
    if (length(u) > 0 ) pr[u] = prub
    
    X[,i] = qnorm( pr, mean=0.0, sd=1.0 )
  } 
  return(X)
}

    # indat = normalize_from_quantiles(indat)  # PCA assumes MVN .. force normality

    # nearest neighbour map: nb
    pg = polygon_data( "COMe" )
    nb = slot(pg, "nb" )[["nbs"]]
    auid_label = pg$AUID_label
    auid = pg$AUID

    # numeric values used by inla
    AU = match( aus, slot(pg, "space.id") )
    TU = match( tus, sort(unique(tus) ) )
 
    # if there is missing data, impute using library 'mice'
    if (length(which(! is.finite( as.matrix(indat) )))>0) {
       indat = impute_data( indat, m=100, reducefn=mean, method="norm")
    }
 
    # roll all data into one odject
    # "C:/NATHALIE - LOCAL/NATHprojects/Projects/Deprivation_Index/Data/ses_data.rdata"
    
    read_write_fast( data=list( 
        indat=indat, nb=nb, AU=AU, TU=TU, varnames=varnames, 
        tus=tus, auid=auid, auid_label=auid_label), 
      fn=file.path( data.directory, "ses_data.rdata" ) 
    )


  # ----  stop ..remainder here are notes

  # # eg., see:
  # https://www.stat.berkeley.edu/~rabbee/correlation.pdf
  # Rodgers, Joseph Lee, and W. Alan Nicewander.
  # “Thirteen Ways to Look at the Correlation Coefficient.”
  # The American Statistician, vol. 42, no. 1, 1988, pp. 59–66.
  # JSTOR, www.jstor.org/stable/2685263. Accessed 17 Aug. 2020.
  # and:
  # Marks, Edmond. “A Note on a Geometric Interpretation of the Correlation Coefficient.”
  #   Journal of Educational Statistics, vol. 7, no. 3, 1982, pp. 233–237.
  #   JSTOR, www.jstor.org/stable/1164647. Accessed 17 Aug. 2020.

  # most reasonable fit is probably (dic) - TO EMPHAZIZE THE IMPORTANCE OF BOTH SPACE AND TIME IN A BALANCED WAY:

  method = "multipleslope_bym.multipleintercept_ar1.inla" #USE THIS METHOD 'MSMI BYM AR1'
    
  correlation_matrix_modelled( indat=indat, TU=TU, AU=AU, map=nb, method=method,
      redo_models=TRUE, data.directory=data.directory )

    # to redo or add new variables that are missing
  correlation_matrix_modelled( indat=indat, TU=TU, AU=AU, map=nb, method=method,
      redo_models=TRUE, data.directory=data.directory,
      new_varnames = c(
        "INCmed_log10", "INCempAVG_log10", "limat65_asin",
        "nonoffLG_asin", "refugee_asin", "hous_crowd_asin")
    )

  # Mine
  pca.vars = c(
    "INCavg_log10", "LAB_unempl_asin", "loneP_asin", "sdw_asin", "noEDUC_asin", "alone_asin")


  #  MATERIAL ONLY
  pca.vars = c(
    "noEDUC_asin",
    "LAB_unempl_asin",
    "INCavg_log10","INCgt_asin", "spend30pct_asin", "limat2064_asin", "limat65_asin",
    "hous_repair_asin","hous_value_log10"  
  )


  #  SOCIAL ONLYY

  pca.vars = c(
    "sdw_asin", "loneP_asin",  "alone_asin", "move1yr_asin",
    "imm5yr_asin", "vminor_asin",  "nonoffLG_asin"
    )

  pca_results = ...


    pca_results$RC1Q = population_quantiles( x=pca_results$rotated_scores[,1], weights=pop, ngroups=5, strata=tus)
    pca_results$RC2Q = population_quantiles( x=pca_results$rotated_scores[,2], weights=pop, ngroups=5, strata=tus)


  iau = match( pg$AUID,  aus[iyr] )


  # CHOOSE BELOW IS USING ROTATED SCORES
  pg["RC1 SOCIAL"] = pca_results$rotated_scores[iyr,1] [iau]
  pg["RC2 MATERIAL"] = pca_results$rotated_scores[iyr,2] [iau]

  pg["RC1Q SOCIAL"] = pca_results$RC1Q[iyr] [iau]
  pg["RC2Q MATERIAL"] = pca_results$RC2Q[iyr] [iau]

  pg["income"] = indat[iyr , "INCavg_log10" ]  [iau]

  plot( pg["income"] )

```


## Import data into julia for analysis and complete a basic PCA (SVD-based)
    

```{julia}

    # first load julia functions and packages
    project_directory = joinpath( homedir(), "projects", "model_covariance"  )
    src_directory = joinpath( project_directory, "src" )

    funcs = ( "startup.jl", "pca_functions.jl",  "regression_functions.jl", "car_functions.jl", "carstm_functions.jl" )
    for f in funcs
      include( joinpath( src_directory, f) )
    end

    Random.seed!(1); # Set a seed for reproducibility.
      
    # choose location of data:
    if false 
      project_data_directory = "C:/NATHALIE - LOCAL/NATHprojects/Projects/Deprivation_Index/Data" # mswin default
      project_data_directory = "/home/nath/deprivation/Data"        # nath default
      project_data_directory = joinpath(project_directory, "data")  # jae default
    end

    project_data_directory = joinpath(project_directory, "data")  # jae default

    fn  = joinpath( project_data_directory, "ses_data.rdata" )
    ses = RData.load( fn, convert=true )
     
    Y = ses["indat"]  
    auid = ses["AU"]
    tuid = ses["TU"]
    
    auid_names = parse.(Int, ses["auid"])
    auid_label = ses["auid_label"]
    vn = ses["varnames"]
    nb = ses["nb"]

    nob, nvar = size(Y)   
    nz = 2  # no latent factors to use
  
    # log_offset (if any)
    nAU = length( auid )  # no of au

    tuid_names = unique( ses["tus"]  )
    tuid_label = convert.(Integer, tuid_names )

    nTU = length( tuid_names )  # no of tu
 
    ## Basic PCA (SVD-based)
    # basic pca ..
    Yn = normalize_from_quantiles(Y) # convert to quantile and then zscore 
        
    evecs, evals, pcloadings, variancepct, C, pcscores = pca_standard( 
      Yn; 
      model="cor_pairwise", obs="rows", scale=true, center=true 
    )  # sigma is std dev, not variance.

    biplot(pcscores=pcscores, pcloadings=pcloadings,  evecs=evecs, evals=evals, vn=vn, variancepct=variancepct, type="standardized"  )   
    #  plot!(xlim=(-2.5, 2.5))
     
 variancepct #29-element Vector{Float64}:
 17.89
 15.19
 10.22
 
```

## Probabilistic Bayesian Householder Transformed PCA 
    
Some preliminary information:

```{julia}

    # PCA with householder transform
    noise=1.0e-9 
    nAU_float = convert(Float64, nAU) 
    n_samples = 500  # posterior sampling
    turing_sampler = Turing.NUTS()  
    nvh, hindex, iz, ltri = PCA_BH_indexes( nvar, nz )  # precompute indices 
    v_prior = eigenvector_to_householder(evecs, nvar, nz, ltri )  # convert evecs to householder
    # householder_to_eigenvector( lower_triangle( v_prior, nvar, nz ) ) .- evecs[:,1:nz] # inverse transform
     
    # param sequence = sigma_noise, sigma(nz), v, r=norm(v)~ 1.0 (scaled)
    sigma_prior = log.(sqrt.(evals)[1:nz])

```

## PPCA

```{julia}

    # direct ppca
    M = ppca_basic( Y' )  # pca first  # rand(M)  
    
    # get better starting conditions
    res = sample( M, Prior(), 10 )   # first from prior (and check if params/model are ok)
    init_params = init_params_extract(res) 
    
    res = sample( M, Turing.SMC(), 1000, init_params=init_params) # cannot be larger 
    init_params = init_params_extract(res, load_from_file=false, fn_inits=joinpath(project_data_directory, "ses_inits.jl2") ) # marginally more informed starting conditions .. save to file (if expensive)
    n_chains =1

    res = sample(M, Turing.NUTS( 0.65 ), MCMCThreads(), 1000,  n_chains,  init_params=init_params )  

    res_ppca = res  # copy for use later 

    summarystats(res)

Iterations        = 501:1:1500
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 2443.74 seconds = 40 mins
 
     parameters      mean       std   naive_se      mcse       ess      rhat   ess_per_sec 
       Symbol   Float64   Float64    Float64   Float64   Float64   Float64       Float64 

    pca_sd[1]    2.2099    0.0147     0.0005    0.0026    2.2370    3.0418        0.0009
    pca_sd[2]    3.0541    0.0207     0.0007    0.0037    2.2520    2.8332        0.0009
  pca_pdef_sd    0.7068    0.0015     0.0000    0.0002    6.1429    1.2294        0.0025
         v[1]    0.9038    0.0024     0.0001    0.0004    3.6295    1.3105        0.0015
         v[2]    1.5076    0.0044     0.0001    0.0008    2.3465    2.4973        0.0010
         v[3]    1.8652    0.0048     0.0002    0.0009    2.3508    2.2819        0.0010

    pca_sd[1]    2.2315    0.0022     0.0001    0.0002    17.9684    1.0577        0.0031
    pca_sd[2]    3.1258    0.0049     0.0002    0.0006    63.8751    1.0021        0.0110
  pca_pdef_sd    0.7124    0.0009     0.0000    0.0002     3.8655    1.3039        0.0007
         v[1]   -0.8141    0.0041     0.0001    0.0003   346.7140    1.0600        0.0596
         v[2]   -1.2518    0.0051     0.0002    0.0007    46.7607    1.0105        0.0080
         v[3]   -1.5385    0.0070     0.0002    0.0009     5.5904    1.1790        0.0010


>>> --- probably want more samples, solutions do not seem stable

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

    # variability of a single solution     
    j = 2  # observation index
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
    
```

## Same  (Probabilistic Bayesian Householder Transformed) PCA but using Variational Inference

```{julia}

     # to do Variational Inference  ~ 40 minutes (no speed gain)
    samples_per_step, max_iters = 2, 1000  # Number of samples used to estimate the ELBO in each optimization step.
    M0 = ppca_basic( Y' )  # pca first  # rand(M0)  
    res_vi =  vi( M0, Turing.ADVI( samples_per_step, max_iters) ); 

    vns = res_vi.name_map.parameters

    pca_sd, evals, evecs, pcloadings, pcscores = 
        PCA_posterior_samples_vi( res_vi, Y, vns, nz=nz, model_type="householder",  n_samples=1000  ) 

    evecs_mean = DataFrame( convert(Array{Float64}, mean(evecs, dims=1)[1,:,:]), :auto)
    pcloadings_mean = DataFrame( convert(Array{Float64}, mean(pcloadings, dims=1)[1,:,:]), :auto)
    pcscores_mean = DataFrame( convert(Array{Float64}, mean(pcscores, dims=1)[1,:,:]), :auto)
    # pcscores_mean = reshape(mapslices( mean, pcscores, dims=1 ), (nob, nz))
  
   
    pl = plot( pcscores_mean[:,1], pcscores_mean[:,2], label=:none, seriestype=:scatter )
 

```

## Now add a spatial model (bym2) to the (Probabilistic Bayesian Householder Transformed) PCA.

So a space model ... no time

```{julia}

    # pre-compute required vars from adjacency_matrix for bym2
    node1, node2, scaling_factor = nodes( nb ) 
    
    # PCA with householder transform
    noise=1.0e-9 
    nAU_float = convert(Float64, nAU) 
    n_samples = 500  # posterior sampling
    turing_sampler = Turing.NUTS()  
    nvh, hindex, iz, ltri = PCA_BH_indexes( nvar, nz )  # precompute indices 
    v_prior = eigenvector_to_householder(evecs, nvar, nz, ltri )  # convert evecs to householder 
 

    z=1  # choose axis
    M = pca_carstm( Y' )  # pca first and then carstm  #  rand(M)  
    res = sample(M, Prior(), 10 )  

    res = sample(M, Turing.SMC(), 100; init_ϵ=0.01)
  
Summary Statistics
   parameters      mean       std   naive_se      mcse       ess      rhat   ess_per_sec 
       Symbol   Float64   Float64    Float64   Float64   Float64   Float64       Float64 
 
    pca_sd[1]    0.9001    0.0000     0.0000    0.0000       NaN       NaN           NaN
    pca_sd[2]    2.4548    0.0000     0.0000    0.0000       NaN       NaN           NaN
  pca_pdef_sd    0.9533    0.0000     0.0000    0.0000    2.0408    0.9899        0.1544
         v[1]    0.6092    0.0000     0.0000    0.0000    2.0408    0.9899        0.1544
         v[2]    0.0717    0.0000     0.0000    0.0000       NaN       NaN           NaN
         v[3]    0.7527    0.0000     0.0000    0.0000    2.0408    0.9899        0.1544


  
    init_params = init_params_copy(res_ppca, res)  # add pca solutions to res
 
    # slow: [ADVI] Optimizing...   0%  ETA: 4 days
    # to do Variational Inference 

    using Turing.Variational

    samples_per_step, max_iters = 3, 500 # Number of samples used to estimate the ELBO in each optimization step.
    res_vi =  vi( M, Turing.ADVI( samples_per_step, max_iters)  )  ;  # Automatic Differentiation Variational Inference (ADVI).

    rand(res_vi)
    logpdf(res_vi, rand(res_vi))

    q0 = Variational.meanfield(M)
    typeof(q0)

    vns = res_vi.name_map.parameters

    pca_sd, evals, evecs, pcloadings, pcscores = 
        PCA_posterior_samples_vi( res_vi, Y, vns, nz=nz, model_type="householder",  n_samples=1000  ) 

    evecs_mean = DataFrame( convert(Array{Float64}, mean(evecs, dims=1)[1,:,:]), :auto)
    pcloadings_mean = DataFrame( convert(Array{Float64}, mean(pcloadings, dims=1)[1,:,:]), :auto)
    pcscores_mean = DataFrame( convert(Array{Float64}, mean(pcscores, dims=1)[1,:,:]), :auto)
    # pcscores_mean = reshape(mapslices( mean, pcscores, dims=1 ), (nob, nz))
  
   
    pl = plot( pcscores_mean[:,1], pcscores_mean[:,2], label=:none, seriestype=:scatter )
 
 

    init_params = init_params_copy(res_ppca, res)

    res = sample(M, Turing.NUTS(), 100; init_params=init_params, init_ϵ=0.01)# ETA: 7 days, 
   
    Turing.setadbackend(:enzyme)
    Turing.setadbackend(:forwarddiff) 

    res = optimize(M, MLE(), Optim.Options(iterations=100) )
    res = optimize(M, MLE())
    res = optimize(M, MLE(), LBFGS(), Optim.Options(iterations=100))
    res = optimize(M, MLE(), NelderMead())
    res = optimize(M, MLE(), SimulatedAnnealing())
    res = optimize(M, MLE(), ParticleSwarm())
    res = optimize(M, MLE(), Newton())
    res = optimize(M, MLE(), AcceleratedGradientDescent(), Optim.Options(iterations=100) )
    res = optimize(M, MLE(), Newton(), Optim.Options(iterations=100, allow_f_increases=true))
 
 
    res_map = optimize(M, MAP())
    init_params = res_map.values.array
    turing_sampler = Turing.HMC(0.01, 1000)    

    turing_sampler = Turing.PG(5)    
    turing_sampler = Turing.SMC()   #   
    # turing_sampler = Turing.SGLD()   # Stochastic Gradient Langevin Dynamics (SGLD); slow, mixes poorly
    turing_sampler = Turing.NUTS( 0.65 ) # , init_ϵ=0.001
    
    pca_sd[1]    0.6488    0.4024     0.0057    0.0475   11.9483    1.0094        0.2637
    pca_sd[2]    1.2565    0.7777     0.0110    0.0898   25.1972    1.0000        0.5561
  pca_pdef_sd    0.9383    0.0541     0.0008    0.0063   12.1021    1.4674        0.2671
         v[1]    0.2961    1.1967     0.0169    0.1386   13.5021    1.0478        0.2980
         v[2]   -0.4846    0.8510     0.0120    0.0980   16.6054    1.0011        0.3665
         v[3]   -0.0689    0.7648     0.0108    0.0871   14.5510    1.1328        0.3212
    
    pca_sd[1]    0.5658    0.0000     0.0000    0.0000   10.5465    0.9998        0.2821
    pca_sd[2]    1.2229    0.0000     0.0000    0.0000   10.5465    0.9998        0.2821
  pca_pdef_sd    0.9014    0.0000     0.0000    0.0000   10.5465    0.9998        0.2821
         v[1]    0.4848    0.0000     0.0000    0.0000   10.5465    0.9998        0.2821
         v[2]    0.1959    0.0000     0.0000    0.0000   10.5465    0.9998        0.2821
         v[3]    0.5875    0.0000     0.0000    0.0000   10.5465    0.9998        0.2821

    res = sample(M, turing_sampler, 1000,  init_params=init_params) # cannot be larger than 1000 , so iteratively restart
  
  
    init_params = init_params_extract(res, override_means=true)  # updates a file each time 
    res = sample(M, turing_sampler, 5000,  init_params=init_params) # cannot be larger than 1000 , so iteratively restart

    init_params = init_params_extract(res, override_means=false)  # updates a file each time 
    res = sample(M, turing_sampler, 1000, drop_warmup=true,  init_params=init_params)  # RAM is a problem ... and sequential ..keep nsamples  ~ 1000  -> 52G
    
        
    turing_sampler = Turing.NUTS( 0.65 ) # , init_ϵ=0.001
    res = sample(M, turing_sampler, 10,  init_params=init_params) # cannot be larger than 1000 , so iteratively restart


    summarystats(res)

    # posterior_summary(res, sym=:pca_sd, stat=:mean, dims=(1, nz))

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
    read_write_fast( data=list(
        pca_sd=pca_sd, evals=evals, evecs=evecs, pcloadings=pcloadings, pcscores=pcscores, 
        f_intercept=f_intercept, eta=eta, t_amp=t_amp, t_period=t_period, t_phase=t_phase, s_theta=s_theta, s_phi=s_phi, s_sigma=s_sigma, s_rho=s_rho, convolved_re_s=convolved_re_s),
      fn='/archive/bio.data/aegis/speciescomposition/data/carstm_pca.rdata' )
    """

 
 
   

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


```

 
   