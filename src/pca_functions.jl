# These are modified from the reference implmentations 
showall(x) = show(stdout, "text/plain", x)
 
function discretize_decimal( x, delta=0.01 ) 
  num_digits = Int(ceil( log10(1.0 / delta)) )   # time floating point rounding
  out = round.( round.( x ./ delta; digits=0 ) .* delta; digits=num_digits)
  return out
end


function normalize_from_quantiles(X; cindices=missing, vns=missing, prob_limits=0.999 ) 

    if !ismissing(vns)
      nr = size(X, 1)
      nc = length(vns)
      if isa(X, DataFrame) 
        cindices = findall( in(vns), names(X) )
      end
    end
  
    if ismissing(cindices) 
      nr, nc = size(X)
      cindices = 1:nc
    end
  
    n = Normal(0.0, 1.0)
  
    pr = (1.0 - prob_limits) / 2.0
    pr0 = pr # lower tail (2-tailed dist) 
    pr1 = 1.0 - pr # upper tail (2-tailed dist) 
  
    for j in cindices 
      x = X[:,j]
      i = findall(u -> isfinite(u), X[:,j] ) 
          if length(i) < 3 
        continue
      end
  
      x[i] = ecdf(x[i])( x[i] ) 
    
      i0 = findall( u -> u < pr0, x ) 
      if length(i0) > 0
        x[i0] .= pr0
      end
      
      i1 = findall( u -> u > pr1, x ) 
      if length(i1) > 0
        x[i1] .= pr1
      end
  
      X[:,j] = quantile( n, x )
  
    end
  
    return X
end
  
function iris_data(; nonlinear=false, subset_data=0, center=true, scale=true, obs="rows" )

    # from R: data("iris")
    # write.csv( iris, file=file.path("~", "projects", "model_covariance" , "data", "iris.csv") )
    # Xdata = CSV.read( joinpath( project_directory, "data", "iris.csv"), DataFrame )
    Xdata = RDatasets.dataset("datasets", "iris")
    sps = Xdata.Species

    if subset_data > 0
        index = shuffle(1:size(Xdata,1))[1:subset_data]
        Xdata = Xdata[index,:]
        sps = sps[index]
    end

    X = Xdata[:, 1:4]
    vn = names(X)

    if nonlinear
        # non-linearize data to demonstrate ability of GPs to deal with non-linearity
        X[:, 1] = 0.5 * X[:, 1] .^ 2 + 0.1 * X[:, 1] .^ 3
        X[:, 2] = X[:, 2] .^ 3 + 0.2 * X[:, 2] .^ 4
        X[:, 3] = 0.1 * exp.(X[:, 3]) - 0.2 * X[:, 3] .^ 2
        X[:, 4] = 0.5 * (X[:, 4]).^ 2 + 0.01 * X[:, 4].^ 5
    end
    
    X = Matrix(X)
    nData = size(X, 1)

    if center
        X = X .- mean(X, dims=1)
    end
 
    if scale
        X = X ./ std(X, dims=1)
    end

    # fake covariates
    G = zeros(nData, 2)
    G[:,1] = rand(Poisson(10), nData)
    G[:,2] = rand(Poisson(20), nData)
    
    id = recode(unwrap.( sps ), "setosa"=>1, "versicolor"=>2, "virginica"=>3)
    
    if obs=="cols"
        X = X'  # [[NOTE:: X is transposed relative to ecology]]
        G = G'
    end
    
    return id, sps, X, G, vn 
end

 
 
function pca_standard(X; model="cor", C=0, G=0, obs="rows", center=true, scale=false, cthreshold = 0.005)
    # https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
    # https://math.stackexchange.com/questions/3869/what-is-the-intuitive-relationship-between-svd-and-pca
    
    if obs == "cols" 
        # X is standard form with rows as observations and cols as features
        X = X'
        G = G'
    end

    if center
        X = X .- mean(X, dims=1)
    end
 
    if scale
        X = X ./ std(X, dims=1)
    end


    if C == 0
        if model=="cor"
            C = cor(X)
        elseif model=="cor_pairwise"
            # X must be quantile normalized zscore in (0,1) = "qscore"
            X[ findall(!isfinite, X) ] .= NaN
            X[ findall(X .<= cthreshold) ] .= NaN
            nvar = size(X, 2)
            C = zeros( nvar, nvar)
            for i in 1:nvar
                for j in 1:nvar
                    x = vec(X[:,i])
                    y = vec(X[:,j])
                    k = findall( isfinite.(x) .& (isfinite.(y)) )
                    if length(k) > 3 
                        C[i,j] = cor( x[k], y[k] )
                    end
                end
            end
            C[findall(!isfinite, C)] .= 0.0
            X[findall(!isfinite, X)] .= 0.0
        elseif model=="cov"
            C = cov(X)
        elseif model=="spearman"
            C = StatsBase.corspearman(X)
        elseif model=="kendall"
            C = StatsBase.corkendall(X)
        elseif model=="partialcor"
            nvar = size(X, 2)
            C = zeros( nvar, nvar) 
            for i in 1:nvar
                for j in 1:nvar
                    C[i,j] = StatsBase.partialcor( vec(X[:,i]), vec(X[:,j]), G)
                end
            end
        elseif model=="direct_svd"
            # do svd directly upon X .. note operating on C gives more control X = USV^T
            n = size(X, 1)
            E = svd(X ./ sqrt(n-1) )  
            evals = E.S.^2  # yes this is squared vs when using C (below)
            # PC_raw = E.U .* E.S'  # "raw pcscores"
            print( "\nOperating directly of data matrix ... amounts to a pca on a covariance matrix as C\n\n")

    end
    end
   
    if C == 0
        # if no valid covariance / correlation matrix just use and return a covariance matrix
        C = cov(X) 
        print( "\nValid correlation not found nor provided ... returning covariance matrix as C\n\n")
    end
    
    if model != "direct_svd"
        # if not a direct svd on data but rather on a cov or corr matrix
        E = svd( C )
        evals = E.S  # yes., this is correct (unlike svd(X ./sqrt(n-1)) .. it is a sqrt operation )
    end
    
    # other stats
    evecs = sign_convention( E.V ) # eigenvectors also known as "rotation"  
    evecs = E.V  # eigenvectors also known as "rotation"  
    
    variancepct = round.(evals ./ sum(evals) .* 100.0; digits=2)
    pcloadings = evecs * Diagonal(sqrt.(evals)) # weight for linear combinations (coeff); '(unstandardized) loadings' in MultivariateStats.jl
    pcscores = X * evecs  # ie. unscaled by eigenvalues ... more standard / common form 
    # pcscores = X * evecs * Diagonal(sqrt.(evals))  # NOTE this is done in R:factominer

    return evecs, evals, pcloadings, variancepct, C, pcscores
 
end

 
function biplot(; pcscores, pcloadings, evecs, evals, vn, variancepct,
    id=nothing, grps=nothing, cols=["orange", "green", "grey", "purple", "blue"], 
    x=1, y=2, type="unstandardized", obs="rows", 
    legend_location=:topright )
    
    # see: https://stats.stackexchange.com/questions/141085/positioning-the-arrows-on-a-pca-biplot
  
    if obs == "cols" 
        pcscores = pcscores'
        G = G'
    end

    pctx = round(variancepct[x], digits=1)
    pcty = round(variancepct[y], digits=1)

    n = size(pcscores,1)
   
    if type =="unstandardized"  
        pcscores = pcscores
        pcloads = pcloadings
    end

    if type =="standardized"  # to unit variance
        # X = pcscores / evecs
        pcscores = pcscores / evecs * inv(pcloadings)'
        pcloads = pcloadings
    end

    if isnothing(id) 
        h = plot( pcscores[:,x], pcscores[:,y], 
        markercolor="orange", 
        markerstrokewidth=0,
        seriesalpha=0.6, 
        seriestype=:scatter,
        alpha=0.5,
        xlabel="PC$x", ylabel="PC$y",
        framestyle=:box, 
        legend=legend_location,
        aspect_ratio = :equal ) 
    else
        h = plot( pcscores[:,x], pcscores[:,y], 
        group=grps[id],
        markercolor=cols[id], 
        markerstrokewidth=0,
        seriesalpha=0.6, 
        seriestype=:scatter,
        alpha=0.5,
        xlabel="PC$x $pctx%", ylabel="PC$y $pcty%",
        framestyle=:box, 
        legend=legend_location,
        aspect_ratio = :equal 
    )
    end

    nvars = length(vn)
    for i=1:nvars; 
        plot!([0, pcloads[i,1]], [0, pcloads[i,2]], arrow=true, label=vn[i],aspect_ratio = :equal  ); 
    end

    display(h)

end

 
 

Turing.@model function pPCA(X; nData=size(X, 1), nvar=size(X, 2), nz=2 ) 
    # means are zero (centered) .. unless there is latent observation bias/error then add
    z ~ filldist(Normal(), nData, nz)       # latent variable (pcscores)
    w ~ filldist(Normal(), nz, nvar)     # actually w'
    m ~ filldist(Normal(), 1, nvar)   
    X ~ arraydist(Normal.( z * w .+ m, 1))
end



Turing.@model function pPCA(X, G;  nData=size(X, 1), nvar=size(X, 2), nz=2, ng=size(G,2) ) 
    # confounding factors or "batch effects" due to eg, a different measurement methods, covariates, etc 
    z ~ filldist(Normal(), nData, nz)       # latent variable (pcscores)
    w ~ filldist(Normal(), nz, nvar)     # actually w'
    m ~ filldist(Normal(), 1, nvar)   
    # covariate (confounder) vector
    w_G ~ filldist(Normal(), ng)   
    X ~ arraydist(Normal.( z * w  .+ G * w_G' .+ m , 1.0) )
end;


Turing.@model function pPCA_ARD(X, ::Type{T}=Float64; nData=size(X, 1), nvar=size(X, 2)  ) where {T}
    # Automatic Relevance Determination (ARD) 
    # ID importance of dimensions 
    # A prior over the factor loadings W 
    # \alpha is a precision hyperparameter 
    # such that smaller values correspond to more important components.
    alpha ~ filldist(Gamma(1.0, 1.0), nvar)     # weight on loadings with Automatic Relevance Determination  
    z ~ filldist(Normal(), nData, nvar)       # latent variable (pcscores)
    zm = zeros(T, nvar, nvar)
    alp = 1.0 ./ sqrt.(alpha') 
    w ~ arraydist(Normal.(zm, alp))     # actually w'
    m ~ filldist(Normal(), 1, nvar)   
    tau ~ Gamma(1.0, 1.0)
    X ~ arraydist(Normal.( z * w  .+ m, 1.0 / sqrt(tau)))
end;


 
## remainder are householder variation copied and modified from
# https://github.com/jae0/HouseholderBPCA/blob/master/ubpca_improved.ipynb
# source: https://github.com/jae0/HouseholderBPCA/blob/master/py_stan_code/ppca_house_improved.stan


function sign_convention(U)
    nz,_ = size(U)
    for k in 1:nz
        if U[1,k] < 0
            U[:,k] *= -1.0
        end
    end
    return U
end
 

function lower_triangle( v, nvar, nz, ltri=tril!(trues(nvar, nz)) )
    v_mat = zeros(Real, nvar, nz)
    v_mat[ltri] .= v
    return v_mat
end

   
function eigenvector_to_householder(U, nvar=size(U,1), nz=2, ltri=tril!(trues(nvar, nz)) )  
    
    # to convert eigenvectors U to householder transformed v (for initializing Turing) 
    vs = Vector{eltype(U)}[] 
    v_mat = zeros(eltype(U), nvar, nz)

    for q in 1:nz
        v = U[q:end, q:end]
        push!(vs, v[:,1] )
        v_mat[q:nvar, q] = vs[q] 
        q == nvar && break
        nvsq = size(vs[q], 1)   
        sgn = sign(vs[q][1])
        u = vs[q] .+ sgn*(norm(vs[q])) * I(nvsq)[1,:]
        u = u ./ norm(u)
        H = Matrix(Float64.(I(nvar) ))
        G = I - 2.0 .* dot(u, u) * (u * u')
        j = (nvar-nvsq + 1) : nvar
        H[j, j] .= -1.0 .* sgn .* G 
        U = H * U
    end

    return v_mat[ltri]   
end


function householder_to_eigenvector(v_mat, nvar, nz )
    
    for q in 1:nz
        v_mat[:, q] = v_mat[:, q] ./ norm(v_mat[:, q])  
    end
  
    U = Diagonal(ones(nvar))
  
    for q in 1:nz
        k = nz - q + 1
        u = v_mat[:,k]
        sgn = sign(u[k])
        u[k] += sgn
        H = I - (2.0 / dot(u, u) * (u * u'))
        j = k:nvar
        H[j, j] = -1.0 * sgn .* H[j, j]
        U = H * U
    end 
    return U[:,1:nz]
  end



@memoize function householder_transform(v, nvar, nz, ltri, pca_sd, pca_pdef_sd, noise) 
    v_mat = lower_triangle(v, nvar, nz, ltri)
    U = householder_to_eigenvector(v_mat, nvar, nz ) 
    W = U * Diagonal(pca_sd)  
    Kmat = W * W' + (pca_pdef_sd^2 + noise) * I(nvar)
    r = sqrt.(mapslices(norm, v_mat[:,1:nz]; dims=1))
    return Kmat, r, U
end
    

function householder_to_eigenvector_nuts(v_mat, nvar=size(v_mat,1), nz=size(v_mat,2) )
    # redundant . to remove
    vm = zeros(eltype(v_mat), nvar, nz)
    for q in 1:nz
        vm[:, q] = v_mat[:, q] ./ norm(v_mat[:, q])  
    end

    U = zeros(eltype(v_mat), nvar, nvar)
    U += Diagonal(ones(nvar))

    for q in 1:nz
        k = nz - q + 1
        u = vm[:,k]
        sgn = sign(u[k])
        u[k] += sgn
        H = I - (2.0 / dot(u, u) * (u * u'))
        j = k:nvar
        H[j, j] = -1.0 * sgn .* H[j, j]
        U = H * U
    end 
    return U[:,1:nz]
end


Turing.@model function ppca_basic( Y, ::Type{T}=Float64) where {T}
    # latent PCA householder transform    
    # Note pca_sd's are small as Y is scaled (to unit std) and centered to 0 .. max value is nvar 
     
    pca_sd ~ Bijectors.ordered( arraydist( LogNormal.(sigma_prior, 1.0)) )  
    # minimum(pca_sd) < noise && Turing.@addlogprob! Inf  
    # maximum(pca_sd) > nvar && Turing.@addlogprob! Inf 
    pca_pdef_sd ~ LogNormal(0.0, 0.5)
    v ~ filldist(Normal(0.0, 1.0), nvh )
    Kmat, r, _ = householder_transform(v, nvar, nz, ltri, pca_sd, pca_pdef_sd, noise)
    # soft priors for r 
    # new .. Gamma in stan is same as in Distributions
    r ~ filldist(Gamma(0.5, 0.5), nz)
    Turing.@addlogprob! sum(-log.(r) .* iz)
    Turing.@addlogprob! -0.5 * sum(pca_sd.^ 2) + (nvar-nz-1) * sum(log.(pca_sd)) 
    Turing.@addlogprob! sum(log.(pca_sd[hindex[:,1]].^ 2) .- pca_sd[hindex[:,2]].^ 2)
    Turing.@addlogprob! sum(log.(2.0 .* pca_sd))
    Y ~ filldist(MvNormal( Symmetric(Kmat)), nData )  # latent factors
  end
  
  
 
PCA_BH_indexes = function(nvar, nz) 
    # indices used withing PCA_BH precomputed to speed things up 
    nvh = Int(nvar*nz - nz*(nz-1)/2)
    iz = float.(nvar .- collect(1:nz))

    hindex = Vector{Int}[]
    for i in 1:nz
        for j in (i + 1):nz
            push!(hindex,[nz-i+1, nz-j+1] )
        end
    end
    hindex = reduce(hcat, hindex)'
    ltri = tril!(trues(nvar, nz)) 
    return nvh, hindex, iz, ltri
end

 

Turing.@model function PCA_BH(Y, noise=1e-9, ::Type{T}=Float64 ) where {T}
  
    # householder, modified to be all in one
    pca_pdef_sd ~ LogNormal(0.0, 0.5)

    # currently, Bijectors.ordered is broken, revert for better posteriors once it works again
    # pca_sd ~ MvLogNormal(MvNormal(ones(nz) ))
    
    pca_sd ~ Bijectors.ordered( MvLogNormal(MvNormal(ones(nz) )) )  
    # pca_sd ~ Bijectors.ordered( MvLogNormal(MvNormal(sigma_prior, repeat([0.5], nz))) )  
     
    v ~ filldist(Normal(0.0, 1.0), nvh )
    
    v_mat = lower_triangle(v, nvar, nz, ltri) 
    U = householder_to_eigenvector(v_mat, nvar, nz ) 
    W = U * Diagonal(pca_sd)  
    Kmat = W * W' + (pca_pdef_sd^2 + noise) * I(nvar)

    r = sqrt.(mapslices(norm, v_mat[:,1:nz]; dims=1))
    r ~ filldist(Gamma(100.0, 100.0), nz)
    Turing.@addlogprob! sum(-log.(r) .* iz)

    if minimum(pca_sd) < noise 
        Turing.@addlogprob! Inf
        return
    end

    Turing.@addlogprob! -0.5 * sum(pca_sd .^ 2) + (nvar - nz - 1) * sum(log.(pca_sd))
    # for i in 1:nz
    #     for j in (i + 1):nz
    #         Turing.@addlogprob! log(pca_sd[nz - i + 1]^2) - pca_sd[nz - j + 1]^2
    #     end
    # end
    Turing.@addlogprob! sum(log.(pca_sd[hindex[:,1]].^ 2) .- pca_sd[hindex[:,2]].^ 2)

    Turing.@addlogprob! sum(log.(2.0 * pca_sd))

    # if Turing had a mVN based on cholesky, the following would speed things up .. maybe one day
    # L = LinearAlgebra.cholesky(Symmetric(Kmat)).L
    # L_full = zeros(T, nvar, nvar)
    # L_full += L * transpose(L)
    # # fix numerical instability (non-posdef matrix)
    # for d in 1:nvar
    #     for k in (d + 1):nvar
    #         L_full[d, k] = L_full[k, d]
    #     end
    # end
    # Y ~ filldist(MvNormal( Symmetric(L_full)), nData )
    
    Y ~ filldist(MvNormal( Symmetric(Kmat)), nData )

    return 
end

    
    
 
   

function PCA_posterior_samples( res, X; nz=size(X,1), obs="rows", model_type="default"  )
    # expect X to have
    
    if obs == "cols" 
        X = X'
    end

    nData, nvar  = size(X)
    n_samples,_,_ = size(res) 
    
    # eigenvalues
    pca_sd = Array(posterior_samples(res, sym=:pca_sd))
    
    # determine sort order from pca_sd/eval
    sigma_mean = posterior_summary(res, sym=:pca_sd, stat=:mean, dims=(1, nz))
    sj = sortperm(vec(sigma_mean), rev=true )  # high var first
    
    sigma_mean = sigma_mean[:,sj]
    pca_sd = pca_sd[:,sj]
    evals = pca_sd.^2.0
    
    # eigenvectors
    if model_type =="default"

        # pca_sd = Array( posterior_samples(res, sym=:pca_sd ) )
        # scores_mean = DataFrame( convert(Array{Float64}, mean(pcscores, dims=3)[:,:,1])', :auto)
        # rename!(sigma_mean, Symbol.(["pc" * string(i) for i in collect(1:nz)]))

    end

    if model_type =="householder"
        vv =  Array( posterior_samples(res, sym=:v ) ) 
        evecs = zeros(Float64, n_samples, nvar, nz)
        Threads.@threads for i in 1:n_samples
            v_mat = lower_triangle( vv[i,:], nvar, nz, ltri)
            evecs[i,:,:] = householder_to_eigenvector( v_mat, nvar, nz )
        end
        evecs = evecs[:,:,sj]
    end

    # "loadings"
    # pcloadings = evecs * Diagonal(sqrt.(evals)) # weight for linear combinations (coeff); '(unstandardized) loadings' in MultivariateStats.jl
    pcloadings = Array{Float64}(undef, n_samples, nvar, nz )
    Threads.@threads for i in 1:n_samples
        pcloadings[i, :, :] = evecs[i,:, :] * Diagonal(pca_sd[i,:])
    end

    # pcscores:  pcscores = X * evecs
    pcscores = Array{Float64}(undef, n_samples, nData, nz )
    Threads.@threads for i in 1:n_samples
        pcscores[i, :, :] = X * evecs[i,:, :]
    end

    return pca_sd, evals, evecs, pcloadings, pcscores 

end



function PCA_posterior_samples_vi( res, X, vns; nz=size(X,1), obs="rows", model_type="default", n_samples=1000  )
    # expect X to have
    
    if obs == "cols" 
        X = X'
    end

    nData, nvar  = size(X) 
    res_samples = Array( rand( res, n_samples ) ) # sample via simulation
    vns = String.(vns)

    # eigenvalues
    pca_sd = res_samples[findall(contains("pca_sd"), vns), :]'
    
    # determine sort order from pca_sd/eval
    sigma_mean, _, _, sigma_sd, _ = summarize_samples( pca_sd )

    sj = sortperm(vec(sigma_mean), rev=true )  # high var first
    sigma_mean = sigma_mean[:,sj]
    sigma_sd = sigma_sd[:,sj]

    pca_sd = pca_sd[:,sj]
    evals = pca_sd.^2.0
    
    # eigenvectors
    if model_type =="default"

        # pca_sd = Array( posterior_samples(res, sym=:pca_sd ) )
        # scores_mean = DataFrame( convert(Array{Float64}, mean(pcscores, dims=3)[:,:,1])', :auto)
        # rename!(sigma_mean, Symbol.(["pc" * string(i) for i in collect(1:nz)]))

    end

    if model_type =="householder"
        vv = res_samples[findall(contains("v"), vns), :]'
        evecs = zeros(Float64, n_samples, nvar, nz)
        Threads.@threads for i in 1:n_samples
            v_mat = lower_triangle( vv[i,:], nvar, nz, ltri)
            evecs[i,:,:] = householder_to_eigenvector( v_mat, nvar, nz )
        end
        evecs = evecs[:,:,sj]
    end

    # "loadings"
    # pcloadings = evecs * Diagonal(sqrt.(evals)) # weight for linear combinations (coeff); '(unstandardized) loadings' in MultivariateStats.jl
    pcloadings = Array{Float64}(undef, n_samples, nvar, nz )
    Threads.@threads for i in 1:n_samples
        pcloadings[i, :, :] = evecs[i,:, :] * Diagonal(pca_sd[i,:])
    end

    # pcscores:  pcscores = X * evecs
    pcscores = Array{Float64}(undef, n_samples, nData, nz )
    Threads.@threads for i in 1:n_samples
        pcscores[i, :, :] = X * evecs[i,:, :]
    end

    return pca_sd, evals, evecs, pcloadings, pcscores 
 
end



Turing.@model function PCA_linearGP(X, ::Type{T}=Float64; N=size(X,1), nvar=size(X,2), nz=2, noise= 1e-9 ) where {T}
    # Priors
    alpha ~ MvLogNormal(MvNormal(zeros(nz), I))
    z ~ filldist(Normal(), nz, N)
    mu ~ filldist(Normal(), N)
    kernel = linear_kernel(alpha)
    gp = GP(mu, kernel)
    cv = cov(gp(ColVecs(z), noise))
    return X ~ filldist(MvNormal(mu, Symmetric(cv)), nvar)
end;


Turing.@model function PCA_nonlinearGP(X, ::Type{T}=Float64; N=size(X,1), nvar=size(X,2), nz=2, noise= 1e-9 ) where {T}
    # Priors
    alpha ~ MvLogNormal(MvNormal(zeros(nz), I))
    pca_sd ~ LogNormal(0.0, 1.0)
    z ~ filldist(Normal(), nz, N)
    mu ~ filldist(Normal(), N)
    kernel = gpkernel(alpha, pca_sd)
    gp = GP(mu, kernel)
    cv = cov(gp(ColVecs(z), noise))
    return X ~ filldist(MvNormal(mu, Symmetric(cv)), nvar)
end;


Turing.@model function PCA_sparseGP(X, ::Type{T}=Float64 ; nvar=size(X,2), D=size(X,1), nz=2, noise = 1e-3,  n_inducing=5) where {T}

    # Priors
    α ~ MvLogNormal(MvNormal(zeros(nz), I))
    σ ~ LogNormal(1.0, 1.0)
    Z ~ filldist(Normal(), nz, nvar)
    mu ~ filldist(Normal(), nvar)

    kernel = σ * SqExponentialKernel() ∘ ARDTransform(α)

    ## Standard
    # gpc = GPC()
    # f = atomic(GP(kernel), gpc)
    # gp = f(ColVecs(Z), noise)
    # X ~ filldist(gp, D)

    ## SPARSE GP
    #  xu = reshape(repeat(locations, nz), :, nz) # inducing points
    #  xu = reshape(repeat(collect(range(-2.0, 2.0; length=20)), nz), :, nz) # inducing points
    #lbound = minimum(X) + 1e-6
    #ubound = maximum(X) - 1e-6
    #  locations ~ filldist(Uniform(lbound, ubound), n_inducing)
    #  locations = [-2., -1.5 -1., -0.5, -0.25, 0.25, 0.5, 1., 2.]
    #  locations = collect(LinRange(lbound, ubound, n_inducing))
    locations = quantile(vec(X), LinRange(0.01, 0.99, n_inducing))
    xu = reshape(locations, 1, :)
    gp = atomic(GP(kernel), GPC())
    fobs = gp(ColVecs(Z), noise)
    finducing = gp(xu, 1e-12)
    sfgp = SparseFiniteGP(fobs, finducing)
    cv = cov(sfgp.fobs)
    return X ~ filldist(MvNormal(mu, cv), D)
end
