# Example work flow from R calling model_covariance functions

Following is based upon examples here: 

https://cran.r-project.org/web/packages/JuliaCall/readme/README.html

https://hwborchers.github.io/


```{r}

if (0) {
    # to set up:
    # julia_setup(JULIA_HOME = "the folder that contains julia binary")
    # options(JULIA_HOME = "the folder that contains julia binary")
    # Set JULIA_HOME in command line environment.
    install.packages("JuliaCall")
    install_julia() # only if you do not have it installed
}

# load JuliaCall interface
library(JuliaCall)
julia <- julia_setup()
J = julia_command  # copy to shorten following text calls

# set up paths: adjust to local copy of github.com/jae0/pca 
project_directory = file.path( "/", "home", "jae", "projects", "model_covariance"  )

# load pca functions
# julia_source cannot traverse directories .. temporarily switch directory
currwd = getwd() 

    julia_assign( "project_directory", project_directory )
    julia_assign( "src_directory", joinpath(project_directory, "src") )

    setwd(src_directory) 
    julia_source( "startup.jl" )  # this might need to re-run a few times if this is the first time being used 

    # J( "using  LazyArrays" )  # having issues in JuliaCall 
    julia_source( "pca_functions.jl" ) 
    julia_source( "regression_functions.jl" ) 
    julia_source( "car_functions.jl" ) 

setwd(currwd) # revert


# Example data: replicate iris analysis S
# Y = scale(iris[, 1:4])  # now part of the pca_functions.jl
# julia_assign("X", t(Y) )  # copy data into julia session

# set up problem parameters and conduct basic PCA to get initial settings
J( 'id, sps, X, G, vn = iris_data(scale=false)' )
J( 'nx, nvar = size(X); nz = 2' ) # nz is number of (latent) factors 
J( 'evecs, evals, pcloadings, variancepct, C, PC = pca_standard(X; model="cor", obs="rows") ' ) # return values: eigenvectors, sqrt(eigenvalues), correlation matrix, pc scores
J( 'v = eigenvector_to_householder(evecs, nz=nz )  ' )  # v is the householder representation of the eigenvectors used by the householder pca

# param sequence = sigma_noise, sigma(nz), v, r=norm(v)~ 1.0 (scaled)
J( 'init_params = [0.1; sqrt.(evals)[1:nz]; v; 1.0 ]' )
J( "M = PCA_BH(X', nz=nz)" )  # all dims == default form
J( 'n_samples=100' )
J( 'res = sample(M, Turing.NUTS(), n_samples; init_params=init_params)' )
J( 'posterior_summary(res, sym=:sigma, stat=:mean, dims=(1, nz))' )   
J( 'sigma, evals, evecs, loadings, scores = PCA_posterior_samples( res, X, nz=nz, model_type="householder" )' ) 
 
# to move data into R
sigma = julia_eval('sigma')  # standard deviations
scores = julia_eval('scores')  # pc scores
eigenvectors = julia_eval('evecs') # weights

# save as it seems JuliaCall alters plotting environment
# fn = file.path("~", "tmp", "pca_scores_posteriors.rdz" )
# read_write_fast( data=scores, fn=fn )

# in an alternate R-session, load results and plot
# fn = file.path("~", "tmp", "pca_scores_posteriors.rdz" )
# scores = aegis::read_write_fast(fn)
sp1 = 1:50
sp2 = 51:100
sp3 =101:150
plot( scores[,,2] ~ scores[,,1], type="n" )
points( scores[,sp1,2] ~ scores[,sp1,1], col="red", pch="."  )
points( scores[,sp2,2] ~ scores[,sp2,1], col="blue", pch="."  )
points( scores[,sp3,2] ~ scores[,sp3,1], col="grey", pch="."  )

``` 
   