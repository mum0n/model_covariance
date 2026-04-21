# automatic load of things related to all projects go here

using Pkg
Pkg.add( ["DrWatson", "Revise", "Requires", "PrecompileTools", "PackageCompiler"] )

using DrWatson, Revise, Requires, PrecompileTools, PackageCompiler

if !@isdefined project_directory 
    project_directory = joinpath( homedir(), "projects", "model_covariance" )
end

quickactivate(project_directory) 

# the following are now handled by DrWatson.quickactivate()
# Pkg.activate(project_directory)  # so now you activate the package
# Base.active_project()  
# push!( LOAD_PATH, project_directory )  # add the directory to the load path, so it can be found

current_directory =  @__DIR__() 
print( "Current directory is: ", current_directory, "\n\n" )

pkgs = unique( [  
    "DrWatson", "Revise", "Requires", 
    "Random", "Statistics", "LinearAlgebra", "DataFrames",
    "StatsBase", "SparseArrays", "Plots",
    "JLD2", "LibGEOS", "Graphs", "DelaunayTriangulation",
    "Random", "Turing", "Distributions", "Statistics", "MCMCChains", "DataFrames",
    "LinearAlgebra", "Clustering", "StatsBase", "LogExpFunctions",
    "JLD2", "FFTW",  "SparseArrays", "StaticArrays", "FillArrays",
    "Bijectors", "DynamicPPL", "AdvancedVI", "Optimisers", "PosteriorStats", "HypothesisTests",
    "CurlHTTP",
    "PrecompileTools", "PackageCompiler","Memoization", "BenchmarkTools", "OhMyREPL",
    "DataFrames", "CSV", "JLD2", "Tables",
    "PalmerPenguins", 
    #"ForecastData", 
    "StatsBase", "Statistics", "MultivariateStats", "LinearAlgebra", "Distributions", "Random", "StatsAPI", "StatsModels", "StatsFuns", "GLM", 
    "StaticArrays", "FillArrays",  "SparseArrays",#"LazyArrays", 
    #"ParameterHandling",  
    # "ArviZ", 
    "Graphs", "PlotThemes", "Colors", "ColorSchemes", "Plots", "StatsPlots", 
    "Distances", "CategoricalArrays",
    "MKL", "PDMats", 
    "Optim", "Flux",  
    "Peaks", "KernelDensity", "DSP", "Interpolations", 
    "ADTypes",  "ForwardDiff",  
    "JLD2", "DelaunayTriangulation",
    "PolygonOps", "GeoInterface", "StatsPlots",
    "MCMCChains",  
    "Clustering",  
    "JLD2", "FFTW",  "HypothesisTests",
    "Bijectors", "DynamicPPL", "AdvancedVI", "Optimisers",
    "AdvancedVI",  "Turing", "Bijectors", "ArchGDAL",
    "KernelFunctions", "AbstractGPs",  "ApproximateGPs", "LogExpFunctions", "TemporalGPs"
] )



# "Lux", "ArchGDAL", 

# using "CodeTracking",  "Setfield",  "AdvancedHMC", "DynamicHMC", "DistributionsAD",   "Libtask", "ReverseDiff"  
    # "Symbolics", "Logging",  
 
# load directly can cause conflicts due to same function names 
pkgtoskipload = [ "CairoMakie", "PlotlyJS"   ]
  # "RCall",
   
print( "Loading libraries:\n\n" ) 

# For RCall:
if Sys.iswindows()
    # ENV["R_HOME"] = "C:\Program Files\R\R-4.5.2\bin\x64\Rgui.exe"
    ENV["R_HOME"] = "C:\\Program Files\\R\\R-4.5.2"
    ENV["path"] = string( ENV["R_HOME"], "\\bin\\x64; ", ENV["path"] )
    #    using Pkg; Pkg.build("RCall")
elseif Sys.islinux()
elseif Sys.isapple()
else
end


for pk in pkgs; 
    if (Base.find_package(pk) === nothing)
        Pkg.add(pk)
        @eval using $(Symbol(pk))
    else
        if !(pk in pkgtoskipload)
            @eval using $(Symbol(pk)); 
        end
    end
end



print( "\nTo (re)-install required packages, run:  install_required_packages() or Pkg.instantiate() \n\n" ) 
  
# to help track variables, add something like this inside of a function:  
# Main.DEBUG[] = y,p,t  # this stores y, p, t into Main.DEBUG 
DEBUG = Ref{Any}()  # initiate

# support functions
include( srcdir( "example_data.jl" ))     
include( srcdir( "shared_functions.jl") )
include( srcdir( "simple_linear_regression.jl") )
include( srcdir( "regression_functions.jl" ))     
include( srcdir( "gaussian_processes_functions.jl" ))     

include( srcdir( "car_functions.jl" ))   
include( srcdir( "carstm_functions.jl" ))   
include( srcdir( "spatiotemporal_functions.jl" ))   
# include( srcdir( "fft_functions.jl" ))   


# Set a seed for reproducibility.
Random.seed!(42)
