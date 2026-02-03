# automatic load of things related to all projects go here
 
current_directory =  @__DIR__() 
print( "Current directory is: ", current_directory, "\n\n" )

if !@isdefined project_directory 
    project_directory = joinpath( homedir(), "projects", "model_covariance" )
end

import Pkg  # or using Pkg
Pkg.activate(project_directory)  # so now you activate the package
Base.active_project()  
push!( LOAD_PATH, project_directory )  # add the directory to the load path, so it can be found

pkgs = [  
    "Revise", "Memoization", "BenchmarkTools", "OhMyREPL",
    "DataFrames", "CSV", "RData", "RDatasets", "JLD2", "ParameterHandling",  # "ArviZ", 
    "StatsBase", "Statistics", "MultivariateStats", "LinearAlgebra", "Distributions", "Random", 
    "StatsModels", "StatsFuns", 
    "StaticArrays", "FillArrays",  "SparseArrays", "Graphs", "Distances", "CategoricalArrays",
    "PlotThemes", "Colors", "ColorSchemes", "Plots", "StatsPlots",
    "MKL", "PDMats", "Optim", 
    "ADTypes",  "ForwardDiff",
    "AdvancedVI",  "Turing", "Bijectors",
    "Stheno", "KernelFunctions", "AbstractGPs",  "ApproximateGPs", "LogExpFunctions"
]

# using ApproximateGPs, Random, "CodeTracking",   "Setfield", "ParameterHandling" 
#   "AdvancedHMC", "DynamicHMC", "DistributionsAD", "Bijectors", "Libtask", "ReverseDiff", 
    # "Symbolics", "Logging", 
   
# load directly can cause conflicts due to same function names 
pkgtoskipload = [  "RCall",   "CairoMakie", "PlotlyJS",  "PlotlyBase",  "PlotlyKaleido", "LazyArrays" ]
 
print( "Loading libraries:\n\n" ) 
 
for pk in pkgs; 
    if !(Base.find_package(pk) === nothing)
        if !(pk in pkgtoskipload)
            @eval using $(Symbol(pk)); 
        end
    end
end

include( srcdir( "shared_functions.jl") )


print( "\nTo (re)-install required packages, run:  install_required_packages() or Pkg.instantiate() \n\n" ) 
 


DEBUG = Ref{Any}()

# add this inside of a function to track vars
# Main.DEBUG[] = y,p,t
