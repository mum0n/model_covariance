# automatic load of things related to all projects go here

# load environment, libraries  
using DrWatson  # install this if you do not have it

if !@isdefined project_directory 
    project_directory = joinpath( homedir(), "projects", "model_covariance" )
end

quickactivate(project_directory) 
 
current_directory =  @__DIR__() 
print( "Current directory is: ", current_directory, "\n\n" )
 
import Pkg  # or using Pkg
Pkg.activate(project_directory)  # so now you activate the package
Base.active_project()  
push!( LOAD_PATH, project_directory )  # add the directory to the load path, so it can be found

pkgs = [  
    "Revise", "Memoization", "BenchmarkTools", "OhMyREPL",
    "DataFrames", "CSV", "RData", "RDatasets", "JLD2", "ParameterHandling",  # "ArviZ", 
    "StatsBase", "Statistics", "MultivariateStats", "LinearAlgebra", "Distributions", "Random", "StatsAPI", 
    "StatsModels", "StatsFuns", "GLM", "Tables",
    "StaticArrays", "FillArrays",  "SparseArrays", "Graphs", "Distances", "CategoricalArrays",
    "PlotThemes", "Colors", "ColorSchemes", "Plots", "StatsPlots",
    "MKL", "PDMats", "Optim", "Peaks", "KernelDensity", "Interpolations", 
    "ADTypes",  "ForwardDiff", 
    "AdvancedVI",  "Turing", "Bijectors", "ArchGDAL",
    "KernelFunctions", "AbstractGPs",  "ApproximateGPs", "LogExpFunctions", "TemporalGPs"
]


# using "CodeTracking",  "Setfield",  "AdvancedHMC", "DynamicHMC", "DistributionsAD",   "Libtask", "ReverseDiff"  
    # "Symbolics", "Logging",  
 
# load directly can cause conflicts due to same function names 
pkgtoskipload = [   "CairoMakie", "PlotlyJS",  "PlotlyBase",  "PlotlyKaleido", "LazyArrays" ]
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


include( srcdir( "shared_functions.jl") )
include( srcdir( "simple_linear_regression.jl") )
include( srcdir( "regression_functions.jl" ))   # support functions  

include( srcdir( "car_functions.jl" ))   # support functions  
include( srcdir( "example_data.jl" ))     


# Set a seed for reproducibility.
Random.seed!(42)
