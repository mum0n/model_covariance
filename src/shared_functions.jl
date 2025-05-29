
function packages_used(model_variation) 

    pkgs_shared = [
        "DrWatson", "Revise", "Test",  "OhMyREPL", "Logging", 
        "StatsBase", "Statistics", "Distributions", "Random", "Setfield", "Memoization", 
        "MCMCChains", 
        "DataFrames", "JLD2", "CSV", "PlotThemes", "Colors", "ColorSchemes", "RData",  
        "Plots",  "StatsPlots", "MultivariateStats", 
        "ForwardDiff", "ReverseDiff", "Enzyme", "ADTypes",
        "StaticArrays", "LazyArrays", "FillArrays", "LinearAlgebra", "MKL", "Turing"
    ]

    if occursin( r"logistic_discrete.*", model_variation )
        pkgs = []
    elseif occursin( r"size_structured_dde.*", model_variation )
        pkgs = [
            "QuadGK", "ModelingToolkit", "DifferentialEquations", "Interpolations",
        ]
    end
    
    pkgs = unique!( [pkgs_shared; pkgs] )
  
    return pkgs  

end


function install_required_packages(pkgs)    # to install packages
    for pk in pkgs; 
        if Base.find_package(pk) === nothing
            Pkg.add(pk)
        end
    end   # Pkg.add( pkgs ) # add required packages
    print( "Pkg.add( \"Bijectors\" , version => \"0.3.16\") # may be required \n" )
end
 
function init_params_extract(X)
  XS = summarize(X)
  vns = XS.nt.parameters  # var names
  init_params = FillArrays.Fill( XS.nt[2] ) # means
  return init_params, vns
end

 
function discretize_decimal( x, delta=0.01 ) 
    num_digits = Int(ceil( log10(1.0 / delta)) )   # time floating point rounding
    out = round.( round.( x ./ delta; digits=0 ) .* delta; digits=num_digits)
    return out
end
 

function expand_grid(; kws...)
    names, vals = keys(kws), values(kws)
    return DataFrame(NamedTuple{names}(t) for t in Iterators.product(vals...))
end
   

function showall( x )
    # print everything to console
    show(stdout, "text/plain", x) # display all estimates
end 
 

function firstindexin(a::AbstractArray, b::AbstractArray)
    bdict = Dict{eltype(b), Int}()
    for i=length(b):-1:1
        bdict[b[i]] = i
    end
    [get(bdict, i, 0) for i in a]
end
   
  
function β( mode, conc )
    # alternate parameterization of beta distribution 
    # conc = α + β     https://en.wikipedia.org/wiki/Beta_distribution
    beta1 = mode *( conc - 2  ) + 1.0
    beta2 = (1.0 - mode) * ( conc - 2  ) + 1.0
    Beta( beta1, beta2 ) 
end 
  
function modelruntime(o)
    dt = ( o.info.stop_time- o.info.start_time )/ 60
    showall( summarize(o) )
    print( dt )
end
 
function code_show(x)
   # printstyled( CodeTracking.@code_string x() )
end
  

function install_required_packages()    # to install packages
    for pk in pkgs; 
        if Base.find_package(pk) === nothing
            Pkg.add(pk)
        end
    end   # Pkg.add( pkgs ) # add required packages

    print( "Pkg.add( \"Bijectors\" , version => \"0.3.16\") # may be required \n" )

end
 


function turingindex( indices, sym=nothing, dims=nothing  ) 
     
    if isa(indices, DynamicPPL.Model)
        _, indices = bijector(turing_model, Val(true));
    end

    if isnothing(sym)
      out = enumerate(keys(indices))
    elseif sym=="varnames"
      out = keys(indices)
    else
      out = union(indices[sym]...)
    end
    
    if !isnothing(dims)
        out = reshape(out, dims)
    end

    return out 
  end

