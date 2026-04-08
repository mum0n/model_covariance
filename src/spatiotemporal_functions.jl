
# Global Helper Functions
function rff_map(coords, W, b)
    projection = (coords * W) .+ b'
    return sqrt(2 / size(W, 2)) .* cos.(projection)
end

function compute_y_waic(mod, ch)
    try
        pll = pointwise_loglikelihoods(mod, ch)
        y_keys = [k for k in keys(pll) if occursin("y_obs", string(k))]
        if !isempty(y_keys)
            loglik_mat = hcat([vec(pll[k]) for k in y_keys]...)
            lppd = sum(log.(mean(exp.(loglik_mat), dims=1)))
            p_waic = sum(var(loglik_mat, dims=1))
            return -2 * (lppd - p_waic)
        end
    catch e
        return NaN
    end
    return NaN
end


function get_posterior_means(ch, param_base, N)
    means = zeros(N)
    for i in 1:N
        p_symbol = Symbol("$param_base[$i]")
        if p_symbol in names(ch, :parameters)
            means[i] = mean(ch[p_symbol])
        else
            @warn "Parameter $p_symbol not found in chain."
            means[i] = 0.0
        end
    end
    return means
end



