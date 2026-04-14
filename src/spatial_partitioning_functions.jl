
function generate_sim_data(n_pts, n_time; rndseed=42 )
    n_total = n_pts * n_time
    Random.seed!(rndseed)
    pts = [(rand() * 10, rand() * 10) for _ in 1:n_pts]
    time_idx = repeat(1:n_time, inner=n_pts)
    weights = ones(n_total)
    trials = ones(Int, n_total)
    cov_indices = rand(1:3, n_total)
    spatial_effect = [sin(p[1]/2) + cos(p[2]/2) for p in pts]
    temporal_effect = sin.(time_idx)
    spatial_effect_long = repeat(spatial_effect, n_time)
    y_sim  = spatial_effect_long + temporal_effect + randn(n_total) * 0.5
    y_binary = y_sim  .> (mean(y_sim) + 0.5)
    return (pts=pts, y_sim=y_sim, y_binary=y_binary, time_idx=time_idx,
            weights=weights, trials=trials, cov_indices=cov_indices)
end

# --- Intensity Estimation Helpers ---

function estimate_intensity_kde_optimized(pts; grid_res=50)
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    x_grid = range(0, 10, length=grid_res)
    y_grid = range(0, 10, length=grid_res)
    bw = 0.9 * min(std(xs), (quantile(xs, 0.75)-quantile(xs, 0.25))/1.34) * length(pts)^(-0.2)
    intensity = zeros(grid_res, grid_res)
    for (i, x) in enumerate(x_grid), (j, y) in enumerate(y_grid)
        intensity[i,j] = sum(exp.(-0.5 * (((x .- xs).^2 .+ (y .- ys).^2) ./ bw^2))) / (2π * bw^2)
    end
    return x_grid, y_grid, intensity
end

function estimate_intensity_gp(pts; grid_res=50, l=2.0)
    x_grid = range(0, 10, length=grid_res)
    y_grid = range(0, 10, length=grid_res)
    intensity = zeros(grid_res, grid_res)
    for (i, x) in enumerate(x_grid), (j, y) in enumerate(y_grid)
        p_g = (x, y)
        dists = [sqrt(sum((p_g .- p).^2)) for p in pts]
        val = sum((1 .+ sqrt(3) .* dists ./ l) .* exp.(-sqrt(3) .* dists ./ l))
        intensity[i,j] = val
    end
    return x_grid, y_grid, intensity
end

function generate_grid_centroids(pts, n_centroids)
    if n_centroids == 0 return [] end
    min_x, max_x = minimum(p[1] for p in pts), maximum(p[1] for p in pts)
    min_y, max_y = minimum(p[2] for p in pts), maximum(p[2] for p in pts)
    domain_width, domain_height = max_x - min_x, max_y - min_y
    aspect_ratio = (domain_height == 0) ? 1.0 : domain_width / domain_height
    rows = max(1, round(Int, sqrt(n_centroids / aspect_ratio)))
    cols = max(1, round(Int, n_centroids / rows))
    while rows * cols < n_centroids
        if rows * aspect_ratio < cols rows += 1 else cols += 1 end
    end
    x_grid = LinRange(min_x, max_x, cols); y_grid = LinRange(min_y, max_y, rows)
    grid_centers = [(x, y) for x in x_grid for y in y_grid]
    closest_pts_indices = Set{Int}()
    for gc in grid_centers[1:min(length(grid_centers), n_centroids)]
        dists = [sum((gc .- p).^2) for p in pts]
        idx = argmin(dists)
        push!(closest_pts_indices, idx)
    end
    return [pts[i] for i in collect(closest_pts_indices)]
end

function compute_hybrid_density_cvt(pts, x_g, y_g, dens, initial_centroids; iters=15)
    centroids = deepcopy(initial_centroids)
    n_seeds = length(centroids)
    if n_seeds == 0 return [], zeros(Int, length(pts)) end
    assignments = zeros(Int, length(pts))
    for _ in 1:iters
        for i in 1:length(pts)
            assignments[i] = argmin([sum((pts[i] .- s).^2) for s in centroids])
        end
        for j in 1:n_seeds
            idx = findall(==(j), assignments)
            if !isempty(idx)
                sub_pts = pts[idx]
                weights = [dens[clamp(argmin(abs.(x_g .- p[1])), 1, length(x_g)), clamp(argmin(abs.(y_g .- p[2])), 1, length(y_g))] for p in sub_pts]
                sum_w = sum(weights)
                if sum_w > 0
                    centroids[j] = (sum([p[1]*w for (p,w) in zip(sub_pts, weights)])/sum_w, sum([p[2]*w for (p,w) in zip(sub_pts, weights)])/sum_w)
                end
            end
        end
    end
    return centroids, assignments
end

function assign_spatial_units_merge_voronoi(pts::Vector{Tuple{Float64, Float64}}, n_time::Int;
                                            buffer=nothing, time_idx::Vector{Int}, min_time_slices::Int=1,
                                            min_area_constraint::Float64=0.0, max_area_constraint::Float64=Inf,
                                            min_total_arealunits::Union{Int, Nothing}=nothing,
                                            max_total_arealunits::Union{Int, Nothing}=nothing,
                                            tol=1e-4)
    n_pts = length(pts)
    x_g, y_g, dens = estimate_intensity_kde_optimized(pts)

    tri_full = triangulate(pts)
    hull = convex_hull(tri_full)
    hull_indices = DelaunayTriangulation.get_vertices(hull)
    raw_hull = pts[hull_indices]
    if isnothing(buffer) buffer = median([sqrt(sum((pts[i] .- pts[j]).^2)) for i in 1:n_pts for j in (i+1):n_pts]) * 0.25 end
    cx, cy = mean(p[1] for p in raw_hull), mean(p[2] for p in raw_hull)
    boundary_hull_final = map(raw_hull) do p
        dx, dy = p[1]-cx, p[2]-cy; mag = sqrt(dx^2+dy^2)
        (p[1]+(dx/mag)*buffer, p[2]+(dy/mag)*buffer)
    end
    push!(boundary_hull_final, boundary_hull_final[1])

    areal_units_points = Dict(i => [i] for i in 1:n_pts)
    centroids_map = Dict(i => pts[i] for i in 1:n_pts)
    target_n_units = isnothing(max_total_arealunits) ? 1 : max_total_arealunits
    current_n = n_pts
    prev_mean, prev_var = NaN, NaN

    println("Merge Voronoi (Density-Prioritized): Starting with $current_n units.")

    while current_n > target_n_units
        ids = sort(collect(keys(centroids_map)))
        pt_counts = [length(areal_units_points[id]) for id in ids]
        curr_mean, curr_var = mean(pt_counts), var(pt_counts)
        if !isnan(prev_mean) && abs(curr_mean - prev_mean)/prev_mean < tol && abs(curr_var - prev_var)/prev_var < tol
            break
        end
        prev_mean, prev_var = curr_mean, curr_var

        adj = Set{Tuple{Int,Int}}()
        if length(ids) >= 3
            tri = triangulate([centroids_map[id] for id in ids])
            for edge in each_edge(tri)
                u, v = edge
                if u > 0 && v > 0 push!(adj, ids[u] < ids[v] ? (ids[u], ids[v]) : (ids[v], ids[u])) end
            end
        else
            for i in 1:length(ids), j in (i+1):length(ids) push!(adj, (ids[i], ids[j])) end
        end

        function get_unit_density(id)
            p_indices = areal_units_points[id]
            return mean([dens[clamp(argmin(abs.(x_g .- pts[idx][1])), 1, length(x_g)), clamp(argmin(abs.(y_g .- pts[idx][2])), 1, length(y_g))] for idx in p_indices])
        end

        pairs = sort(collect(adj), by=pair -> get_unit_density(pair[1]) + get_unit_density(pair[2]))
        merged_in_pass = false

        for (u_id, v_id) in pairs
            if !haskey(areal_units_points, u_id) || !haskey(areal_units_points, v_id) continue end
            comb_pts = vcat(areal_units_points[u_id], areal_units_points[v_id])
            if length(comb_pts) <= max_area_constraint
                new_id = maximum(keys(areal_units_points)) + 1
                areal_units_points[new_id] = comb_pts
                centroids_map[new_id] = (mean(pts[i][1] for i in comb_pts), mean(pts[i][2] for i in comb_pts))
                delete!(areal_units_points, u_id); delete!(areal_units_points, v_id)
                delete!(centroids_map, u_id); delete!(centroids_map, v_id)
                current_n -= 1; merged_in_pass = true
                if current_n <= target_n_units break end
            end
        end
        if !merged_in_pass break end
    end

    final_ids = sort(collect(keys(areal_units_points))); n_final = length(final_ids)
    output_centroids = [centroids_map[id] for id in final_ids]

    # FIXED: Re-assign all original points to the nearest final centroid
    output_assign = zeros(Int, n_pts)
    for i in 1:n_pts
        output_assign[i] = argmin([sum((pts[i] .- c).^2) for c in output_centroids])
    end

    output_W = spzeros(n_final, n_final)
    if n_final >= 3
        tri_f = triangulate(output_centroids)
        for e in each_edge(tri_f)
            u, v = e; if u > 0 && v > 0 && u <= n_final && v <= n_final output_W[u,v] = 1.0; output_W[v,u] = 1.0 end
        end
    end
    return output_centroids, output_assign, Symmetric(output_W), repeat(output_assign, n_time), boundary_hull_final
end

function plot_spatial_graph(pts, assignments, centroids, W; title="Spatial Partition", show_boundaries=true, boundary_hull=nothing)
    p = scatter([pt[1] for pt in pts], [pt[2] for pt in pts], marker_z=assignments, color=:viridis, alpha=0.4, label="Data Points", title=title, aspect_ratio=:equal, markersize=3)
    if show_boundaries && !isnothing(boundary_hull) && length(centroids) >= 3
        tri = triangulate(centroids); vorn = voronoi(tri)
        min_x, max_x = minimum(pt[1] for pt in boundary_hull), maximum(pt[1] for pt in boundary_hull)
        min_y, max_y = minimum(pt[2] for pt in boundary_hull), maximum(pt[2] for pt in boundary_hull)
        plot!(p, [c[1] for c in boundary_hull], [c[2] for c in boundary_hull], color=:blue, lw=2, label="Global Boundary")
        for i in each_polygon_index(vorn)
            coords = get_polygon_coordinates(vorn, i, (min_x, max_x, min_y, max_y))
            plot!(p, [c[1] for c in coords], [c[2] for c in coords], color=:black, alpha=0.3, label="", lw=0.8)
        end
    end
    for i in 1:length(centroids), j in (i+1):length(centroids)
        if W[i,j] > 0 plot!(p, [centroids[i][1], centroids[j][1]], [centroids[i][2], centroids[j][2]], color=:red, alpha=0.6, lw=1.5, label="") end
    end
    scatter!(p, [c[1] for c in centroids], [c[2] for c in centroids], marker=:star, color=:red, markersize=8, label="Centroids")
    return p
end

struct QuadtreeNode
    boundary::Tuple{Float64, Float64, Float64, Float64} # min_x, max_x, min_y, max_y
    points::Vector{Tuple{Float64, Float64}}
    point_indices::Vector{Int}
    children::Vector{QuadtreeNode}
    is_leaf::Bool
end

function area(boundary)
    return (boundary[2] - boundary[1]) * (boundary[4] - boundary[3])
end

function build_quadtree(pts, time_idx, capacity, min_time_slices, min_area, max_area; depth=0, max_depth=10)
    min_x = minimum(p[1] for p in pts); max_x = maximum(p[1] for p in pts)
    min_y = minimum(p[2] for p in pts); max_y = maximum(p[2] for p in pts)
    initial_boundary = (min_x, max_x, min_y, max_y)
    return _build_quadtree_recursive(pts, collect(1:length(pts)), initial_boundary, capacity, depth, max_depth)
end

function _build_quadtree_recursive(pts, indices, boundary, capacity, depth, max_depth)
    if length(indices) <= capacity || depth >= max_depth
        return QuadtreeNode(boundary, [pts[i] for i in indices], indices, QuadtreeNode[], true)
    end

    mid_x = (boundary[1] + boundary[2]) / 2
    mid_y = (boundary[3] + boundary[4]) / 2

    child_boundaries = [
        (boundary[1], mid_x, boundary[3], mid_y), (mid_x, boundary[2], boundary[3], mid_y),
        (boundary[1], mid_x, mid_y, boundary[4]), (mid_x, boundary[2], mid_y, boundary[4])
    ]

    children = QuadtreeNode[]
    for cb in child_boundaries
        child_indices = filter(i -> pts[i][1] >= cb[1] && pts[i][1] <= cb[2] && pts[i][2] >= cb[3] && pts[i][2] <= cb[4], indices)
        push!(children, _build_quadtree_recursive(pts, child_indices, cb, capacity, depth + 1, max_depth))
    end

    return QuadtreeNode(boundary, [pts[i] for i in indices], indices, children, false)
end

function get_leaf_nodes(node::QuadtreeNode)
    if node.is_leaf return [node] end
    return vcat([get_leaf_nodes(c) for c in node.children]...)
end


function assign_spatial_units_quadtree(pts::Vector{Tuple{Float64, Float64}}, n_time::Int; capacity::Int=5, buffer=nothing, time_idx::Vector{Int}, min_time_slices::Int=1, min_area_constraint::Float64, max_area_constraint::Float64)
    n = length(pts)

    # Internal Hull Computation with Dynamic Buffer
    tri_full = triangulate(pts)
    hull = convex_hull(tri_full)
    hull_indices = DelaunayTriangulation.get_vertices(hull)
    raw_hull = pts[hull_indices]

    if isnothing(buffer)
        dists = Float64[]
        for i in 1:min(n, 100), j in (i+1):min(n, 100)
            push!(dists, sqrt(sum((pts[i] .- pts[j]).^2)))
        end
        buffer = isempty(dists) ? 0.5 : median(dists) * 0.25
    end

    # Apply buffer by expanding from hull centroid
    cx = mean(p[1] for p in raw_hull)
    cy = mean(p[2] for p in raw_hull)
    boundary_hull = map(raw_hull) do p
        dx = p[1] - cx; dy = p[2] - cy
        mag = sqrt(dx^2 + dy^2)
        (p[1] + (dx/mag)*buffer, p[2] + (dy/mag)*buffer)
    end
    push!(boundary_hull, boundary_hull[1]) # Close loop

    # Build Quadtree - now passing time_idx and min_time_slices and area constraints
    quadtree = build_quadtree(pts, time_idx, capacity, min_time_slices, min_area_constraint, max_area_constraint)
    all_leaf_nodes = get_leaf_nodes(quadtree)

    # Filter out empty leaf nodes and those violating min_time_slices or area constraints
    valid_leaf_nodes_filtered = QuadtreeNode[]

    for node in all_leaf_nodes
        num_points_in_node = length(node.points)
        # Corrected: If a node has points, it has n_time unique time slices
        current_time_slices = isempty(node.point_indices) ? 0 : n_time
        node_area = area(node.boundary)

        if num_points_in_node > 0 &&
           (min_area_constraint == 0.0 || node_area >= min_area_constraint) &&
           (max_area_constraint == Inf || node_area <= max_area_constraint) &&
           (min_time_slices == 0 || current_time_slices >= min_time_slices)
            push!(valid_leaf_nodes_filtered, node)
        end
    end

    if isempty(valid_leaf_nodes_filtered)
        println("Warning: All Quadtree units violate constraints or are empty. Returning a single unit.")
        centroids = [(mean(p[1] for p in pts), mean(p[2] for p in pts))]
        assignments = ones(Int, n)
        n_arealunits = 1
        area_idx = repeat(assignments, n_time)
        W = spzeros(n_arealunits, n_arealunits)
        return centroids, assignments, Symmetric(W), area_idx, boundary_hull
    end

    # n_arealunits
    n_arealunits = length(valid_leaf_nodes_filtered)
    centroids = Vector{Tuple{Float64, Float64}}(undef, n_arealunits)

    for (new_idx, node) in enumerate(valid_leaf_nodes_filtered)
        centroids[new_idx] = (mean(p[1] for p in node.points), mean(p[2] for p in node.points))
    end

    # FIXED: Re-assign all points to the nearest final centroid to ensure valid Voronoi classification
    assignments = zeros(Int, n)
    for i in 1:n
        assignments[i] = argmin([sum((pts[i] .- c).^2) for c in centroids])
    end

    area_idx = repeat(assignments, n_time)
    W = spzeros(n_arealunits, n_arealunits)

    # Create adjacency matrix based on Delaunay triangulation of centroids
    if n_arealunits >= 3
        tri = triangulate(centroids)
        for edge in each_edge(tri)
            u, v = edge
            if u > 0 && v > 0 && u <= n_arealunits && v <= n_arealunits
                W[u, v] = 1.0; W[v, u] = 1.0
            end
        end
    end

    return centroids, assignments, Symmetric(W), area_idx, boundary_hull
end

function assign_spatial_units(pts, area_method, n_time; kwargs...)
    # 1. Parameter Extraction
    time_idx = get(kwargs, :time_idx, nothing)
    buffer = get(kwargs, :buffer, nothing)
    capacity = get(kwargs, :capacity, 5)
    min_area_constraint = Float64(get(kwargs, :min_area_constraint, 0.0))
    max_area_constraint = Float64(get(kwargs, :max_area_constraint, Inf))
    min_total_units = get(kwargs, :min_total_arealunits, 3)
    max_total_units = get(kwargs, :max_total_arealunits, floor(Int, sqrt(length(pts))))
    tol = get(kwargs, :tol, 0.1)

    current_target = max_total_units
    prev_mean, prev_var = NaN, NaN
    best_result = nothing

    println("Starting convergence search for method: $area_method")

    while current_target >= min_total_units
        local c, assign, W, a_idx, hull

        if area_method == :quadtree
            adj_capacity = max(capacity, floor(Int, length(pts)/current_target))
            c, assign, W, a_idx, hull = assign_spatial_units_quadtree(pts, n_time;
                capacity=adj_capacity, time_idx=time_idx,
                min_area_constraint=min_area_constraint, max_area_constraint=max_area_constraint)

        elseif area_method == :avt # Renamed from :merge_voronoi
            c, assign, W, a_idx, hull = assign_spatial_units_merge_voronoi(pts, n_time;
                max_total_arealunits=current_target, time_idx=time_idx, buffer=buffer,
                min_area_constraint=min_area_constraint, max_area_constraint=max_area_constraint,
                tol=tol)

        else
            tri_full = triangulate(pts); hull_v = convex_hull(tri_full)
            raw_hull = pts[DelaunayTriangulation.get_vertices(hull_v)]
            cx, cy = mean(p[1] for p in raw_hull), mean(p[2] for p in raw_hull)
            buf_val = isnothing(buffer) ? 0.5 : buffer
            hull = map(p -> (p[1]+(p[1]-cx)/sqrt((p[1]-cx)^2+(p[2]-cy)^2)*buf_val, p[2]+(p[2]-cy)/sqrt((p[1]-cx)^2+(p[2]-cy)^2)*buf_val), raw_hull)
            push!(hull, hull[1])

            init_c = generate_grid_centroids(pts, current_target)

            if area_method == :triangulation
                c, assign = init_c, [argmin([sum((p .- s).^2) for s in init_c]) for p in pts]
            elseif area_method == :cvt
                # Merged granular_cvt into standard cvt
                c, assign = compute_hybrid_density_cvt(pts, 1:10, 1:10, ones(10,10), init_c)
            elseif area_method == :cvt_poisson
                xg, yg, dens = estimate_intensity_kde_optimized(pts)
                c, assign = compute_hybrid_density_cvt(pts, xg, yg, dens, init_c)
            elseif area_method == :cvt_gp
                xg, yg, dens = estimate_intensity_gp(pts)
                c, assign = compute_hybrid_density_cvt(pts, xg, yg, dens, init_c)
            else
                error("Unknown method: $area_method")
            end

            n_f = length(c); W = spzeros(n_f, n_f)
            if n_f >= 3
                tr = triangulate(c)
                for e in each_edge(tr)
                    u,v = e; if u > 0 && v > 0 && u <= n_f && v <= n_f W[u,v]=1.0; W[v,u]=1.0 end
                end
            end
            a_idx = repeat(assign, n_time)
        end

        counts = [count(==(i), assign) for i in 1:length(c)]
        curr_mean, curr_var = mean(counts), var(counts)

        if !isnan(prev_mean) && abs(curr_mean - prev_mean)/prev_mean < tol && abs(curr_var - prev_var)/prev_var < tol
            println("Asymptotic convergence reached at $current_target units.")
            break
        end

        best_result = (c, assign, Symmetric(W), a_idx, hull)
        prev_mean, prev_var = curr_mean, curr_var
        current_target -= 1
    end

    return best_result
end