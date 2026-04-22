function expand_hull_v0(pts, buffer_dist)
    """
    Synopsis: Computes the convex hull of points and expands it by a buffer distance.
    Inputs:
    - pts: Vector of (x, y) tuples.
    - buffer_dist: Distance to buffer the convex hull.
    Outputs:
    - A LibGEOS Polygon geometry representing the buffered convex hull.
    """
    if isempty(pts) return LibGEOS.Polygon([[ (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0) ]]) end
    points_geom = LibGEOS.MultiPoint([(p[1], p[2]) for p in pts])
    hull = LibGEOS.convexhull(points_geom)
    buffered_hull = LibGEOS.buffer(hull, buffer_dist)
    return buffered_hull
end


function get_coords_from_geom_v0(geom)
    """
    Synopsis: Extracts coordinates from various LibGEOS geometry types.
    Inputs:
    - geom: A LibGEOS geometry object.
    Outputs:
    - A vector of (x, y) coordinates.
    """
    coords = []
    if LibGEOS.geomTypeId(geom) == LibGEOS.GEOS_POINT
        push!(coords, (LibGEOS.getX(geom), LibGEOS.getY(geom)))
    elseif LibGEOS.geomTypeId(geom) == LibGEOS.GEOS_LINESTRING || LibGEOS.geomTypeId(geom) == LibGEOS.GEOS_LINEARRING
        for i in 1:LibGEOS.getNumPoints(geom)
            push!(coords, (LibGEOS.getX(geom, i), LibGEOS.getY(geom, i)))
        end
    elseif LibGEOS.geomTypeId(geom) == LibGEOS.GEOS_POLYGON
        exterior = LibGEOS.getExteriorRing(geom)
        for i in 1:LibGEOS.getNumPoints(exterior)
            push!(coords, (LibGEOS.getX(exterior, i), LibGEOS.getY(exterior, i)))
        end
    elseif LibGEOS.geomTypeId(geom) == LibGEOS.GEOS_MULTIPOLYGON
        for i in 1:LibGEOS.getNumGeometries(geom)
            poly = LibGEOS.getGeometryN(geom, i)
            exterior = LibGEOS.getExteriorRing(poly)
            for j in 1:LibGEOS.getNumPoints(exterior)
                push!(coords, (LibGEOS.getX(exterior, j), LibGEOS.getY(exterior, j)))
            end
            if i < LibGEOS.getNumGeometries(geom) push!(coords, (NaN, NaN)) end # Separator for plotting
        end
    end
    return coords
end



function expand_hull(pts, buffer_dist)
    """
    Synopsis: Computes the convex hull of points and expands it by a buffer distance.
    Inputs:
    - pts: Vector of (x, y) tuples.
    - buffer_dist: Distance to buffer the convex hull.
    Outputs:
    - A LibGEOS Polygon geometry representing the buffered convex hull.
    """

    if isempty(pts) return LibGEOS.Polygon([[ (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0) ]]) end
    coords_vec = [[Float64(p[1]), Float64(p[2])] for p in pts]
    points_geom = LibGEOS.MultiPoint(coords_vec)
    hull = LibGEOS.convexhull(points_geom)
    buffered_hull = LibGEOS.buffer(hull, buffer_dist)
    return buffered_hull
end

function get_coords_from_geom(geom)
    """
    Synopsis: Extracts coordinates from various LibGEOS geometry types.
    Inputs:
    - geom: A LibGEOS geometry object.
    Outputs:
    - A vector of (x, y) coordinates.
    """

    coords = Tuple{Float64, Float64}[]
    local type_id = -1
    try
        type_id = LibGEOS.geomTypeId(geom)
        if type_id == LibGEOS.GEOS_POINT
             # Access coordinate sequence directly for point types
             seq = LibGEOS.getCoordSeq(geom)
             push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
             return coords
        elseif type_id == LibGEOS.GEOS_POLYGON
            ring = LibGEOS.exteriorRing(geom)
            n = LibGEOS.numPoints(ring)
            for i in 1:n
                p = LibGEOS.getPoint(ring, i)
                seq = LibGEOS.getCoordSeq(p)
                push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
            end
        elseif type_id == LibGEOS.GEOS_MULTIPOLYGON
            for i in 1:LibGEOS.numGeometries(geom)
                poly = LibGEOS.getGeometryN(geom, i)
                ring = LibGEOS.exteriorRing(poly)
                n = LibGEOS.numPoints(ring)
                for j in 1:n
                    p = LibGEOS.getPoint(ring, j)
                    seq = LibGEOS.getCoordSeq(p)
                    push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
                end
                if i < LibGEOS.numGeometries(geom); push!(coords, (NaN, NaN)); end
            end
        elseif type_id in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_LINEARRING]
            n = LibGEOS.numPoints(geom)
            for i in 1:n
                p = LibGEOS.getPoint(geom, i)
                seq = LibGEOS.getCoordSeq(p)
                push!(coords, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
            end
        end
    catch e
        @warn "Coordinate extraction failed for type $type_id: $e"
    end
    return coords
end


function get_bvt_centroids(pts, n_target, min_u, tol)
    """
    Synopsis: Binary Vector Tree partitioning. Recursively splits along the axis of maximum variance.
    Inputs:
    - n_target: Hard maximum number of units.
    - min_u: Minimum required units.
    - tol: Tolerance for density CV stabilization.
    """
    regions = [pts]
    prev_cv = Inf
    while length(regions) < n_target
        v_idx = argmax([length(r) > 1 ? max(std([p[1] for p in r]), std([p[2] for p in r])) : 0.0 for r in regions])
        if length(regions[v_idx]) < 2 break end
        target_r = regions[v_idx]
        xs, ys = [p[1] for p in target_r], [p[2] for p in target_r]
        if std(xs) > std(ys)
            med = median(xs)
            r1, r2 = filter(p -> p[1] <= med, target_r), filter(p -> p[1] > med, target_r)
        else
            med = median(ys)
            r1, r2 = filter(p -> p[2] <= med, target_r), filter(p -> p[2] > med, target_r)
        end
        if isempty(r1) || isempty(r2) break end

        splice!(regions, v_idx, [r1, r2])
        counts = length.(regions)
        curr_cv = std(counts) / (mean(counts) + 1e-9)
        if length(regions) >= min_u && abs(prev_cv - curr_cv) < (tol * 0.1) break end
        prev_cv = curr_cv
    end
    return [(mean(p[1] for p in r), mean(p[2] for p in r)) for r in regions]
end

function get_qvt_centroids(pts, n_target, min_u, tol)
    """
    Synopsis: Quadrant Voronoi Tessellation. Recursively divides space into four quadrants.
    """
    regions = [pts]
    prev_cv = Inf
    while length(regions) < n_target
        v_idx = argmax([length(r) for r in regions])
        if length(regions[v_idx]) < 2 break end
        target_r = splice!(regions, v_idx)
        mx, my = mean(p[1] for p in target_r), mean(p[2] for p in target_r)
        qs = [filter(p -> p[1] <= mx && p[2] <= my, target_r),
              filter(p -> p[1] > mx && p[2] <= my, target_r),
              filter(p -> p[1] <= mx && p[2] > my, target_r),
              filter(p -> p[1] > mx && p[2] > my, target_r)]

        for q in qs
            if !isempty(q)
                push!(regions, q)
                if length(regions) >= n_target break end
            end
        end

        counts = length.(regions)
        curr_cv = std(counts) / (mean(counts) + 1e-9)
        if length(regions) >= min_u && abs(prev_cv - curr_cv) < (tol * 0.1) break end
        prev_cv = curr_cv
        if length(regions) >= n_target break end
    end
    return [(mean(p[1] for p in r), mean(p[2] for p in r)) for r in regions]
end



function assign_spatial_units(pts, area_method; seeding=:kde, kwargs...)
    """
    assign_spatial_units(pts, area_method; seeding=:kde, kwargs...)

    Description: High-level wrapper for spatial partitioning with temporal and density constraints.

    Inputs:
    - area_method: :cvt (Iterative), :bvt (Recursive), :qvt (Quadrant), :avt (Agglomerative).
    - target_units: Goal for CVT/AVT.
    - max_units: Hard cap for BVT/QVT and CVT expansion.
    - min_time_slices: Minimum unique time indices required per unit.
    """
    max_u = get(kwargs, :max_units, 50)
    target_u = get(kwargs, :target_units, 20)
    min_u_req = get(kwargs, :min_units, 2)
    buffer_dist_val = get(kwargs, :buffer_dist, 0.5)
    tol = get(kwargs, :tol, 1e-1)
    min_ts_req = get(kwargs, :min_time_slices, 1)
    t_idx = get(kwargs, :time_idx, ones(Int, length(pts)))

    u_pts = unique(pts)
    hull_geom = expand_hull(u_pts, buffer_dist_val)
    hull_coords = get_coords_from_geom(hull_geom)

    local c
    if area_method == :bvt
        c = get_bvt_centroids(u_pts, max_u, min_u_req, tol)
    elseif area_method == :qvt
        c = get_qvt_centroids(u_pts, max_u, min_u_req, tol)
    elseif area_method == :avt
        c_init = u_pts[sample(1:length(u_pts), min(target_u, length(u_pts)), replace=false)]
        c = get_avt_centroids(c_init, pts, hull_coords, hull_geom; min_pts=get(kwargs, :min_pts, 5), min_units=min_u_req, tol=tol)
    elseif area_method == :cvt
        curr_target = target_u
        best_c = []
        for retry in 1:3
            c_iter = u_pts[sample(1:length(u_pts), min(curr_target, length(u_pts)), replace=false)]
            prev_cv = Inf
            for i in 1:50
                assign = [argmin([sum((p .- sj).^2) for sj in c_iter]) for p in pts]
                for k in 1:length(c_iter)
                    idx = findall(==(k), assign)
                    if !isempty(idx) c_iter[k] = (mean(p[1] for p in pts[idx]), mean(p[2] for p in pts[idx])) end
                end
                counts = [count(==(k), assign) for k in 1:length(c_iter)]
                current_cv = std(counts) / (mean(counts) + 1e-9)
                if abs(prev_cv - current_cv) < tol break end
                prev_cv = current_cv
            end
            best_c = c_iter
            counts = [count(==(k), [argmin([sum((p .- sj).^2) for sj in c_iter]) for p in pts]) for k in 1:length(c_iter)]
            if (std(counts)/mean(counts)) > tol * 2 && curr_target < max_u
                curr_target = min(max_u, floor(Int, curr_target * 1.2))
            else
                break
            end
        end
        c = best_c
    end

    while length(c) > min_u_req
        assigns = [argmin([sum((p .- sj) .^ 2) for sj in c]) for p in pts]
        ts_counts = [length(unique(t_idx[findall(==(i), assigns)])) for i in 1:length(c)]
        violators = findall(count -> count < min_ts_req, ts_counts)
        if isempty(violators) break end
        target_idx = violators[argmin(ts_counts[violators])]
        deleteat!(c, target_idx)
    end

    final_assignments = [argmin([sum((p .- sj) .^ 2) for sj in c]) for p in pts]
    polys_coords, v_edges_output = get_voronoi_polygons_and_edges(c, hull_geom)
    res = (centroids=c, assignments=final_assignments, polygons=polys_coords, adjacency_edges=v_edges_output, hull_coords=hull_coords)
    g = ensure_connected!(SimpleGraph(length(c)), c)
    return merge(res, (graph=g,))
end



function get_voronoi_polygons_and_edges(centroids, hull_geom)
    # Description: Generates clipped Voronoi polygons based on centroids and identifies neighbors sharing a boundary for graph construction.
    # Inputs:
    #   - centroids: Vector of (x, y) coordinates for unit centers.
    #   - hull_geom: LibGEOS geometry used to clip the Voronoi cells.
    # Outputs:
    #   - polygons: Vector of coordinate vectors for the polygons.
    #   - edges: Adjacency list showing connected centroid pairs.

    pts_dt = [(Float64(c[1]), Float64(c[2])) for c in centroids]
    tri = triangulate(pts_dt)
    hull_coords = get_coords_from_geom(hull_geom)
    xs = [p[1] for p in hull_coords if !isnan(p[1])]
    ys = [p[2] for p in hull_coords if !isnan(p[2])]
    bbox = (minimum(xs), maximum(xs), minimum(ys), maximum(ys))
    vorn = voronoi(tri)

    clipped_polys = []
    for i in each_generator(vorn)
        vertices = get_polygon_coordinates(vorn, i, bbox)
        if !isempty(vertices)
            poly_pts = [[v[1], v[2]] for v in vertices]
            if poly_pts[1] != poly_pts[end] push!(poly_pts, poly_pts[1]) end
            try
                lg_poly = LibGEOS.Polygon([poly_pts])
                clipped = LibGEOS.intersection(lg_poly, hull_geom)
                if !LibGEOS.isEmpty(clipped)
                    push!(clipped_polys, (id=i, geom=clipped))
                end
            catch e end
        end
    end

    final_coords = [get_coords_from_geom(p.geom) for p in clipped_polys]
    v_edges = []

    # Precision-robust adjacency using LibGEOS predicates
    for i in 1:length(clipped_polys)
        g1 = clipped_polys[i].geom
        for j in i+1:length(clipped_polys)
            g2 = clipped_polys[j].geom
            
            # Standard GEOS predicate for adjacency
            if LibGEOS.touches(g1, g2)
                push!(v_edges, (centroids[clipped_polys[i].id], centroids[clipped_polys[j].id]))
            else
                # Fallback: Robust check using a tiny buffer for floating-point misalignments
                g1_buffered = LibGEOS.buffer(g1, 1e-6)
                if LibGEOS.intersects(g1_buffered, g2)
                    inter = LibGEOS.intersection(g1_buffered, g2)
                    if !LibGEOS.isEmpty(inter) && (LibGEOS.area(inter) > 1e-9 || LibGEOS.geomTypeId(inter) in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_MULTILINESTRING])
                        push!(v_edges, (centroids[clipped_polys[i].id], centroids[clipped_polys[j].id]))
                    end
                end
            end
        end
    end

    return final_coords, v_edges
end


function check_connectivity(g)
    """
    Synopsis: Evaluates the connectivity of a spatial graph.
    Inputs:
    - g: A SimpleGraph.
    Outputs:
    - NamedTuple showing connection status and components.
    """
    comps = connected_components(g)
    return (is_connected = length(comps) == 1, n_components = length(comps), components = comps)
end



function ensure_connected!(g, centroids)
    """
    Synopsis: Force-connects disconnected graph components using nearest-neighbor edges.
    Inputs:
    - g: SimpleGraph (modified in-place).
    - centroids: Vector of (x, y) tuples.
    Outputs:
    - The modified connected SimpleGraph.
    """
    comps = connected_components(g)
    while length(comps) > 1
        # Take the first component and find the closest node in any other component
        c1 = comps[1]
        others = vcat(comps[2:end]...)
        
        min_dist = Inf
        best_pair = (0, 0)
        
        for i in c1
            for j in others
                d = sum((centroids[i] .- centroids[j]).^2)
                if d < min_dist
                    min_dist = d
                    best_pair = (i, j)
                end
            end
        end
        
        if best_pair[1] > 0
            add_edge!(g, best_pair[1], best_pair[2])
        end
        
        # Re-calculate components after adding the edge
        comps = connected_components(g)
    end
    return g
end



function get_avt_centroids(c_init, pts, hull_coords, hull_geom; min_pts=5, min_units=2, tol=1e-4)
    """
    Synopsis: Agglomerative Voronoi Tessellation that strictly enforces min_pts.
    """
    centroids = copy(c_init)
    
    while length(centroids) > min_units
        assignments = [argmin([sum((p .- sj) .^ 2) for sj in centroids]) for p in pts]
        counts = [count(==(i), assignments) for i in 1:length(centroids)]
        
        # Identify units below the point threshold
        violators = findall(c -> c < min_pts, counts)
        if isempty(violators)
            break # All units satisfy min_pts
        end

        # Merge the unit with the fewest points into its nearest neighbor
        target_idx = violators[argmin(counts[violators])]
        
        dists = [i == target_idx ? Inf : sum((centroids[target_idx] .- centroids[i]).^2) for i in 1:length(centroids)]
        merge_with = argmin(dists)
        
        if dists[merge_with] == Inf break end

        # Update centroid of the merged unit to be the weighted mean of both
        total_pts = counts[target_idx] + counts[merge_with]
        if total_pts > 0
            new_c = ( (centroids[target_idx][1]*counts[target_idx] + centroids[merge_with][1]*counts[merge_with])/total_pts,
                      (centroids[target_idx][2]*counts[target_idx] + centroids[merge_with][2]*counts[merge_with])/total_pts )
            centroids[merge_with] = new_c
        end

        deleteat!(centroids, target_idx)
    end

    return centroids
end



function plot_spatial_graph(pts, spatial_res; title="Spatial Partitioning", domain_boundary=[])
    """
    Inputs:
    - pts: Data points.
    - spatial_res: Result containing polygons, centroids, and edges.

    Outputs:
    - A Plots.Plot object.
    """
    p = Plots.plot(aspect_ratio=:equal, title=title, legend=false)
    if !isempty(domain_boundary)
        bx = [pt[1] for pt in domain_boundary if !isnan(pt[1])]
        by = [pt[2] for pt in domain_boundary if !isnan(pt[2])]
        if !isempty(bx) && (bx[1], by[1]) != (bx[end], by[end])
            push!(bx, bx[1]); push!(by, by[1])
        end
        Plots.plot!(p, bx, by, color=:black, lw=2)
    end
    for poly_coords in spatial_res.polygons
        if length(poly_coords) > 2
            px, py = [pt[1] for pt in poly_coords if !isnan(pt[1])], [pt[2] for pt in poly_coords if !isnan(pt[1])]
            if !isempty(px); push!(px, px[1]); push!(py, py[1]); Plots.plot!(p, px, py, seriestype=:shape, fillalpha=0.3, linecolor=:white, lw=0.5); end
        end
    end
    for edge in spatial_res.adjacency_edges
        p1, p2 = edge
        Plots.plot!(p, [p1[1], p2[1]], [p1[2], p2[2]], color=:red, lw=1.5, alpha=0.6)
    end
    Plots.scatter!(p, [pt[1] for pt in pts], [pt[2] for pt in pts], markersize=1, markercolor=:gray, alpha=0.3)
    Plots.scatter!(p, [pt[1] for pt in spatial_res.centroids], [pt[2] for pt in spatial_res.centroids], markersize=4, markercolor=:blue, markerstrokecolor=:white)
    return p
end



function generate_sim_data(n_pts, n_time; rndseed=42 )
    Random.seed!(rndseed)
    unique_pts = [(rand() * 10, rand() * 10) for _ in 1:n_pts]

    # Repeat the unique points for each time slice to match n_total observations
    pts_full_dataset = repeat(unique_pts, n_time)

    n_total = n_pts * n_time # This is now consistent with pts_full_dataset
    time_idx = repeat(1:n_time, inner=n_pts)
    weights = ones(n_total)
    trials = ones(Int, n_total)
    cov_indices = rand(1:3, n_total)

    spatial_effect = [sin(p[1]/2) + cos(p[2]/2) for p in unique_pts]
    spatial_effect_long = repeat(spatial_effect, n_time)

    temporal_effect = sin.(time_idx)
    y_sim  = 1.5 .* spatial_effect_long + 1.0 .* temporal_effect + randn(n_total) * 0.5
    y_binary = y_sim  .> (mean(y_sim) + 0.5)

    return (pts=pts_full_dataset, y_sim=y_sim, y_binary=y_binary, time_idx=time_idx,
            weights=weights, trials=trials, cov_indices=cov_indices)
end


function estimate_local_kde_with_extrapolation(pts, time_idx, target_ts; grid_res=600, sd_extension_factor=0.25)
    """
    Synopsis: Estimates 2D KDE for a specific time slice with extrapolation.
    Inputs:
    - pts: Vector of (x, y) coordinates for all time points.
    - time_idx: Vector of time indices corresponding to pts.
    - target_ts: The specific time slice to estimate KDE for.
    - grid_res: Resolution of the output grid (e.g., 100 for 100x100 grid).
    - sd_extension_factor: Multiplier for standard deviation to define the bandwidth.
    Outputs:
    - Tuple (x_grid, y_grid, intensity) where intensity is a matrix.
    """
    # Filter points for the target time slice
    filtered_pts = [p for (i, p) in enumerate(pts) if time_idx[i] == target_ts]
    if isempty(filtered_pts)
        error("No points found for the target time slice $target_ts")
    end
    xs, ys = [p[1] for p in filtered_pts], [p[2] for p in filtered_pts]
    # Calculate bandwidth based on standard deviation of points
    bw_x = std(xs) * sd_extension_factor
    bw_y = std(ys) * sd_extension_factor
    # Define grid boundaries extending slightly beyond the data range
    x_min, x_max = minimum(xs) - bw_x, maximum(xs) + bw_x
    y_min, y_max = minimum(ys) - bw_y, maximum(ys) + bw_y
    x_grid = collect(range(x_min, stop=x_max, length=grid_res))
    y_grid = collect(range(y_min, stop=y_max, length=grid_res))
    intensity = zeros(grid_res, grid_res)
    # Gaussian KDE implementation
    for i in 1:grid_res
        for j in 1:grid_res
            x_val, y_val = x_grid[i], y_grid[j]
            for (px, py) in filtered_pts
                dx = (x_val - px) / bw_x
                dy = (y_val - py) / bw_y
                intensity[i, j] += exp(-0.5 * (dx^2 + dy^2))
            end
        end
    end
    # Normalize intensity to sum to 1 (optional, depending on desired output)
    intensity ./= sum(intensity)
    return x_grid, y_grid, intensity
end


function calculate_metrics(spatial_res, pts)
    # Description: Calculates density metrics (mean, SD, CV) across the partitioned spatial units.
    # Inputs:
    #   - spatial_res: Partitioning results containing polygons and assignments.
    #   - pts: The data points being analyzed.
    # Outputs:
    #   - NamedTuple: (mean_density, sd_density, cv_density).

    roll(v, k) = v[mod1.(1:length(v), length(v) .- k)]

    n_units = length(spatial_res.centroids)
    counts = [count(==(i), spatial_res.assignments) for i in 1:n_units]

    areas = Float64[]
    for poly in spatial_res.polygons
        # Filter out NaN separators and ensure we only have unique vertices for the shoelace
        valid_pts = [p for p in poly if !isnan(p[1])]

        # Remove the last point if it's a duplicate of the first (common in ring formats)
        if length(valid_pts) > 1 && valid_pts[1] == valid_pts[end]
            pop!(valid_pts)
        end

        if length(valid_pts) > 2
            x = [p[1] for p in valid_pts]
            y = [p[2] for p in valid_pts]
            # Shoelace formula
            a = 0.5 * abs(dot(x, roll(y, 1)) - dot(y, roll(x, 1)))
            push!(areas, max(a, 1e-9))
        else
            push!(areas, 1e-9)
        end
    end

    densities = counts ./ areas
    return (
        mean_density = mean(densities),
        sd_density = std(densities),
        cv_density = std(densities) / mean(densities)
    )
end

 

function get_spatial_graph(spatial_res)
    """
    Synopsis: Converts partitioning results into a formal SimpleGraph.
    Inputs:
    - spatial_res: NamedTuple from assign_spatial_units.
    Outputs:
    - A SimpleGraph object.
    """
    n = length(spatial_res.centroids)
    g = SimpleGraph(n)
    centroid_map = Dict(c => i for (i, c) in enumerate(spatial_res.centroids))
    for edge in spatial_res.adjacency_edges
        u_idx, v_idx = get(centroid_map, edge[1], 0), get(centroid_map, edge[2], 0)
        if u_idx > 0 && v_idx > 0 add_edge!(g, u_idx, v_idx) end
    end
    return g
end
