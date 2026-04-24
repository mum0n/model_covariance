

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



function get_kde_seeds(pts, target_u)
    # Basic KDE-based seeding using StatsBase weights based on local density
    if isempty(pts) return [] end
    n = length(pts)
    dists = [sum((p1 .- p2).^2) for p1 in pts, p2 in pts]
    # Inverse of mean distance as a density proxy
    weights = 1.0 ./ (mean(dists, dims=2)[:] .+ 1e-6)
    idx = sample(1:n, Weights(weights), min(target_u, n), replace=false)
    return pts[idx]
end

 


function assign_spatial_units_inferred(adjacency_matrix; iterations=50, learning_rate=0.1, buffer_dist=0.5, input_polygons = nothing)
    """
    Synopsis: Manually constructs a spatial_res object for areal data like the Lip Cancer dataset.
              Centroid locations are spatially inferred from connectivity using a rudimentary force-directed layout.
    Inputs:
    - adjacency_matrix: The adjacency matrix (W) of the areal units.
    - iterations: Number of iterations for the force-directed layout.
    - learning_rate: Step size for moving centroids in the layout algorithm.
    - buffer_dist: Distance to buffer the convex hull when polygons are inferred.
    - input_polygons: Optional. A vector of LibGEOS Polygons. If provided, centroids and hull are derived from these.
    """
    nAU = size(adjacency_matrix, 1)

    local final_centroids
    local adjacency_edges_output
    local polys_output
    local hull_coords_output
    local g_final # The final graph that will be in the result

    if input_polygons !== nothing && !isempty(input_polygons)
        # Case 1: Polygons are provided
        # 1. Extract centroids from input_polygons
        final_centroids_geoms = [LibGEOS.centroid(p) for p in input_polygons]
        final_centroids = map(final_centroids_geoms) do g_pt
            seq = LibGEOS.getCoordSeq(g_pt)
            (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
        end

        # 2. Determine hull by dissolving all internal edges
        united_geom = LibGEOS.unaryunion(input_polygons)
        hull_coords_output = get_coords_from_geom(united_geom)

        # 3. Determine adjacency from input_polygons (using LibGEOS.touches)
        adjacency_edges_output = []
        for i in 1:nAU
            g1 = input_polygons[i]
            for j in (i+1):nAU
                g2 = input_polygons[j]
                if LibGEOS.touches(g1, g2)
                    push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                else
                    # Fallback robust check, similar to get_voronoi_polygons_and_edges
                    g1_buffered = LibGEOS.buffer(g1, 1e-6)
                    if LibGEOS.intersects(g1_buffered, g2)
                        inter = LibGEOS.intersection(g1_buffered, g2)
                        if !LibGEOS.isEmpty(inter) && (LibGEOS.area(inter) > 1e-9 || LibGEOS.geomTypeId(inter) in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_MULTILINESTRING])
                            push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                        end
                    end
                end
            end
        end

        polys_output = [get_coords_from_geom(p) for p in input_polygons]

        # Build graph from the determined adjacency edges and ensure connectivity
        g_final = SimpleGraph(nAU)
        centroid_map = Dict(c => i for (i, c) in enumerate(final_centroids))
        for (c1, c2) in adjacency_edges_output
            u_idx = get(centroid_map, c1, 0)
            v_idx = get(centroid_map, c2, 0)
            if u_idx > 0 && v_idx > 0 && !has_edge(g_final, u_idx, v_idx)
                add_edge!(g_final, u_idx, v_idx)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids)

    else
        # Case 2: Polygons are not provided, infer centroids and use tessellation
        # 1. Build initial graph from adjacency_matrix for force-directed layout
        g_initial_for_layout = SimpleGraph(adjacency_matrix)

        # 2. Infer initial centroids using force-directed layout
        side = ceil(Int, sqrt(nAU))
        initial_centroids_fd = [(Float64(i % side), Float64(i ÷ side)) for i in 0:(nAU-1)]
        centroids_vec = [SVector{2, Float64}(c) for c in initial_centroids_fd]

        for iter in 1:iterations
            new_centroids_vec = copy(centroids_vec)
            for i in 1:nAU
                neighbors_i = Graphs.neighbors(g_initial_for_layout, i)
                if !isempty(neighbors_i)
                    avg_neighbor_pos = sum(centroids_vec[n] for n in neighbors_i) / length(neighbors_i)
                    new_centroids_vec[i] = centroids_vec[i] + learning_rate * (avg_neighbor_pos - centroids_vec[i])
                end
            end
            centroids_vec = new_centroids_vec
        end
        forced_layout_centroids = [(p[1], p[2]) for p in centroids_vec]

        # 3. Determine hull_geom from inferred centroids for clipping
        hull_geom = expand_hull(forced_layout_centroids, buffer_dist)
        hull_coords_output = get_coords_from_geom(hull_geom)

        # 4. Use tessellation to determine polygon coordinates and initial adjacency
        polys_coords_raw, _ = get_voronoi_polygons_and_edges(forced_layout_centroids, hull_geom)

        # 5. RECOMPUTE CENTROIDS from the generated (clipped) polygons
        final_centroids = Vector{Tuple{Float64, Float64}}(undef, length(polys_coords_raw))
        lg_polygons_for_adjacency = Vector{Union{LibGEOS.Polygon, Nothing}}(undef, length(polys_coords_raw))
        polys_output = polys_coords_raw

        for (idx, poly_coord_list) in enumerate(polys_coords_raw)
            if !isempty(poly_coord_list) && length(poly_coord_list) >= 3
                if poly_coord_list[1] != poly_coord_list[end]
                    push!(poly_coord_list, poly_coord_list[1])
                end
                lg_poly = LibGEOS.Polygon([ [Float64[p[1], p[2]] for p in poly_coord_list] ])
                centroid_geom = LibGEOS.centroid(lg_poly)
                seq = LibGEOS.getCoordSeq(centroid_geom)
                final_centroids[idx] = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
                lg_polygons_for_adjacency[idx] = lg_poly
            else
                @warn "Invalid or empty polygon at index $idx. Fallback used."
                final_centroids[idx] = forced_layout_centroids[idx]
                lg_polygons_for_adjacency[idx] = nothing
            end
        end

        # 6. Re-build adjacency
        adjacency_edges_output = []
        if !isempty(lg_polygons_for_adjacency)
            for i in 1:length(lg_polygons_for_adjacency)
                g1 = lg_polygons_for_adjacency[i]
                if g1 === nothing continue end
                for j in (i+1):length(lg_polygons_for_adjacency)
                    g2 = lg_polygons_for_adjacency[j]
                    if g2 === nothing continue end
                    if LibGEOS.touches(g1, g2)
                        push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                    else
                        g1_buffered = LibGEOS.buffer(g1, 1e-6)
                        if LibGEOS.intersects(g1_buffered, g2)
                            inter = LibGEOS.intersection(g1_buffered, g2)
                            if !LibGEOS.isEmpty(inter) && (LibGEOS.area(inter) > 1e-9 || LibGEOS.geomTypeId(inter) in [LibGEOS.GEOS_LINESTRING, LibGEOS.GEOS_MULTILINESTRING])
                                push!(adjacency_edges_output, (final_centroids[i], final_centroids[j]))
                            end
                        end
                    end
                end
            end
        end

        # 7. Build final graph
        g_final = SimpleGraph(nAU)
        centroid_map = Dict(c => i for (i, c) in enumerate(final_centroids))
        for (c1, c2) in adjacency_edges_output
            u_idx = get(centroid_map, c1, 0)
            v_idx = get(centroid_map, c2, 0)
            if u_idx > 0 && v_idx > 0 && !has_edge(g_final, u_idx, v_idx)
                add_edge!(g_final, u_idx, v_idx)
            end
        end
        g_final = ensure_connected!(g_final, final_centroids)
    end

    return (
        centroids = final_centroids,
        adjacency_edges = adjacency_edges_output,
        graph = g_final,
        polygons = polys_output,
        hull_coords = hull_coords_output
    )
end

function get_polygon_area(poly_coords)
    # Calculates the area of a polygon using the Shoelace formula.
    valid_pts = [p for p in poly_coords if !isnan(p[1])]
    if length(valid_pts) > 1 && valid_pts[1] == valid_pts[end]
        pop!(valid_pts)
    end
    if length(valid_pts) < 3 return 0.0 end
    x = [p[1] for p in valid_pts]
    y = [p[2] for p in valid_pts]
    return 0.5 * abs(dot(x, circshift(y, 1)) - dot(y, circshift(x, 1)))
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
 function assign_spatial_units(input_data, area_method; seeding=:kde, kwargs...)
    pts = input_data
    u_pts = unique(pts)
    n_pts_raw = length(pts)
    target_u = get(kwargs, :target_units, 10)
    min_total_u = get(kwargs, :min_total_arealunits, 1)
    min_ts_req = get(kwargs, :min_time_slices, 1)
    min_pts_val = get(kwargs, :min_points, 1)
    max_pts_val = get(kwargs, :max_points, n_pts_raw)
    min_area_val = get(kwargs, :min_area, 0.0)
    max_area_val = get(kwargs, :max_area, Inf)
    cv_min_val = get(kwargs, :cv_min, 1.0)
    buffer_dist = get(kwargs, :buffer_dist, 0.5)
    t_idx = get(kwargs, :time_idx, ones(Int, n_pts_raw))
    tol = get(kwargs, :tolerance, 0.1)

    hull_geom = expand_hull(pts, buffer_dist)

    local c_mid
    if area_method == :cvt
        c_mid = get_cvt_centroids(pts, hull_geom, target_u, tol; cv_min=cv_min_val)
    elseif area_method == :kvt
        c_mid = get_kvt_centroids(pts, t_idx, target_u, min_total_u, min_ts_req, min_pts_val, max_pts_val, tol; cv_min=cv_min_val)
    elseif area_method == :bvt
        c_mid = get_bvt_centroids(u_pts, t_idx, target_u, min_total_u, min_ts_req, min_pts_val, max_pts_val, tol; cv_min=cv_min_val)
    elseif area_method == :qvt
        c_mid = get_qvt_centroids(u_pts, t_idx, target_u, min_total_u, min_ts_req, min_pts_val, max_pts_val, tol; cv_min=cv_min_val)
    elseif area_method == :avt
        c_mid = get_avt_centroids(u_pts, pts, t_idx; min_u=min_total_u, min_ts=min_ts_req, min_pts=min_pts_val, cv_min=cv_min_val, tol=tol)
    else
        error("Unknown method: $area_method")
    end

    polys_coords, v_edges = get_voronoi_polygons_and_edges(c_mid, hull_geom)

    final_centroids = Tuple{Float64, Float64}[]
    lg_polys = []
    for p_coords in polys_coords
        if p_coords[1] != p_coords[end]; push!(p_coords, p_coords[1]); end
        lg_p = LibGEOS.Polygon([[[p[1], p[2]] for p in p_coords]])
        push!(lg_polys, lg_p)
        cent_g = LibGEOS.centroid(lg_p)
        seq = LibGEOS.getCoordSeq(cent_g)
        push!(final_centroids, (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1)))
    end

    new_assignments = [argmin([sum((p .- sj) .^ 2) for sj in final_centroids]) for p in pts]
    n_units = length(final_centroids)
    g = SimpleGraph(n_units)
    for i in 1:n_units, j in (i+1):n_units
        if LibGEOS.touches(lg_polys[i], lg_polys[j]); add_edge!(g, i, j); end
    end
    g = ensure_connected!(g, final_centroids)

    return (centroids = final_centroids, assignments = new_assignments, polygons = polys_coords, adjacency_edges = v_edges, graph = g, hull_coords = get_coords_from_geom(hull_geom))
end


function get_qvt_centroids(pts, t_idx, n_target, min_u, min_ts, min_pts, max_pts, tol; cv_min=1.0)
    # Quadtree-style splitting: 4-way split per iteration based on median x/y
    data = collect(zip(pts, t_idx))
    regions = [data]
    prev_cv = Inf

    while length(regions) < n_target
        # Choose the most 'unbalanced' or largest region to split
        v_idx = argmax([length(r) > max_pts ? length(r) * 1e6 : length(r) for r in regions])
        target = regions[v_idx]
        if length(target) < 4 break end

        xs, ys = [p[1][1] for p in target], [p[1][2] for p in target]
        mx, my = median(xs), median(ys)

        r1 = filter(p -> p[1][1] <= mx && p[1][2] <= my, target)
        r2 = filter(p -> p[1][1] > mx  && p[1][2] <= my, target)
        r3 = filter(p -> p[1][1] <= mx && p[1][2] > my,  target)
        r4 = filter(p -> p[1][1] > mx  && p[1][2] > my,  target)

        # Only proceed if we actually achieve a split to prevent stagnation
        valid_splits = filter(r -> !isempty(r), [r1, r2, r3, r4])
        if length(valid_splits) < 2 break end

        splice!(regions, v_idx, valid_splits)

        # Balancing check
        counts = length.(regions)
        ts_counts = [length(unique([p[2] for p in r])) for r in regions]
        curr_cv = std(counts) / (mean(counts) + 1e-9)

        if length(regions) >= min_u && (curr_cv <= cv_min || abs(prev_cv - curr_cv) < (tol * 0.1)) && all(ts_counts .>= min_ts) break end
        prev_cv = curr_cv
    end
    return [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in regions]
end


function get_kvt_centroids(pts, t_idx, n_target, min_u, min_ts, min_pts, max_pts, tol; cv_min=1.0)
    u_pts_map = unique(pts)
    n_pts = length(u_pts_map)
    if n_pts == 0 return [] end
    idx_init = StatsBase.sample(1:n_pts, min(Int(n_target), n_pts); replace=false)
    c_iter = [u_pts_map[i] for i in idx_init]
    data = collect(zip(pts, t_idx))
    damping = 0.7

    for iter in 1:100
        old_centroids = copy(c_iter)
        assigns = [argmin([sum((p[1] .- sj).^2) for sj in c_iter]) for p in data]
        for k in 1:length(c_iter)
            idx = findall(==(k), assigns)
            if !isempty(idx) 
                new_center = (mean(data[j][1][1] for j in idx), mean(data[j][1][2] for j in idx))
                c_iter[k] = ((1.0 - damping) * old_centroids[k][1] + damping * new_center[1], (1.0 - damping) * old_centroids[k][2] + damping * new_center[2])
            end
        end
        counts = [count(==(k), assigns) for k in 1:length(c_iter)]
        if (std(counts)/(mean(counts)+1e-9)) < cv_min break end
        damping *= 0.99
    end
    return c_iter
end

function get_cvt_centroids(pts, hull_geom, n_target, tol; cv_min=1.0, max_iter=100)
    u_pts = unique(pts)
    idx = StatsBase.sample(1:length(u_pts), min(n_target, length(u_pts)), replace=false)
    curr_centroids = [u_pts[i] for i in idx]

    for iter in 1:max_iter
        polys, _ = get_voronoi_polygons_and_edges(curr_centroids, hull_geom)
        new_centroids = Tuple{Float64, Float64}[]
        max_shift = 0.0

        for (i, poly_coords) in enumerate(polys)
            if length(poly_coords) > 2
                temp_poly = LibGEOS.Polygon([[[p[1], p[2]] for p in poly_coords]])
                cent_geom = LibGEOS.centroid(temp_poly)
                seq = LibGEOS.getCoordSeq(cent_geom)
                new_c = (LibGEOS.getX(seq, 1), LibGEOS.getY(seq, 1))
                shift = sqrt(sum((new_c .- curr_centroids[i]).^2))
                max_shift = max(max_shift, shift)
                push!(new_centroids, new_c)
            else
                push!(new_centroids, curr_centroids[i])
            end
        end
        
        # Optional: Balanced termination check
        assigns = [argmin([sum((p .- sj).^2) for sj in new_centroids]) for p in pts]
        counts = [count(==(k), assigns) for k in 1:length(new_centroids)]
        cv = std(counts) / (mean(counts) + 1e-9)
        
        curr_centroids = new_centroids
        if max_shift < tol || cv <= cv_min break end
    end
    return curr_centroids
end

function get_bvt_centroids(pts, t_idx, n_target, min_u, min_ts, min_pts, max_pts, tol; cv_min=1.0)
    data = collect(zip(pts, t_idx))
    regions = [data]
    prev_cv = Inf
    while length(regions) < n_target
        v_idx = argmax([length(r) > max_pts ? length(r) * 1e6 : (length(r) > 1 ? max(std([p[1][1] for p in r]), std([p[1][2] for p in r])) : 0.0) for r in regions])
        if length(regions[v_idx]) < 2 break end
        target = regions[v_idx]
        xs, ys = [p[1][1] for p in target], [p[1][2] for p in target]
        dim = std(xs) > std(ys) ? 1 : 2
        med = median(dim == 1 ? xs : ys)
        mask = [p[1][dim] <= med for p in target]
        r1, r2 = target[mask], target[.!mask]
        if isempty(r1) || isempty(r2) break end
        splice!(regions, v_idx, [r1, r2])
        counts = length.(regions)
        ts_counts = [length(unique([p[2] for p in r])) for r in regions]
        curr_cv = std(counts) / (mean(counts) + 1e-9)
        
        if length(regions) >= min_u && (curr_cv <= cv_min || abs(prev_cv - curr_cv) < (tol * 0.1)) && all(ts_counts .>= min_ts) break end
        prev_cv = curr_cv
    end
    return [(mean(p[1][1] for p in r), mean(p[1][2] for p in r)) for r in regions]
end
 
 
 function get_avt_centroids(c_init, pts, t_idx; min_u=1, min_ts=1, min_pts=1, min_area=0.0, tol=0.1, cv_min=1.0)
    # Enhanced AVT: Includes CV logic and cv_min threshold for termination
    data = collect(zip(pts, t_idx))
    if isempty(c_init) || isempty(data) return c_init end

    curr_centroids = [SVector{2, Float64}(c) for c in c_init]
    prev_cv = Inf

    while length(curr_centroids) > min_u
        cluster_map = [Int[] for _ in 1:length(curr_centroids)]
        for (i, d) in enumerate(data)
            dist_sq = [sum((d[1] .- c) .^ 2) for c in curr_centroids]
            push!(cluster_map[argmin(dist_sq)], i)
        end

        counts = length.(cluster_map)
        ts_counts = [length(unique([data[idx][2] for idx in cluster_map[k]])) for k in 1:length(curr_centroids)]
        curr_cv = std(counts) / (mean(counts) + 1e-9)

        violators = findall(k -> counts[k] < min_pts || ts_counts[k] < min_ts, 1:length(curr_centroids))

        # Stopping criteria: No violations AND (reached target min_u OR CV meets cv_min OR stabilized)
        if isempty(violators) && (length(curr_centroids) <= min_u || curr_cv <= cv_min || abs(prev_cv - curr_cv) < (tol * 0.1))
            break
        end

        prev_cv = curr_cv
        target_idx = !isempty(violators) ? violators[argmin(counts[violators])] : argmin(counts)
        deleteat!(curr_centroids, target_idx)
    end

    final_cluster_map = [Int[] for _ in 1:length(curr_centroids)]
    for (i, d) in enumerate(data)
        dist_sq = [sum((d[1] .- c) .^ 2) for c in curr_centroids]
        push!(final_cluster_map[argmin(dist_sq)], i)
    end

    final_c = Tuple{Float64, Float64}[]
    for idx in 1:length(curr_centroids)
        idxs = final_cluster_map[idx]
        if !isempty(idxs)
            push!(final_c, (mean(data[j][1][1] for j in idxs), mean(data[j][1][2] for j in idxs)))
        else
            push!(final_c, Tuple(curr_centroids[idx]))
        end
    end
    return final_c
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
    # Ensures the spatial graph is connected by adding edges between the nearest 
    # components based on the provided centroid coordinates.
    while !is_connected(g)
        comps = connected_components(g)
        best_dist = Inf
        best_pair = (0, 0)
        
        # Find the two closest nodes belonging to different components
        for i in 1:length(comps), j in (i+1):length(comps)
            for u in comps[i], v in comps[j]
                d = sum((centroids[u] .- centroids[v]).^2)
                if d < best_dist
                    best_dist = d
                    best_pair = (u, v)
                end
            end
        end
        
        if best_pair != (0, 0)
            add_edge!(g, best_pair[1], best_pair[2])
        else
            break
        end
    end
    return g
end


 
function plot_spatial_graph(pts, au; title="Spatial Partitioning", domain_boundary=nothing)
    # 1. Base Plot with Polygons
    plt = Plots.plot(aspect_ratio=:equal, title=title, legend=false)
    
    # Plot Polygons
    for poly_coords in au.polygons
        if length(poly_coords) > 2
            px = [p[1] for p in poly_coords if !isnan(p[1])]
            py = [p[2] for p in poly_coords if !isnan(p[2])]
            if !isempty(px) && (px[1], py[1]) != (px[end], py[end])
                push!(px, px[1]); push!(py, py[1])
            end
            Plots.plot!(plt, px, py, seriestype=:shape, fillalpha=0.1, linecolor=:black, lw=0.5)
        end
    end

    # 2. Plot Adjacency Graph Edges (Using au.centroids directly for nodes)
    for edge in Graphs.edges(au.graph)
        u, v = src(edge), dst(edge)
        p1, p2 = au.centroids[u], au.centroids[v]
        Plots.plot!(plt, [p1[1], p2[1]], [p1[2], p2[2]], color=:red, lw=1.5, alpha=0.6)
    end

    # 3. Plot Centroids and Raw Points
    Plots.scatter!(plt, [p[1] for p in pts], [p[2] for p in pts], 
        markersize=1, color=:gray, alpha=0.3, label="Points")
    Plots.scatter!(plt, [c[1] for c in au.centroids], [c[2] for c in au.centroids], 
        markersize=4, color=:blue, markerstrokecolor=:white, label="Centroids")

    if !isnothing(domain_boundary)
        bx = [p[1] for p in domain_boundary if !isnan(p[1])]
        by = [p[2] for p in domain_boundary if !isnan(p[2])]
        Plots.plot!(plt, bx, by, color=:black, lw=2, ls=:dash)
    end

    return plt
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
    # Dynamically adjusted to the number of polygons returned.

    roll(v, k) = v[mod1.(1:length(v), length(v) .- k)]

    # Use the number of final polygons as the reference for units
    n_units = length(spatial_res.polygons)
    
    # If no polygons were created, return NaN metrics
    if n_units == 0
        return (mean_density = NaN, sd_density = NaN, cv_density = NaN)
    end

    # Count assignments for the units that actually have polygons
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

    # Ensure counts and areas are the same length
    len = min(length(counts), length(areas))
    densities = counts[1:len] ./ areas[1:len]
    
    return (
        mean_density = mean(densities),
        sd_density = std(densities),
        cv_density = std(densities) / (mean(densities) + 1e-9)
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



function plot_kde_simple(pts; grid_res=600, sd_extension_factor=0.25, title="Spatial Intensity (KDE)")
    # Internal wrapper for estimate_local_kde_with_extrapolation
    # Description: Generates a simple 2D Heatmap of spatial intensity using Kernel Density Estimation.
    # Inputs:
    #   - pts: Vector of (x, y) coordinate tuples.
    #   - grid_res: Resolution of the output grid.
    #   - sd_extension_factor: Factor to extend the bandwidth standard deviation.
    #   - title: Title for the generated plot.
    # Outputs:
    #   - A Plots.Plot object (Heatmap with scatter overlay).
    # Using a dummy time_idx of 1s since we are plotting a static slice
    t_idx_dummy = ones(Int, length(pts))
    x_g, y_g, intensity = estimate_local_kde_with_extrapolation(pts, t_idx_dummy, 1; grid_res=grid_res, sd_extension_factor=sd_extension_factor)

    plt = Plots.heatmap(x_g, y_g, intensity',
                  title=title,
                  c=:viridis,
                  aspect_ratio=:equal,
                  xlabel="X", ylabel="Y")
    Plots.scatter!(plt, [p[1] for p in pts], [p[2] for p in pts],
                   markersize=2, markercolor=:white, markeralpha=0.5, label="Points")
    return plt
end
