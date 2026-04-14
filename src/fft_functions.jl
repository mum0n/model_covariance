function anisotropic_matern_spectral_density(Lx, Ly, var, ell1, ell2, theta, nu, freq_x, freq_y)
    # Input:
    # Lx, Ly: Grid dimensions (for normalization of frequencies)
    # var: Spatial variance
    # ell1, ell2: Anisotropic length scales
    # theta: Rotation angle (radians)
    # nu: Smoothness parameter of the Matern kernel
    # freq_x, freq_y: 2D arrays of spatial frequencies

    # Rotate frequencies
    fx_rot = freq_x .* cos(theta) + freq_y .* sin(theta)
    fy_rot = -freq_x .* sin(theta) + freq_y .* cos(theta)

    # Squared effective radial frequency in the anisotropic space
    # The factor (2π) is often absorbed into the length scales definition
    # Here we define the length scales directly related to the frequency scaling.
    # Using the definition where length scale 'rho' corresponds to (1/rho) in frequency domain.
    kappa_sq = (1.0 / ell1)^2 .* fx_rot.^2 + (1.0 / ell2)^2 .* fy_rot.^2

    # Spectral density formula (up to a constant factor)
    # For Matern, the spectral density is (1 + kappa_sq)^(-(nu + D/2)), where D=2 for 2D
    # We also need to scale by variance
    S = var .* (1.0 .+ kappa_sq).^(-(nu + 1.0))

    # Handle the zero frequency component if it exists, to avoid issues or make it finite.
    # In practice, S[1,1] (DC component) needs careful handling for mean processes.
    # For generating realizations, it's typically fine to let it be, but for inference
    # a prior on the mean might be needed or the DC component handled separately.
    S[1,1] = S[1,1] # This is just a placeholder, often S[1,1] is related to the mean/total power.

    # The spectral density must be real and non-negative. It's also symmetric for real fields.
    # We'll use this to generate a complex spectral field.
    return S
end

# Helper function to generate frequency grids
function generate_freq_grid(N, L)
    fftfreq_vals = FFTW.fftfreq(N, N / L)
    return repeat(fftfreq_vals, 1, N), repeat(fftfreq_vals', N, 1)
end
