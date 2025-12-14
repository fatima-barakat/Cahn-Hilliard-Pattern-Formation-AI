import numpy as np

def cahn_hilliard_1d(
    N=128,
    L=1.0,
    dt=1e-4,
    steps=5000,
    M=1.0,
    kappa=1e-2,
    noise=0.01,
    seed=None
):
    """
    Solve the 1D Cahn–Hilliard equation with periodic boundary conditions.

    Parameters
    ----------
    N : int
        Number of spatial grid points
    L : float
        Domain length
    dt : float
        Time step
    steps : int
        Number of time steps
    M : float
        Mobility
    kappa : float
        Gradient energy coefficient
    noise : float
        Initial noise amplitude
    seed : int or None
        Random seed

    Returns
    -------
    phi : ndarray
        Final concentration field
    """

    if seed is not None:
        np.random.seed(seed)

    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)

    # Initial condition: small random fluctuations
    phi = noise * (2 * np.random.rand(N) - 1)

    def laplacian(f):
        return (np.roll(f, -1) - 2 * f + np.roll(f, 1)) / dx**2

    for _ in range(steps):
        mu = phi**3 - phi - kappa * laplacian(phi)
        phi += dt * M * laplacian(mu)

    return x, phi


def dominant_wavelength(phi, L):
    """
    Compute dominant wavelength using FFT.
    """
    N = len(phi)
    fft_vals = np.abs(np.fft.fft(phi))
    freqs = np.fft.fftfreq(N, d=L/N)

    positive = freqs > 0
    freqs = freqs[positive]
    fft_vals = fft_vals[positive]

    k_dom = freqs[np.argmax(fft_vals)]
    wavelength = 1.0 / k_dom if k_dom != 0 else np.inf

    return wavelength


if __name__ == "__main__":
    x, phi = cahn_hilliard_1d()
    wl = dominant_wavelength(phi, L=1.0)
    print(f"Dominant wavelength: {wl:.4f}")
