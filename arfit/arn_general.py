import numpy as np

"""A general, slow model of CAR(n) processes.

"""

def greens_coefficients(roots):
    r"""Returns the coefficients of the Green's function for the given
    roots of the ODE characteristic equation.

    The Green's function can always be written as 

    .. math::

      G(t; \xi) = \sum_i g_i \exp\left(r_i (t-\xi)\right)

    This function returns the :math:`g_i`.

    """
    roots = np.atleast_1d(roots)
    n = roots.shape[0]

    m = np.zeros((n,n), dtype=roots.dtype) # complex if necessary

    m[0,:] = np.ones(n)
    for i in range(1,n):
        m[i,:] = m[i-1,:]*roots

    b = np.zeros(n, dtype=roots.dtype)
    b[-1] = 1.0

    return np.linalg.solve(m, b)

def covariance_matrix(roots, ts):
    r"""Returns the covariance matrix of the stochastic solution to the ODE

    .. math::

      \left[ \prod_i \frac{d}{dt} - r_i \right] y(t) = \eta(t),

    with 

    .. math::

      \left\langle \eta(s) \eta(t) \right\rangle = \delta(s-t),

    at the indicated times.  In other words, returns 

    .. math::

      C_{ij} = \left\langle y\left(t_i\right) y\left(t_j \right) \right\rangle.

    """
    roots = np.atleast_1d(roots)
    ts = np.atleast_1d(ts)

    g = greens_coefficients(roots)

    dts = np.abs(ts.reshape((-1, 1)) - ts.reshape((1, -1)))

    m = np.zeros(dts.shape, dtype=roots.dtype)

    for ri, gi in zip(roots, g):
        for rj, gj in zip(roots, g):
            m -= gi*gj*np.exp(dts*rj)/(ri + rj)

    return np.real(m) # Covariance is always real

def generate_data(sigma, roots, ts):
    r"""Returns a sample from the stochastic process described by

    .. math::

      \left[ \prod_i \frac{d}{dt} - r_i \right] y(t) = \sigma \eta(t)

    with 

    .. math::

      \left\langle \eta(t) \eta(s) \right\rangle = \delta(s - t)

    at the indicated times.

    """

    ts = np.atleast_1d(ts)

    m = covariance_matrix(roots, ts)*sigma*sigma

    return np.random.multivariate_normal(np.zeros(ts.shape[0]), m)

def psd(sigma, roots, fs):
    r"""Returns the power spectral density of the stochastic process

    .. math::

      \left\langle \tilde{y}\left(f_1\right) \tilde{y}\left(f_2\right) \right\rangle = P\left(f_1\right) \delta\left( f_1 - f_2 \right)

    """

    fs = np.atleast_1d(fs)
    roots = np.atleast_1d(roots)

    denom = (2.0*np.pi*1j*fs.reshape((-1, 1)) - roots.reshape((1, -1)))
    denom = np.prod(denom, axis=1)
    denom = np.abs(denom)
    denom = denom*denom

    return sigma*sigma/denom
