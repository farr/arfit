import arn_loops as al
import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
import warnings

r"""A model based on CAR processes for fitting the properties of noise.
Our model is that the measured noise, :math:`y_i = y\left( t_i
\right)` obeys the stochastic differential equation 

.. math::

  \left[ \prod_{i = 1}^p \frac{d}{dt} - r_i \right] y(t) = \eta(t)

where :math:`\eta(t)` is a white-noise process with 

.. math::

  \left\langle \eta(t) \right \rangle = 0

and 

.. math::

  \left\langle \eta(t) \eta(t') \right\rangle = \sigma^2 \delta(t-t')

The parameters of our model are the p "roots" :math:`r_i` and the
noise amplitude :math:`\sigma`.  We use the measured times and values
:math:`y_i = y(t_i)` to constrain these parameters.  Note that the
roots can be complex, but if so must occur in complex-conjugate pairs
so that the noise remains real.

The roots have dimensions of inverse time, and correspond to the
reciprocal of the decay timescale of an eigenmode of the process (real
part) and/or to an angular frequency of an eigenmode of the process
(imaginary part).

The power spectrum of the noise process described by our model is 

.. math::

  \left\langle \tilde{y}(f) \tilde{y}(f') \right\rangle = \frac{\sigma^2}{\prod_{j=1}^p \left| 2 \pi i f - r_j \right|^2} \delta(f - f')

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

def banded_covariance(roots, ts):
    """Returns only the terms in the covariance matrix that one might need
    to calculate the covariance of the reduced data:

    .. math::

      x_i = y_i - \sum_{k=1}^p \alpha_k y_{i-k}

    Only the upper :math:`p+1` diagonals of the covariance matrix will
    be returned.

    """

    roots = np.atleast_1d(roots)
    ts = np.atleast_1d(ts)

    n = ts.shape[0]
    p = roots.shape[0]

    g = greens_coefficients(roots)

    dts = np.zeros((2*p+1, n))
    for i in range(1, 2*p+1):
        j = 2*p - i
        dts[j, i:] = np.abs(ts[:-i] - ts[i:])

    m = np.zeros(dts.shape, dtype=roots.dtype)

    for ri, gi in zip(roots, g):
        for rj, gj in zip(roots, g):
            m -= gi*gj*np.exp(dts*rj)/(ri + rj)
            
    for i in range(2*p):
        m[i, :(2*p-i)] = 0.0

    return np.real(m)

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

def alpha(roots, ts):
    r"""Returns the weights in the difference equation that the homogeneous
    solution at times ``ts`` satisfies:

    .. math::

      y_h\left( t_p \right) = \sum_{i = 0}^{p-1} \alpha_i y_h\left( t_i \right)

    """
    roots = np.atleast_1d(roots)
    ts = np.atleast_1d(ts)

    n = ts.shape[0]
    p = roots.shape[0]

    many_ts = np.zeros((n-p, p+1))
    for j in range(p+1):
        many_ts[:,j] = ts[j:n-p+j]
    many_ts = many_ts - many_ts[:,0].reshape((-1, 1))
    
    m = np.exp(roots.reshape((1, -1, 1))*many_ts[:,:-1].reshape((n-p, 1, -1)))
    b = np.exp(roots.reshape((1, -1))*many_ts[:, -1].reshape((n-p, 1)))

    return np.real(np.linalg.solve(m,b))

class BadParameterWarning(Warning):
    """Used to indicate a bad region of parameter space in the likelihood.

    """

    pass

class Posterior(object):
    """A posterior for the AR(n) process.

    """

    def __init__(self, ts, ys, p, ncomplex=0):
        """Initialise the posterior with the given times and data.  If given,
        ``ncomplex`` indicates the number of complex roots
        (i.e. oscillatory components) in the process.  ``p`` is the
        order of the process.

        """

        self._ts = ts
        self._ys = ys
        self._p = p
        self._nc = ncomplex

    @property
    def ts(self):
        """Sample times.

        """
        return self._ts

    @property
    def ys(self):
        """Sample values

        """
        return self._ys

    @property
    def p(self):
        """The order of the CAR process.

        """
        return self._p

    @property
    def n(self):
        """The number of data points.

        """
        return self.ts.shape[0]

    @property
    def nc(self):
        """The number of complex roots (should always be even).

        """
        return self._nc

    @property
    def dtype(self):
        """A named-field data type for model parameters.

        """
        return np.dtype([('log_sigma', np.float),
                         ('root_params', np.float, self.p)])

    @property
    def nparams(self):
        """The number of parameters in the model.

        """
        return 1 + self.p

    def to_params(self, p):
        r"""Returns a view of ``p`` with named fields for the parameters.  Our
        parameterisation uses :math:`\log \sigma` and treats the roots
        as follows.  For the real roots, the roots are sorted from
        smallest to largest, and 

        .. math::

          p_0 = - \log\left( -r_0 \right)

        .. math::

          p_i = \log\left( r_i - r_{i-1} \right) - \log\left( -r_i \right)

        In this way, :math:`-\infty < p_j < \infty` generates all
        possible sorted sets of negative roots.

        We use the same parameterisation for the real parts of the
        complex roots.  For the imaginary parts, we use the log of the
        positive angular frequency as a parameter, again generating
        all possible complex conjugate pairs of roots (sorted in
        increasing order of real part) as the parameters range over
        the real line.

        """
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def _real_roots(self, rp):
        rp = np.atleast_1d(rp)

        rs = np.zeros(rp.shape[0])

        rs[0] = -np.exp(-rp[0])
        for i in range(1, rp.shape[0]):
            rs[i] = rs[i-1]/(1.0 + np.exp(rp[i]))

        return rs

    def _complex_roots(self, rp):
        rp = np.atleast_1d(rp)
        realp = rp[::2]
        imagp = rp[1::2]

        # The real parts of the complex roots are stored in the same
        # way as the real roots
        reals = self._real_roots(realp)
        imags = np.exp(imagp)*1j

        roots = np.zeros(rp.shape[0], dtype=np.complex)
        roots[::2] = reals + imags
        roots[1::2] = reals - imags

        return roots

    def roots(self, p):
        """Returns the roots of the characteristic polynomial given the model
        parameters.

        """

        p = self.to_params(p)

        if self.nc == 0:
            return self._real_roots(p['root_params'])
        elif self.nc == self.p:
            return self._complex_roots(p['root_params'])
        else:
            rp = p['root_params']
            # Complex roots stored first
            crs = self._complex_roots(rp[:self.nc])
            rrs = self._real_roots(rp[self.nc:])

            return np.concatenate((crs, rrs))

    def _real_params(self, roots):
        roots = np.sort(roots)

        ps = np.zeros(roots.shape[0])

        ps[0] = -np.log(-roots[0])
        for i in range(1, roots.shape[0]):
            ps[i] = np.log(roots[i] - roots[i-1]) - np.log(-roots[i])

        return ps

    def _complex_params(self, roots):
        roots = roots[np.argsort(np.real(roots))]
        ps = np.zeros(roots.shape[0])

        ps[::2] = self._real_params(np.real(roots[::2]))
        ps[1::2] = np.log(np.abs(np.imag(roots[::2])))

        return ps

    def parameters(self, roots):
        """Returns the root parameters associated with the given roots.

        """

        if self.nc == 0:
            return self._real_params(roots)
        elif self.nc == self.p:
            return self._complex_params(roots)
        else:
            return np.concatenate((self._complex_params(roots[:self.nc]), self._real_params(roots[self.nc:])))

    def _real_roots_log_jacobian(self, p, r):
        p = np.atleast_1d(p)

        lj = -p[0]
        for i in range(1, p.shape[0]):
            lj += np.log(-r[i-1]) + p[i] - 2.0*np.log1p(np.exp(p[i]))

        return lj

    def _complex_roots_log_jacobian(self, p, r):
        p = np.atleast_1d(p)

        real_lj = self._real_roots_log_jacobian(p[::2], np.real(r))
        imag_lj = np.sum(p[1::2])

        return real_lj + imag_lj

    def roots_log_jacobian(self, p, r):
        r"""Given parameters ``p`` and associated roots ``r``, return the log
        of the jacobian transformation

        .. math::

          \left| \frac{\partial r}{\partial p} \right|

        """

        p = self.to_params(p)
        r = np.atleast_1d(r)

        if self.nc == 0:
            return self._real_roots_log_jacobian(p['root_params'], r)
        elif self.nc == self.p:
            return np.real(self._complex_roots_log_jacobian(p['root_params'], r))
        else:
            ljc = self._complex_roots_log_jacobian(p['root_params'][:self.nc], r[:self.nc])
            ljr = self._real_roots_log_jacobian(p['root_params'][self.nc:], r[self.nc:])
            return np.real(ljc + ljr)

    def _complex_roots_log_prior(self, croots, tau_min, tau_max):
        omega_min = 2.0*np.pi/tau_max
        omega_max = 2.0*np.pi/tau_min

        root_min = -1.0/tau_min
        root_max = -1.0/tau_max

        if np.any(np.real(croots) < root_min) or np.any(np.real(croots) > root_max) or \
           np.any(np.abs(np.imag(croots)) < omega_min) or \
           np.any(np.abs(np.imag(croots)) > omega_max):
            return np.NINF
        else:
            return -0.5*np.sum(np.log(np.abs(np.real(croots[::2])))) - 0.5*np.sum(np.log(np.abs(np.imag(croots[1::2]))))

    def _real_roots_log_prior(self, rroots, tau_min, tau_max):
        root_min = -1.0 / tau_min
        root_max = -1.0 / tau_max

        if np.any(rroots < root_min) or np.any(rroots > root_max):
            return np.NINF
        else:
            return -np.sum(np.log(-rroots))

    def log_prior(self, p):
        r"""The prior on parameters.  Our prior is flat in :math:`\log \sigma`,
        and flat in :math:`\log r_i` (or :math:`\log \Re r_i` and
        :math:`\log \Im r_i` if complex) between 

        .. math::

          r_\mathrm{min} = \frac{1}{10T}

        and 

        .. math:

          r_\mathrm{max} = \frac{10}{\delta t_\mathrm{min}}

        where :math:`T` is the total observation time and
        :math:`\delta t_\mathrm{min}` is the smallest time spacing
        between samples.

        """

        p = self.to_params(p)

        # Flat in log_sigma
        tau_min = np.min(np.diff(self.ts))/10.0
        tau_max = (self.ts[-1] - self.ts[0])*10.0

        roots = self.roots(p)
        
        if self.nc == 0:
            return self._real_roots_log_prior(roots, tau_min, tau_max) + self.roots_log_jacobian(p, roots)
        elif self.nc == self.p:
            return np.real(self._complex_roots_log_prior(roots, tau_min, tau_max) + self.roots_log_jacobian(p, roots))
        else:
            return np.real(self._complex_roots_log_prior(roots[:self.nc], tau_min, tau_max) + self._real_roots_log_prior(roots[self.nc:], tau_min, tau_max) + self.roots_log_jacobian(p, roots))

    def full_log_likelihood(self, p):
        """Returns the log-likelihood for the data under the model by
        explicitly constructing the data covariance matrix.  This is
        foolproof, but very slow, and so probably only useful for
        testing.

        """
        p = self.to_params(p)
        roots = self.roots(p)

        sigma = np.exp(p['log_sigma'])

        cov = sigma*sigma*covariance_matrix(roots, self.ts)

        try:
            return ss.multivariate_normal.logpdf(self.ys, np.zeros(self.ys.shape[0]), cov)
        except:
            warnings.warn('exception in multivariate_normal (probably singular cov)',
                          BadParameterWarning)
            return np.NINF

    def log_likelihood(self, p):
        """Returns the log-likelihood of the data under the model, but
        efficiently.

        """
        p = self.to_params(p)

        try:
            alpha = self._alpha_matrix(p)
            beta = self._beta_matrix(p, alpha=alpha)
        except np.linalg.LinAlgError:
            warnings.warn('exception in alpha/beta computation, probably too-small timescales',
                          BadParameterWarning)
            return np.NINF

        xs = np.zeros(self.n)

        al.log_likelihood_xs_loop(self.n, self.p, alpha, self.ys, xs)

        try:
            evals = sl.eigvals_banded(beta, lower=False)
            if np.any(evals <= 0):
                warnings.warn('found non-positive-definite transformed covariance',
                              BadParameterWarning)
                return np.NINF

            log_evals = np.log(evals)

            return -0.5*self.n*np.log(2.0*np.pi) - 0.5*np.sum(log_evals) - 0.5*np.dot(xs, sl.solveh_banded(beta, xs, lower=False))
        except sl.LinAlgError:
            warnings.warn('exception in log-likelihood computation, probably singular cov',
                          BadParameterWarning)
            return np.NINF

    def __call__(self, p):
        """Returns the log-posterior at parameters ``p``.

        """
        lp = self.log_prior(p)

        if lp == np.NINF:
            return lp
        else:
            return lp + self.log_likelihood(p)

    def _alpha_matrix(self, p):
        p = self.to_params(p)

        roots = self.roots(p)

        n = self.ts.shape[0]
        pp = roots.shape[0]

        am = np.zeros((pp+1, n))
        alp = alpha(roots, self.ts)

        al.fill_alpha_loop(n, pp, alp, am)

        return am

    def _beta_matrix(self, p, alpha=None):
        p = self.to_params(p)

        sigma = np.exp(p['log_sigma'])

        roots = self.roots(p)

        n = self.ts.shape[0]
        pp = roots.shape[0]

        if alpha is None:
            alpha = self._alpha_matrix(p)

        cov = sigma*sigma*banded_covariance(self.roots(p), self.ts)

        beta = np.zeros((pp, n))

        al.beta_matrix_loop(n, pp, alpha, cov, beta)

        return beta

    def whitened_residuals(self, p):
        r"""Returns an array of the same shape as the input data, but whose
        values are indepenent :math:`N(0,1)` variables under the model
        with parameters ``p``.

        """
        p = self.to_params(p)

        alpha = self._alpha_matrix(p)
        beta = self._beta_matrix(p)

        xs = np.zeros(self.n)

        al.log_likelihood_xs_loop(self.n, self.p, alpha, self.ys, xs)

        beta12 = sl.cholesky_banded(beta, lower=False)
        beta12T = np.zeros(beta12.shape)

        for j in range(self.n):
            for i in range(j, min(j+self.p, self.n)):
                beta12T[j-i, j] = beta12[self.p-1-(i-j), i]

        return sl.solve_banded((self.p-1, 0), beta12T, xs)

    def power_spectrum(self, p, fs):
        r"""The power spectrum for the model with parameters ``p`` at the
        frequencies ``fs``.

        .. math::

          \left\langle \tilde{y}(f) \tilde{y}(f') \right\rangle = \frac{\sigma^2}{\prod_{j=1}^p \left| 2 \pi i f - r_j \right|^2} \delta(f - f')

        """
        p = self.to_params(p)

        roots = self.roots(p)

        return np.exp(2.0*p['log_sigma'])/np.prod(np.square(np.abs(2.0*np.pi*1j*fs.reshape((-1, 1)) - roots.reshape((1, -1)))), axis=1)

    def residuals(self, p):
        r"""Returns ``x, sigma``, where 

        .. math::

          x_i = y_i - \sum_{k=1}^p \alpha_{ik} y_{i-k}

        is the error in the model prediction for :math:`y_i` and 

        .. math::

          \sigma_i^2 = \mathrm{var}\left( x_i \right)

        is the variance of this quantity predicted by the model.  Note
        that the :math:`x_i` are not statistically independent, so one
        should not, e.g., form a chi-squared out of these quantities.
        For independent residuals, see the
        :method:`Posterior.whitened_residuals` method.

        """

        p = self.to_params(p)

        alpha = self._alpha_matrix(p)
        beta = self._beta_matrix(p)

        xs = np.zeros(self.n)

        al.log_likelihood_xs_loop(self.n, self.p, alpha, self.ys, xs)

        return xs, np.sqrt(beta[-1,:])
