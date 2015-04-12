import numpy as np
import scipy.linalg as sl

class AR1Posterior(object):
    r"""Represents the posterior for an AR(1) process at arbitrary sample
    times:

    .. math::

      x_{i+1} - \mu = \alpha_i \left(x_i - \mu\right) + \beta_i \epsilon

    with :math:`\epsilon \sim N(0,1)`, :math:`\alpha_i` and
    :math:`\beta_i` chosen so that the process has a consistent ACF,
    given by 

    .. math::

      \rho(\Delta t) = \sigma^2 \left( (1-\nu) \exp\left( - \frac{\Delta t}{\tau} \right) + \nu \delta(\Delta t) \right)

    where :math:`\sigma^2` is the variance of the process, and
    :math:`\tau` is the autocorrelation time for the process.

    """

    def __init__(self, ts, ys):
        """:param times: The sample times.

        :param samples: The data samples.

        """
        self._ts = ts
        self._ys = ys

    @property
    def ts(self):
        """Sample times again."""

        return self._ts

    @property
    def ys(self):
        """Samples."""
        return self._ys

    @property
    def T(self):
        """Total time.

        """
        return self.ts[-1] - self.ts[0]

    @property
    def dt_min(self):
        """Minimum step.

        """
        return np.min(np.diff(self.ts))

    @property
    def dtype(self):
        """Returns a data type appropriate for parameters.  See :meth:`to_params`.

        """
        return np.dtype([('mu', np.float), ('lnsigma', np.float),
                         ('lntau', np.float), ('logitnu', np.float)])

    @property
    def nparams(self):
        """Returns the number of parameters.

        """
        return 4

    def to_params(self, p):
        r"""Returns a view of the array ``p`` that corresponds to a
        parameterization of the AR(1) process.  There are two
        parameter names:

        ``mu``
          The mean of the process.

        ``lnsigma``
          The natural log of :math:`\sigma`

        ``lntau``
          The natural log of :math:`\tau`

        ``logitnu``
          The logit of the :math:`\nu` parameter giving the fraction
          of white and coloured noise

        """
        p = np.atleast_1d(p)

        return p.view(self.dtype)

    def _alphas_betas(self, p):
        p = self.to_params(p)

        sigma = np.exp(p['lnsigma'])
        tau = np.exp(p['lntau'])
        dts = np.diff(self.ts)
        alphas = np.exp(-dts/tau)
        betas = sigma * np.sqrt(1.0 - alphas*alphas)

        betas = np.concatenate(([sigma[0]], betas))

        return alphas, betas

    def _inv_logit(self, y):
        if y < 0:
            ey = np.exp(y)
            return y/(1+y)
        else:
            return 1.0/(1.0+np.exp(-y))

    def _banded_covariance(self, p):
        p = self.to_params(p)

        sigma = np.exp(p['lnsigma'])
        nu = self._inv_logit(p['logitnu'])

        alphas, betas = self._alphas_betas(p)

        bc = np.zeros((2, self.ts.shape[0]))
        bc[0, :] = (1-nu)*betas*betas

        bc[0, :] += nu*sigma*sigma
        bc[0, 1:] += nu*sigma*sigma*np.square(alphas)

        bc[1, :-1] = -alphas*nu*sigma*sigma

        return bc

    def residuals(self, p):
        """Returns a series of residuals that should be independent,
        normally-distributed.
        """

        p = self.to_params(p)

        alphas, betas = self._alphas_betas(p)

        ys = self.samples.copy() - p['mu']
        ys[1:] = (ys[1:] - alphas*ys[0:-1])

        return ys

    def prediction(self, p):
        """Returns a series of predicted values at the sample times
        given parameters p.
        """
        
        p = self.to_params(p)

        alphas, betas = self._alphas_betas(p)

        ys = self.samples.copy() - p['mu']

        preds = np.zeros(ys.shape)

        preds[1:] = alphas*ys[0:-1]

        preds += p['mu']

        return preds, betas

    def power_spectrum(self, fs, p):
        r"""Returns the power spectrum at the indicated frequencies for the
        AR(1) process represented by the given parameters.  The power
        spectrum is given by 

        .. math::

          P(f) = \frac{4 \sigma^2 \tau}{4 \pi^2 f^2 \tau^2 + 1}

        """

        p = self.to_params(p)

        sigma = np.exp(p['lnsigma'])
        tau = np.exp(p['lntau'])

        return 4.0*sigma*sigma*tau/(np.square(2.0*np.pi*tau*fs) + 1)

    def log_prior(self, p):
        r"""Returns the log of the prior function on parameters.  The prior is
        uniform in :math:`\ln(\sigma)` and :math:`\ln(\tau)`.

        """

        # Flat priors in everything except logit-nu.
        # Flat prior in nu.

        lp = 0.0
        p = self.to_params(p)

        lnu = p['logitnu']

        if lnu < 0:
            lp += lnu - 2.0*np.log1p(np.exp(lnu))
        else:
            lp += -lnu - 2.0*np.log1p(np.exp(-lnu))

        return lp

    def log_likelihood(self, p):
        r"""Returns the log of the likelihood for the stored data given
        parameters ``p``.  The likelihood is computed in time
        proportional to :math:`\mathcal{O}(N)`, where :math:`N` is the
        number of data samples.

        """
        p = self.to_params(p)

        alphas, betas = self._alphas_betas(p)
        bc = self._banded_covariance(p)

        ys = self.ys.copy() - p['mu']
        ys[1:] = ys[1:] - alphas*ys[0:-1]

        dts = self.ts.reshape((-1, 1)) - self.ts.reshape((1, -1))
        tau = np.exp(p['lntau'])
        nu = self._inv_logit(p['logitnu'])
        sigma = np.exp(p['lnsigma'])
        full_cov = sigma*sigma*((1-nu)*np.exp(-np.abs(dts)/tau) + nu)

        llow = sl.cholesky_banded(bc, lower=True)

        logdet = np.sum(np.log(llow[0, :]))

        return -0.5*self.ts.shape[0]*np.log(2.0*np.pi) - logdet - 0.5*np.dot(ys, sl.cho_solve_banded((llow, True), ys))

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return np.NINF
        else:
            return lp + self.log_likelihood(p)
