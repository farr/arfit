import numpy as np

class AR1Posterior(object):
    r"""Represents the posterior for an AR(1) process at arbitrary sample
    times:

    .. math::

      x_{i+1} = \alpha_i x_i + \beta_i \epsilon

    with :math:`\epsilon \sim N(0,1)` and :math:`\alpha_i` and
    :math:`\beta_i` chosen so that the process has a consistent ACF,
    given by 

    .. math::

      \rho(\Delta t) = \sigma^2 \exp\left( - \frac{\Delta t}{\tau} \right)

    where :math:`\sigma^2` is the variance of the process, and
    :math:`\tau` is the autocorrelation time for the process.

    """

    def __init__(self, times, samples):
        """:param times: The sample times.

        :param samples: The data samples.

        """
        self._times = times
        self._samples = samples

    @property
    def times(self):
        """The sample times."""
        return self._times

    @property
    def samples(self):
        """The data samples."""
        return self._samples

    @property
    def dtype(self):
        """Returns a data type appropriate for parameters.  See :meth:`to_params`.

        """
        return np.dtype([('lnsigma', np.float), ('lntau', np.float)])

    def to_params(self, p):
        r"""Returns a view of the array ``p`` that corresponds to a
        parameterization of the AR(1) process.  There are two
        parameter names:

        ``lnsigma``
          The natural log of :math:`\sigma`

        ``lntau``
          The natural log of :math:`\tau`

        """
        p = np.atleast_1d(p)

        return p.view(self.dtype)

    def _alphas_betas(self, p):
        p = self.to_params(p)

        sigma = np.exp(p['lnsigma'])
        tau = np.exp(p['lntau'])
        dts = np.diff(self.times)
        alphas = np.exp(-dts/tau)
        betas = sigma * np.sqrt(1.0 - alphas*alphas)

        betas = np.concatenate(([sigma[0]], betas))

        return alphas, betas

    def whitened_residuals(self, p):
        """Returns a series of residuals that should be independently,
        :math:`N(0,1)` distributed for the given parameters.

        """

        alphas, betas = self._alphas_betas(p)

        ys = self.samples.copy()
        ys[1:] = (ys[1:] - alphas*ys[0:-1])/betas[1:]
        ys[0] = ys[0] / betas[0]

        return ys

    def power_spectrum(self, p, fs):
        r"""Returns the power spectrum at the indicated frequencies for the
        AR(1) process represented by the given parameters.  The power
        spectrum is given by 

        .. math::

          P(f) = \frac{4 \sigma^2 \tau}{4 \pi^2 f^2 \tau^2 + 1}

        """

        p = self.to_params(p)

        sigma = np.exp(p['lnsigma'])
        tau = np.exp(p['lntau'])

        return 4.0*sigma*sigma*tau*tau/(np.square(2.0*np.pi*tau*fs) + 1)

    def log_prior(self, p):
        r"""Returns the log of the prior function on parameters.  The prior is
        uniform in :math:`\ln(\sigma)` and :math:`\ln(\tau)`.

        """

        return 0.0

    def log_likelihood(self, p):
        r"""Returns the log of the likelihood for the stored data given
        parameters ``p``.  The likelihood is computed in time
        proportional to :math:`\mathcal{O}(N)`, where :math:`N` is the
        number of data samples.

        """
        alphas, betas = self._alphas_betas(p)

        ys = self.samples.copy()
        ys[1:] = (ys[1:] - alphas*ys[0:-1])/betas[1:]
        ys[0] = ys[0] / betas[0]
        
        return -0.5*ys.shape[0]*np.log(2.0*np.pi) - np.sum(np.log(betas)) - 0.5*np.sum(np.square(ys))

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return np.NINF
        else:
            return lp + self.log_likelihood(p)

    def draw_data(self, p):
        """Returns a draw of data from the model with parameters p.

        """

        alphas, betas = self._alphas_betas(p)

        ys = [np.random.randn(1)*betas[0]]
        for a,b in zip(alphas, betas[1:]):
            ys.append(a*ys[-1] + b*np.random.randn(1))

        return np.array(ys).squeeze()
