import ar1_kalman_core as ac
import numpy as np

def logit(x, low, high):
    return np.log(x-low) - np.log(high-x)

def inv_logit(y, low, high):
    if y < 0:
        ey = np.exp(y)
        return (ey*high + low)/(1.0 + ey)
    else:
        ey = np.exp(-y)
        return (high + ey*low)/(1.0 + ey)

def logit_log_jacobian(y, low, high):
    if y < 0:
        return np.log(high-low) + y - 2*np.log1p(np.exp(y))
    else:
        return np.log(high-low) - y - 2*np.log1p(np.exp(-y))

    if y < 0:
        ey = np.exp(y)
        return ey*(high-low)/np.square(1 + ey)
    else:
        ey = np.exp(-y)
        return ey*(high-low)/np.square(1 + ey)

class AR1KalmanPosterior(object):
    def __init__(self, t, y, dy, tau_low=None, tau_high=None, sigma_low=None, sigma_high=None):
        self._t = t
        self._y = y
        self._dy = dy
        self._vy = dy*dy
        self._scale_low = scale_low
        self._scale_high = scale_high

        if sigma_low is None:
            self._sigma_low = np.std(y)*0.1
        else:
            self._sigma_low = sigma_low
        if sigma_high is None:
            self._sigma_high = np.std(y)*10.0
        else:
            self._sigma_high = sigma_high

        if tau_low is None:
            self._tau_low = scale_low*np.min(np.diff(t))
        else:
            self._tau_low = tau_low

        if tau_high is None:
            self._tau_high = scale_high*(t[-1] - t[0])
        else:
            self._tau_high = tau_high
            
        self._wn_var = np.trapz(dy*dy, t)/(t[-1] - t[0])

    @property
    def t(self):
        return self._t
    @property
    def y(self):
        return self._y
    @property
    def dy(self):
        return self._dy
    @property
    def vy(self):
        return self._vy
    @property
    def scale_low(self):
        return self._scale_low
    @property
    def scale_high(self):
        return self._scale_high
    @property
    def sigma_low(self):
        return self._sigma_low
    @property
    def sigma_high(self):
        return self._sigma_high
    @property
    def tau_low(self):
        return self._tau_low
    @property
    def tau_high(self):
        return self._tau_high
    @property
    def wn_var(self):
        return self._wn_var

    @property
    def nparams(self):
        return 4

    @property
    def dtype(self):
        return np.dtype([('mu', np.float),
                         ('logitsigma', np.float),
                         ('logittau', np.float),
                         ('logitnu', np.float)])

    def to_params(self, p):
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def log_prior(self, p):
        p = self.to_params(p)

        lnu = p['logitnu']
        nu = inv_logit(lnu, self.scale_low, self.scale_high)
        lp = logit_log_jacobian(lnu, self.scale_low, self.scale_high)
        lp -= np.log(nu)

        ls = p['logitsigma']
        s = inv_logit(ls, self.sigma_low, self.sigma_high)
        lp += logit_log_jacobian(ls, self.sigma_low, self.sigma_high)
        lp -= np.log(s)

        lt = p['logittau']
        t = inv_logit(lt, self.tau_low, self.tau_high)
        lp += logit_log_jacobian(lt, self.tau_low, self.tau_high)
        lp -= np.log(t)

        return lp

    def predict(self, p):
        p = self.to_params(p)

        sigma = inv_logit(p['logitsigma'], self.sigma_low, self.sigma_high)
        tau = inv_logit(p['logittau'], self.tau_low, self.tau_high)
        nu = inv_logit(p['logitnu'], self.scale_low, self.scale_high)

        t = self.t
        y = self.y - p['mu']
        
        vy = nu*nu*self.vy

        yp = np.zeros(self.y.shape)
        vyp = np.zeros(self.y.shape)

        ac.predict(t, y, vy, yp, vyp, sigma, tau)

        yp = yp + p['mu']

        return yp, vyp

    def log_likelihood(self, p):
        p = self.to_params(p)

        yp, vyp = self.predict(p)

        chi2 = np.sum(np.square(self.y - yp)/vyp)

        return -0.5*chi2 - 0.5*self.y.shape[0]*np.log(2*np.pi) - 0.5*np.sum(np.log(vyp))

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return lp
        else:
            return self.log_likelihood(p) + lp

    def parameterise(self, mu, sigma, tau, nu):
        p = self.to_params(np.zeros(self.nparams))

        p['mu'] = mu
        p['logitsigma'] = logit(sigma, self.sigma_low, self.sigma_high)
        p['logittau'] = logit(tau, self.tau_low, self.tau_high)
        p['logitnu'] = logit(nu, self.scale_low, self.scale_high)

        return p.reshape((1,)).view(float)

    def deparameterise(self, p):
        p = self.to_params(p)

        mu = p['mu']
        sigma = inv_logit(p['logitsigma'], self.sigma_low, self.sigma_high)
        tau = inv_logit(p['logittau'], self.tau_low, self.tau_high)
        nu = inv_logit(p['logitnu'], self.scale_low, self.scale_high)

        return mu, sigma, tau, nu

    def ar1_power_spectrum(self, p, fs):
        mu, sigma, tau, nu = self.deparameterise(p)

        return sigma*sigma*4.0*tau / (1.0 + 4.0*np.pi*np.pi*tau*tau*fs*fs)

    def wn_power_spectrum(self, p, fs):
        mu, sigma, tau, nu = self.deparameterise(p)

        v = nu*nu*self.wn_var
        bw = fs[-1] - fs[0]

        return v/bw

    def power_spectrum(self, p, fs):
        return self.ar1_power_spectrum(p, fs) + self.wn_power_spectrum(p, fs)
