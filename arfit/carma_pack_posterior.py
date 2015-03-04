import carmcmc as cm
import numpy as np

class Posterior(object):
    def __init__(self, t, y, dy, p=2, q=1):
        self._t = t.copy()
        self._y = y.copy()
        self._dy = dy.copy()

        self._p = p
        self._q = q

        self._tv = cm.vecD()
        self._tv.extend(t)

        self._yv = cm.vecD()
        self._yv.extend(y)

        self._dyv = cm.vecD()
        self._dyv.extend(dy)

        self._carma_process = cm.run_mcmc_carma(1, 25, self._tv, self._yv, self._dyv, self.p, self.q, 10, False, 1)

        self._carma_process.SetMLE(True)

        self._T = self.t[-1] - self.t[0]
        self._dt_min = np.min(np.diff(self.t))

        self._mean_variance()

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
    def p(self):
        return self._p
    @property
    def q(self):
        return self._q
    @property
    def nparams(self):
        return 3 + self.p + self.q
    @property
    def dtype(self):
        type = [('mu', np.float),
                ('log_sigma', np.float),
                ('log_nu', np.float), # Scale factor for variance!
                ('log_quad_p', np.float, self.p)]
        
        if self.q > 0:
            type.append(('log_quad_q', np.float, self.q))

        return np.dtype(type)
    @property
    def mu(self):
        return self._mu
    @property
    def var(self):
        return self._var
    @property
    def wn_var(self):
        return self._wn_var
    @property
    def std(self):
        return self._std
    @property
    def T(self):
        return self._T
    @property
    def dt_min(self):
        return self._dt_min
    @property
    def root_min(self):
        return 1.0/(2.0*self.T)
    @property
    def root_max(self):
        return 2.0/self.dt_min

    def _mean_variance(self):
        T = self.T
        
        mu = np.trapz(self.y, self.t) / T
        var = np.trapz(np.square(self.y - mu), self.t) / T

        self._mu = mu
        self._var = var
        self._std = np.sqrt(var)

        self._wn_var = np.trapz(np.square(self.dy), self.t) / T

    def _sorted_roots(self, r):
        real_sel = np.imag(r) == 0.0
        cplx_roots = r[~real_sel]
        real_roots = r[real_sel]

        sorted_cplx_roots = cplx_roots[np.argsort(-np.abs(np.imag(cplx_roots)))]

        sorted_real_roots = []
        real_roots = np.sort(real_roots)
        while len(real_roots) > 1:
            imin = np.argmin(np.diff(real_roots))
            r1 = real_roots[imin]
            r2 = real_roots[imin+1]

            sorted_real_roots.append(r1)
            sorted_real_roots.append(r2)

            real_roots = np.concatenate((real_roots[:imin], real_roots[imin+2:]))

        if len(real_roots) == 1:
            sorted_real_roots.append(real_roots[0])

        return np.concatenate((sorted_cplx_roots, sorted_real_roots))            

    def _roots_to_quad(self, r):
        r = self._sorted_roots(r)

        quads = []
        for r1, r2 in zip(r[::2], r[1::2]):
            b = np.real(-(r1+r2))
            c = np.real(r1*r2)
            quads.append(c)
            quads.append(b)

        if r.shape[0] % 2 == 1:
            quads.append(-np.real(r[-1]))

        return np.log(quads)

    def _quad_to_roots(self, qp):
        if qp.shape[0] % 2 == 0:
            cs = np.exp(qp[::2])
            bs = np.exp(qp[1::2])

            discs2 = bs*bs - 4.0*cs

            roots = np.zeros(qp.shape[0], dtype=np.complex)

            disc_factors = np.ones(qp.shape[0]/2, dtype=np.complex)
            disc_factors[discs2 < 0] = 1.0j

            ys = disc_factors*np.sqrt(np.abs(discs2))
            
            roots[::2] = 0.5*(-bs - ys)
            roots[1::2] = 0.5*(-bs + ys)

            return roots
        else:
            root = -np.exp(qp[-1])
            roots = self._quad_to_roots(qp[:-1])
            return np.concatenate((roots, [root]))

    def ar_roots(self, p):
        p = self.to_params(p)

        return self._quad_to_roots(p['log_quad_p'])

    def ma_roots(self, p):
        p = self.to_params(p)

        return self._quad_to_roots(p['log_quad_q'])

    def ar_poly(self, p):
        return np.real(np.poly(self.ar_roots(p)))

    def ma_poly(self, p):
        poly = np.real(np.poly(self.ma_roots(p)))

        return poly / poly[-1]

    def resort_params(self, p):
        p = self.to_params(p).copy()

        ps = p['log_quad_p']
        roots = self._quad_to_roots(ps)
        roots = self._sorted_roots(roots)
        p['log_quad_p'] = self._roots_to_quad(roots)

        ps = p['log_quad_q']
        roots = self._quad_to_roots(ps)
        roots = self._sorted_roots(roots)
        p['log_quad_q'] = self._roots_to_quad(roots)

        return p
        
    def to_params(self, p):
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def to_carmapack_params(self, p):
        p = self.to_params(p)

        pc = np.zeros(self.nparams)

        pc[0] = np.exp(p['log_sigma'])
        pc[1] = np.exp(p['log_nu'])
        pc[2] = p['mu']
        pc[3:3+self.p] = p['log_quad_p']
        pc[3+self.p:] = p['log_quad_q']

        return pc

    def carmapack_to_params(self, pc):
        p = self.to_params(np.zeros(self.nparams))

        p['log_sigma'] = np.log(pc[0])
        p['log_nu'] = np.log(pc[1])
        p['mu'] = pc[2]
        p['log_quad_p'] = pc[3:3+self.p]
        p['log_quad_q'] = pc[3+self.p:]

        return p

    def draw_roots(self, n):
        roots_real = np.random.uniform(low=-self.root_max, high=-self.root_min, size=n/2)
        roots_imag = 1j*np.random.uniform(low=0, high=self.root_max, size=n/2)

        roots1 = roots_real + roots_imag

        if n % 2 == 1:
            roots1 = np.concatenate((roots1, [np.random.uniform(low=-self.root_max, high=-self.root_min)]))
        
        roots2 = roots_real - roots_imag

        roots = np.zeros(n, dtype=np.complex)
        roots[..., ::2] = roots1
        roots[..., 1::2] = roots2

        return self._sorted_roots(roots)

    def draw_prior(self):
        mus = np.random.uniform(low=self.mu - 5*self.std,
                                high=self.mu + 5*self.std)
        log_sigmas = np.random.uniform(low=np.log(self.std/10.0),
                                       high=np.log(self.std*10.0))
        log_nus = np.random.uniform(low=np.log(0.1),
                                    high=np.log(10.0))
        ar_quad = self._roots_to_quad(self.draw_roots(self.p))
        ma_quad = self._roots_to_quad(self.draw_roots(self.q))

        return np.concatenate(([mus, log_sigmas, log_nus], ar_quad, ma_quad))

    def log_prior(self, p):
        p = self.to_params(p)

        # Prior ranges
        if p['mu'] < self.mu - 5*self.std or p['mu'] > self.mu + 5*self.std:
            return np.NINF
        if p['log_sigma'] < np.log(self.std/10.0) or p['log_sigma'] > np.log(10.0*self.std):
            return np.NINF
        if p['log_nu'] < np.log(0.1) or p['log_nu'] > np.log(10.0):
            return np.NINF

        root_max = self.root_max
        root_min = self.root_min

        ar_roots = self._quad_to_roots(p['log_quad_p'])
        ma_roots = self._quad_to_roots(p['log_quad_q'])

        if np.any(np.real(ar_roots) < -root_max) or np.any(np.real(ar_roots) > -root_min):
            return np.NINF
        if np.any(np.real(ma_roots) < -root_max) or np.any(np.real(ma_roots) > -root_min):
            return np.NINF
        if np.any(np.abs(np.imag(ar_roots)) > root_max):
            return np.NINF
        if np.any(np.abs(np.imag(ma_roots)) > root_max):
            return np.NINF

        # Root uniqueness---this introduces some *major* discontinuities!
        ar_roots_sorted = self._sorted_roots(ar_roots)
        ma_roots_sorted = self._sorted_roots(ma_roots)

        if not np.all(ar_roots_sorted == ar_roots):
            return np.NINF
        if not np.all(ma_roots_sorted == ma_roots):
            return np.NINF

        # Otherwise, flat prior
        return 0.0

    def log_likelihood(self, p):
        pc = self.to_carmapack_params(p)

        pcv = cm.vecD()
        pcv.extend(pc)

        return self._carma_process.getLogDensity(pcv)

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return lp
        else:
            return lp + self.log_likelihood(p)

    def __getstate__(self):
        d = self.__dict__.copy()

        del d['_tv']
        del d['_yv']
        del d['_dyv']
        del d['_carma_process']

        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

        self._tv = cm.vecD()
        self._tv.extend(self.t)

        self._yv = cm.vecD()
        self._yv.extend(self.y)

        self._dyv = cm.vecD()
        self._dyv.extend(self.dy)

        self._carma_process = cm.run_mcmc_carma(1, 25, self._tv, self._yv, self._dyv, self.p, self.q, 10, False, 1)
        self._carma_process.SetMLE(True)

    def power_spectrum(self, fs, p):
        p = self.to_params(p)

        ar_roots = self._quad_to_roots(p['log_quad_p'])
        ma_roots = self._quad_to_roots(p['log_quad_q'])

        ar_coefs = np.real(np.poly(ar_roots))

        ma_coefs = np.real(np.poly(ma_roots))
        ma_coefs /= ma_coefs[-1]
        ma_coefs = ma_coefs[::-1]

        sigma = np.exp(p['log_sigma']) / np.sqrt(cm.carma_variance(1.0, ar_roots, ma_coefs))

        return cm.power_spectrum(fs, sigma, ar_coefs, ma_coefs)

    def white_noise(self, p, bw):
        p = self.to_params(p)

        return self.wn_var * np.exp(p['log_nu']) / bw

    def standardised_residuals(self, p):
        p = self.to_params(p)

        kfilter = self._make_kalman_filter(p)

        kmean = np.asarray(kfilter.GetMean())
        kvar = np.asarray(kfilter.GetVar())

        return (self.y - p['mu'] - kmean) / np.sqrt(kvar)

    def predict(self, p, ts):
        p = self.to_params(p)

        kfilter = self._make_kalman_filter(p)

        ypred = []
        ypred_var = []
        for t in ts:
            yp = kfilter.Predict(t)
            ypred.append(yp.first)
            ypred_var.append(yp.second)

        return np.array(ypred) + p['mu'], np.array(ypred_var)

    def _make_kalman_filter(self, p):
        p = self.to_params(p)

        ar_roots = self.ar_roots(p)
        ma_coefs = self.ma_poly(p)

        sigma = np.exp(p['log_sigma']) / np.sqrt(cm.carma_variance(1.0, ar_roots, ma_coefs))
        sigmasq = sigma*sigma
        
        tv = cm.vecD()
        tv.extend(self.t)
        yv = cm.vecD()
        yv.extend(self.y - p['mu'])
        dyv = cm.vecD()
        dyv.extend(self.dy)
        arv = cm.vecC()
        arv.extend(ar_roots)
        mav = cm.vecD()
        mav.extend(ma_coefs)
        kfilter = cm.KalmanFilterp(tv, yv, dyv, sigmasq, arv, mav)

        kfilter.Filter()

        return kfilter
