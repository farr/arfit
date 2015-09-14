import carmcmc as cm
import numpy as np
import plotutils.parameterizations as par

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

        self._carma_process = cm.run_mcmc_carma(1, 1, self._tv, self._yv, self._dyv, self.p, self.q, 1, False, 1)

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
        type = [('logit_mu', np.float),
                ('logit_sigma', np.float),
                ('logit_nu', np.float),
                ('ar_roots_p', np.float, self.p)]
        
        if self.q > 0:
            type.append(('ma_roots_p', np.float, self.q))

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

    @property
    def mu_min(self):
        return self.mu - 5.0*self.std
    @property
    def mu_max(self):
        return self.mu + 5.0*self.std

    @property
    def sigma_min(self):
        return self.std/10.0
    @property
    def sigma_max(self):
        return self.std*10.0

    @property
    def nu_min(self):
        return 0.1
    @property
    def nu_max(self):
        return 10.0

    def _mean_variance(self):
        T = self.T
        
        mu = np.trapz(self.y, self.t) / T
        var = np.trapz(np.square(self.y - mu), self.t) / T

        self._mu = mu
        self._var = var
        self._std = np.sqrt(var)

        self._wn_var = np.trapz(np.square(self.dy), self.t) / T

    def _roots_to_quad(self, r):
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

        return par.stable_polynomial_roots(p['ar_roots_p'], self.root_min, self.root_max)

    def ma_roots(self, p):
        p = self.to_params(p)

        return par.stable_polynomial_roots(p['ma_roots_p'], self.root_min, self.root_max)

    def ar_poly(self, p):
        return np.real(np.poly(self.ar_roots(p)))

    def ma_poly(self, p):
        poly = np.real(np.poly(self.ma_roots(p)))

        return poly / poly[-1]

    def to_params(self, p):
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def deparameterize(self, p):
        p = self.to_params(p)

        mu = par.bounded_values(p['logit_mu'], low=self.mu_min, high=self.mu_max)
        sigma = par.bounded_values(p['logit_sigma'], low=self.sigma_min, high=self.sigma_max)
        nu = par.bounded_values(p['logit_nu'], low=self.nu_min, high=self.nu_max)

        ar_roots = self.ar_roots(p)

        arr = []
        for i in range(0, ar_roots.shape[0]-1, 2):
            ar1 = ar_roots[i]
            ar2 = ar_roots[i+1]
            if np.imag(ar1) > 0:
                arr.append(np.real(ar1))
                arr.append(np.abs(np.imag(ar1)))
            else:
                arr.append(np.real(ar1))
                arr.append(np.real(ar2))
        if ar_roots.shape[0] % 2 == 1:
            arr.append(np.real(ar_roots[-1]))
        arr = np.array(arr)

        if self.q > 0:
            ma_roots = self.ma_roots(p)
            
            mar = []
            for i in range(0, ma_roots.shape[0]-1, 2):
                ma1 = ma_roots[i]
                ma2 = ma_roots[i+1]
                if np.imag(ma1) > 0:
                    mar.append(np.real(ma1))
                    mar.append(np.abs(np.imag(ma1)))
                else:
                    mar.append(np.real(ma1))
                    mar.append(np.real(ma2))
            if ma_roots.shape[0] % 2 == 1:
                mar.append(np.real(ma_roots[-1]))
            mar = np.array(mar)

            return np.concatenate((mu, sigma, nu, arr, mar))
        else:
            return np.concatenate((mu, sigma, nu, arr))   

    def to_carmapack_params(self, p):
        p = self.to_params(p)

        mu = par.bounded_values(p['logit_mu'], low=self.mu_min, high=self.mu_max)
        sigma = par.bounded_values(p['logit_sigma'], low=self.sigma_min, high=self.sigma_max)
        nu = par.bounded_values(p['logit_nu'], low=self.nu_min, high=self.nu_max)

        pc = np.zeros(self.nparams)

        pc[0] = sigma
        pc[1] = nu*nu # carma_pack wants variance scaling
        pc[2] = mu

        assert self.p >= 1, 'Must have at least one AR root'
        pc[3:3+self.p] = self._roots_to_quad(par.stable_polynomial_roots(p['ar_roots_p'], self.root_min, self.root_max))

        if self.q >= 1:
            pc[3+self.p:] = self._roots_to_quad(par.stable_polynomial_roots(p['ma_roots_p'], self.root_min, self.root_max))

        return pc

    def carmapack_to_params(self, pc):
        p = self.to_params(np.zeros(self.nparams))

        mu = pc[2]
        nu = np.sqrt(pc[1])
        sigma = pc[0]
        
        p['logit_mu'] = par.bounded_params(mu, low=self.mu_min, high=self.mu_max)
        p['logit_sigma'] = par.bounded_params(sigma, low=self.sigma_min, high=self.sigma_max)
        p['logit_nu'] = par.bounded_params(nu, low=self.nu_min, high=self.nu_max)

        p['ar_roots_p'] = par.stable_polynomial_params(self._quad_to_roots(pc[3:3+self.p]))

        if self.q >= 1:
            p['ma_roots_p'] = par.stable_polynomial_params(self._quad_to_roots(pc[3+self.p:]))

        return p

    def draw_roots(self, n):
        rs = []

        for i in range(0, n-1, 2):
            if np.random.rand() < 0.5:
                # Complex
                re = np.random.uniform(low=-self.root_max, high=-self.root_min)
                im = np.random.uniform(low=0.0, high=self.root_max)

                rs.append(re + im*1j)
                rs.append(re - im*1j)
            else:
                # Real
                xy = np.random.uniform(low=-self.root_max, high=-self.root_min, size=2)

                rs.append(xy[0])
                rs.append(xy[1])

        if n % 2 == 1:
            x = np.random.uniform(low=-self.root_max, high=-self.root_min)
            rs.append(x)

        return np.array(rs, dtype=np.complex)

    def draw_prior(self):
        mu = np.random.uniform(low=self.mu_min, high=self.mu_max)
        sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
        nu = np.exp(np.random.uniform(low=np.log(self.nu_min),
                                      high=np.log(self.nu_max)))
        ar_quad = self._roots_to_quad(self.draw_roots(self.p))
        ma_quad = self._roots_to_quad(self.draw_roots(self.q))

        mup = par.bounded_params(mu, low=self.mu_min, high=self.mu_max)
        sigmap = par.bounded_params(sigma, low=self.sigma_min, high=self.sigma_max)
        nup = par.bounded_params(nu, low=self.nu_min, high=self.nu_max)

        return np.concatenate((mup, sigmap, nup, ar_quad, ma_quad))

    def log_prior(self, p):
        p = self.to_params(p)

        lp = 0.0

        # mu
        lp += par.bounded_log_jacobian(p['logit_mu'], low=self.mu_min, high=self.mu_max)

        # sigma
        lp += par.bounded_log_jacobian(p['logit_sigma'], low=self.sigma_min, high=self.sigma_max)
        sigma = par.bounded_values(p['logit_sigma'], low=self.sigma_min, high=self.sigma_max)
        lp -= np.log(sigma)

        # nu
        lp += par.bounded_log_jacobian(p['logit_nu'], low=self.nu_min, high=self.nu_max)
        nu = par.bounded_values(p['logit_nu'], low=self.nu_min, high=self.nu_max)
        lp -= np.log(nu)

        # ar roots
        lp += par.stable_polynomial_log_jacobian(p['ar_roots_p'], self.root_min, self.root_max)
        roots = par.stable_polynomial_roots(p['ar_roots_p'], self.root_min, self.root_max)

        lp -= np.sum(np.log(np.abs(roots)))

        if self.q >= 1:
            lp += par.stable_polynomial_log_jacobian(p['ma_roots_p'], self.root_min, self.root_max)
            roots = par.stable_polynomial_roots(p['ma_roots_p'], self.root_min, self.root_max)

            lp -= np.sum(np.log(np.abs(roots)))

        if np.isnan(lp):
            lp = -np.inf
            
        return lp

    def log_likelihood(self, p):
        pc = self.to_carmapack_params(p)

        pcv = cm.vecD()
        pcv.extend(pc)

        ll = self._carma_process.getLogDensity(pcv)

        if np.isnan(ll):
            ll = np.NINF
        
        return ll

    def __call__(self, p):
        lp = self.log_prior(p)

        if lp == np.NINF:
            return np.NINF
        else:
            return float(lp + self.log_likelihood(p))

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

        self._carma_process = cm.run_mcmc_carma(1, 1, self._tv, self._yv, self._dyv, self.p, self.q, 1, False, 1)
        self._carma_process.SetMLE(True)

    def power_spectrum(self, fs, p):
        p = self.to_params(p)

        ar_roots = self.ar_roots(p)
        ma_roots = self.ma_roots(p)

        ar_coefs = np.real(np.poly(ar_roots))

        ma_coefs = np.real(np.poly(ma_roots))
        ma_coefs /= ma_coefs[-1]
        ma_coefs = ma_coefs[::-1]

        s = par.bounded_values(p['logit_sigma'], low=self.sigma_min, high=self.sigma_max)

        sigma = s / np.sqrt(cm.carma_variance(1.0, ar_roots, ma_coefs))

        return cm.power_spectrum(fs, sigma, ar_coefs, ma_coefs)

    def white_noise(self, p, bw):
        p = self.to_params(p)

        nu = par.bounded_values(p['logit_nu'], low=self.nu_min, high=self.nu_max)
        
        return self.wn_var * nu*nu / bw

    def standardised_residuals(self, p):
        p = self.to_params(p)

        kfilter = self._make_kalman_filter(p)

        kmean = np.asarray(kfilter.GetMean())
        kvar = np.asarray(kfilter.GetVar())

        mu = par.bounded_values(p['logit_mu'], low=self.mu_min, high=self.mu_max)
        
        return (self.y - mu - kmean) / np.sqrt(kvar)

    def residuals(self, p):
        p = self.to_params(p)

        kfilter = self._make_kalman_filter(p)

        kmean = np.asarray(kfilter.GetMean())
        kvar = np.asarray(kfilter.GetVar())

        mu = par.bounded_values(p['logit_mu'], low=self.mu_min, high=self.mu_max)

        return (self.y - mu - kmean), kvar

    def predict(self, p, ts):
        p = self.to_params(p)

        kfilter = self._make_kalman_filter(p)

        ypred = []
        ypred_var = []
        for t in ts:
            yp = kfilter.Predict(t)
            ypred.append(yp.first)
            ypred_var.append(yp.second)

        mu = par.bounded_values(p['logit_mu'], low=self.mu_min, high=self.mu_max)
            
        return np.array(ypred) + mu, np.array(ypred_var)

    def simulate(self, p, ts):
        p = self.to_params(p)

        kfilter = self._make_kalman_filter(p)

        vtime = cm.vecD()
        vtime.extend(ts)

        ysim = np.asarray(kfilter.Simulate(vtime))

        mu = par.bounded_values(p['logit_mu'], low=self.mu_min, high=self.mu_max)

        return ysim + mu

    def _make_kalman_filter(self, p):
        p = self.to_params(p)

        ar_roots = self.ar_roots(p)
        ma_coefs = self.ma_poly(p)

        mu = par.bounded_values(p['logit_mu'], low=self.mu_min, high=self.mu_max)
        nu = par.bounded_values(p['logit_nu'], low=self.nu_min, high=self.nu_max)

        s = par.bounded_values(p['logit_sigma'], low=self.sigma_min, high=self.sigma_max)
        
        sigma = s / np.sqrt(cm.carma_variance(1.0, ar_roots, ma_coefs))
        sigmasq = sigma*sigma
        sigmasq = float(sigmasq)
        
        tv = cm.vecD()
        tv.extend(self.t)
        yv = cm.vecD()
        yv.extend(self.y - mu)
        dyv = cm.vecD()
        dyv.extend(self.dy*nu)
        arv = cm.vecC()
        arv.extend(ar_roots)
        mav = cm.vecD()
        mav.extend(ma_coefs)
        kfilter = cm.KalmanFilterp(tv, yv, dyv, sigmasq, arv, mav)

        kfilter.Filter()

        return kfilter
