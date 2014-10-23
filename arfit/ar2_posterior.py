import numpy as np

def gcov(eta_plus, eta_minus, ti, tj, tk, tl):
    r"""Computes 

    .. math::

      \int_{t_k}^{t_l} d\xi G\left( t_i ; \xi \right) G\left( t_j ; \xi \right)

    where :math:`G(t; \xi)` is the Green's function for the
    second-order ODE with homogeneous solutions 

    .. math::

      y(t) = A e^{\eta_+ t} + B e^{\eta_- t}

    """

    factor = 1.0/(2.0*np.square(eta_plus - eta_minus))

    em_factor = 1.0/eta_minus*(np.exp(eta_minus*(ti + tj - 2.0*tk)) - \
                               np.exp(eta_minus*(ti + tj - 2.0*tl)))

    ep_factor = 1.0/eta_plus*(np.exp(eta_plus*(ti + tj - 2.0*tk)) - \
                              np.exp(eta_plus*(ti + tj - 2.0*tl)))

    epem_factor = 2.0/(eta_plus + eta_minus) * \
                  (np.exp(-tl*(eta_minus + eta_plus)) * \
                   (np.exp(tj*eta_minus + ti*eta_plus) + \
                    np.exp(ti*eta_minus + tj*eta_plus)) - \
                   np.exp(-tk*(eta_plus + eta_minus)) * \
                   (np.exp(tj*eta_minus + ti*eta_plus) + \
                    np.exp(ti*eta_minus + tj*eta_plus)))

    return factor*(em_factor + ep_factor + epem_factor)

            
