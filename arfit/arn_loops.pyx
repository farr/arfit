import numpy as np
cimport numpy as np

def beta_matrix_loop(unsigned int n, unsigned int pp, \
                     np.ndarray[np.float_t, ndim=2] alpha, \
                     np.ndarray[np.float_t, ndim=2] cov, \
                     np.ndarray[np.float_t, ndim=2] beta):
    cdef int i
    cdef int j
    cdef int k
    cdef int l

    assert alpha.shape[0] == pp + 1, 'bad alpha shape[0]'
    assert alpha.shape[1] == n, 'bad alpha shape[1]'

    assert cov.shape[0] == 2*pp + 1, 'bad cov shape[0]'
    assert cov.shape[1] == n, 'bad cov shape[1]'

    assert beta.shape[0] == pp, 'bad beta shape[0]'
    assert beta.shape[1] == n, 'bad beta shape[1]'

    assert n > 2*pp+1, 'n <= 2*pp + 1'

    for i in range(n):
        for j in range(i, min(i+pp, n)):
            for k in range(max(i-pp, 0), i+1):
                for l in range(max(j-pp, 0), j+1):
                    if l >= k:
                        beta[pp-(j-i)-1, j] += alpha[i-k, k]*cov[2*pp-(l-k), l]*alpha[j-l, l]
                    else:
                        beta[pp-(j-i)-1, j] += alpha[i-k, k]*cov[2*pp-(k-l), k]*alpha[j-l, l]

def log_likelihood_xs_loop(unsigned int n, unsigned int p, \
                           np.ndarray[np.float_t, ndim=2] alpha, \
                           np.ndarray[np.float_t, ndim=1] ys, \
                           np.ndarray[np.float_t, ndim=1] xs):
    cdef int j
    cdef int k

    assert alpha.shape[0] == p + 1, 'bad alpha shape[0]'
    assert alpha.shape[1] == n, 'bad alpha shape[1]'

    assert ys.shape[0] == n, 'bad ys shape[0]'
    
    assert xs.shape[0] == n, 'bad xs shape[0]'

    assert n > p, 'n <= p'
    
    for j in range(n - p):
        for k in range(j, j+p+1):
            xs[k] += ys[j]*alpha[k-j, j]

    for j in range(n-p, n):
        for k in range(j, n):
            xs[k] += ys[j]*alpha[k-j, j]
