cimport cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
     double exp(double x)

@cython.boundscheck(False)
def predict(np.ndarray[np.float_t, ndim=1] t,
            np.ndarray[np.float_t, ndim=1] y,
            np.ndarray[np.float_t, ndim=1] vy,
            np.ndarray[np.float_t, ndim=1] yp,
            np.ndarray[np.float_t, ndim=1] vyp,
            double sigma,
            double tau):
    cdef unsigned int i
    cdef double kp
    cdef double vkp
    cdef double dt
    cdef double alpha
    cdef double sigma2
    cdef double gain

    cdef unsigned int n

    n = t.shape[0]

    assert y.shape[0] == n, 'bad y shape'
    assert yp.shape[0] == n, 'bad yp shape'
    assert vyp.shape[0] == n, 'bad vyp shape'

    sigma2 = sigma*sigma

    kp = 0.0
    vkp = sigma2

    for i in range(0, n-1):
        # Accumulate the prediction
        yp[i] = kp
        vyp[i] = vkp + vy[i]

        # Update the internal state
        gain = vkp/vyp[i]
        kp = kp + (y[i] - yp[i])*gain
        vkp = vkp - vyp[i]*gain*gain

        # Advance to the next timestep
        dt = t[i+1] - t[i]
        alpha = exp(-dt/tau)
        kp = kp*alpha
        vkp = alpha*alpha*vkp + (1 - alpha*alpha)*sigma2

    # Predict final timestep
    yp[n-1] = kp
    vyp[n-1] = vkp + vy[n-1]
