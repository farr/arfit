import numpy as np
import scipy.interpolate as si
import scipy.signal as ss

def spectrum(ts, data, **kwargs):
    """Produces an estimate of the PSD using Welch's method (see
    :func:`ss.welch` for details).  The time spacing is chosen to be
    the minimum time spacing in ``ts``, and the data are linearly
    interpolated to this uniform spacing.

    """

    data = data - np.mean(data)
    dts = np.diff(ts)
    dt = np.min(dts[dts>0])

    f = si.interp1d(ts, data)
    its = np.arange(ts[0], ts[-1], dt)
    idata = f(its)

    idata = idata - np.mean(idata)

    return ss.welch(idata, fs=1.0/dt, **kwargs)

def acf(ts, data):
    """Returns an estimate of the normalised autocorrelation function for
    the given data.

    """
    data = data - np.mean(data)
    dts = np.diff(ts)
    dt = np.min(dts[dts>0])

    f = si.interp1d(ts, data)
    its = np.arange(ts[0], ts[-1], dt)
    idata = f(its)

    idata = idata - np.mean(idata)

    nextpowtwo = 1
    while nextpowtwo < idata.shape[0]:
        nextpowtwo = nextpowtwo << 1
    nextpowtwo << 1

    paddedidata = np.zeros(nextpowtwo)
    paddedidata[:idata.shape[0]] = idata

    paddedacf = np.fft.ifft(np.square(np.abs(np.fft.fft(paddedidata))))
    return its-its[0], paddedacf[:idata.shape[0]]/paddedacf[0]
