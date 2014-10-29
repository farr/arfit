import arfit.arn_posterior as pos
import numpy as np

roots = np.array([-1.0/(3600.0*24.0)+2.0*np.pi*1j/(5.0*3600), -1.0/(3600.0*24.0)-2.0*np.pi*1j/(5.0*3600), -1.0/(10.0*3600)])
sigma = 1e-8

def draw_observations(Tmax):
    # Observe once an hour up to Tmax
    ts = np.arange(0, Tmax, 3600.0)

    # Scatter the observations +/- 15 minutes
    ts = ts + np.random.uniform(low=-15.0*60, high=15.0*60, size=ts.shape[0])

    # Observe only during the night
    ts = ts[(np.fmod(ts, 24.0*3600) < 6.0*3600) | (np.fmod(ts, 24.0*3600) > 18.0*3600)]

    # 1/3 of the time it's cloudy
    ts = ts[np.random.uniform(size=ts.shape[0]) < 2.0/3.0]

    ys = pos.generate_data(sigma, roots, ts)

    return ts, ys
