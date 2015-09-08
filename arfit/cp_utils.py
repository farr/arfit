import carmcmc as cm
from gatspy.periodic import LombScargleFast
import matplotlib.pyplot as plt
import numpy as np

def csample_from_files(datafile, chainfile, p, q):
    data = np.loadtxt(datafile)

    times, tind = np.unique(data[:,0], return_index=True)
    data = data[tind, :]
    
    chain = np.loadtxt(chainfile)
    assert chain.shape[1] == p + q + 5, 'dimension mismatch'

    return cm.CarmaSample(data[:,0], data[:,1], data[:,2], None, q=q, trace=chain[:,:-2], loglike=chain[:,-2], logpost=chain[:,-1])

def normalised_lombscargle(ts, ys, dys):
    model = LombScargleFast().fit(ts, ys, dys)

    pers, pows = model.periodogram_auto()
    fs = 1.0/pers

    T = np.max(ts) - np.min(ts)
    mu = 1.0/T*np.trapz(ys, ts)
    s2 = 1.0/T*np.trapz(np.square(ys-mu), ts)

    return fs, s2*pows/np.trapz(pows, fs)


def plot_psd_sample_data(sample):
    sample.plot_power_spectrum(doShow=False)

    noise_level = 2.0*np.mean(np.diff(sample.time))*np.mean(np.square(sample.ysig))

    plt.axhline(noise_level, color='g')

    fs, psd = normalised_lombscargle(sample.time, sample.y, sample.ysig)

    plt.loglog(fs, psd, '-k')

def plot_psd_sample_draw(sample, loc='upper left'):
    fs, psd = normalised_lombscargle(sample.time, sample.y, sample.ysig)

    ys_draw = sample.predict(sample.time, bestfit='random')[0]

    fs, dpsd = normalised_lombscargle(sample.time, ys_draw, sample.ysig)

    plt.loglog(fs, psd, '-k', label='Data')
    plt.loglog(fs, dpsd, '-b', label='Prediction')
    plt.legend(loc=loc)
