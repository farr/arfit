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

def normalised_lombscargle(ts, ys, dys, oversampling=5, nyquist_factor=3):
    model = LombScargleFast().fit(ts, ys, dys)

    pers, pows = model.periodogram_auto(oversampling=oversampling, nyquist_factor=nyquist_factor)
    fs = 1.0/pers

    T = np.max(ts) - np.min(ts)

    mu = 1/T*np.trapz(ys, ts)
    s2 = 1/T*np.trapz(np.square(ys-mu), ts)

    return fs, s2*pows/np.trapz(pows, fs)


def plot_psd_sample_data(sample, oversampling=5, nyquist_factor=3):
    psd_low, psd_high, psd_med, fs = sample.plot_power_spectrum(doShow=False)

    plt.clf()

    plt.loglog(fs, psd_med, '-b', alpha=0.33)
    plt.fill_between(fs, psd_low, psd_high, color='b', alpha=0.17)

    fs, psd = normalised_lombscargle(sample.time, sample.y, sample.ysig, oversampling=oversampling, nyquist_factor=nyquist_factor)

    bw = fs[-1] - fs[0]
    T = sample.time[-1] - sample.time[0]

    s2 = 1/T*np.trapz(np.square(sample.ysig), sample.time)
    noise_level = s2/bw
    levels = noise_level*np.sqrt(sample.get_samples('measerr_scale'))
    plt.axhline(np.median(levels), color='g', alpha=0.33)
    plt.fill_between(fs, np.percentile(levels, 84)+0*fs, np.percentile(levels, 16)+0*fs, color='g', alpha=0.17)

    plt.loglog(fs, psd, '-r', alpha=0.33)

def plot_psd_sample_draw(sample, loc='upper left', oversampling=5, nyquist_factor=3):
    fs, psd = normalised_lombscargle(sample.time, sample.y, sample.ysig, oversampling=oversampling, nyquist_factor=nyquist_factor)

    ys_draw = sample.predict(sample.time, bestfit='random')[0]

    fs, dpsd = normalised_lombscargle(sample.time, ys_draw, sample.ysig, oversampling=oversampling, nyquist_factor=nyquist_factor)

    plt.loglog(fs, psd, '-k', label='Data', alpha=0.5)
    plt.loglog(fs, dpsd, '-b', label='Prediction', alpha=0.5)
    plt.legend(loc=loc)
