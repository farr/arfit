from pylab import *
import pspec as ps
import bz2
import emcee 
import os
import os.path as op
import pickle
import plotutils.autocorr as ac
import plotutils.plotutils as pu
import plotutils.runner as pr
import scipy.stats as ss
import triangle as tri

def plot_psd(logpost, chain, xlabel=None, ylabel=None, Npts=1000, Nmcmc=1000, oversampling=1):
    
    ls_fs, ls = ps.normalised_lombscargle(logpost.t, logpost.y, logpost.dy, oversampling=oversampling)

    fs = linspace(np.min(ls_fs), np.max(ls_fs), Npts)
    
    psds = []
    wns = []
    for p in permutation(chain)[:Nmcmc,:]:
        psds.append(logpost.power_spectrum(fs, p))
        wns.append(logpost.white_noise(p, np.max(fs) - np.min(fs)))
    psds = array(psds)
    wns = array(wns)
    ar_psds = psds
    psds = wns.reshape((-1, 1)) + psds
    
    loglog(ls_fs, ls, '-k')
    
    loglog(fs, median(psds, axis=0), '-b')
    fill_between(fs, percentile(psds, 84, axis=0), percentile(psds, 16, axis=0), color='b', alpha=0.25)
    fill_between(fs, percentile(psds, 97.5, axis=0), percentile(psds, 2.5, axis=0), color='b', alpha=0.25)
    
    loglog(fs, median(ar_psds, axis=0), '-r')
    fill_between(fs, percentile(ar_psds, 84, axis=0), percentile(ar_psds, 16, axis=0), color='r', alpha=0.25)
    fill_between(fs, percentile(ar_psds, 97.5, axis=0), percentile(ar_psds, 2.5, axis=0), color='r', alpha=0.25)
    try:
        plot(fs, 0*fs + median(wns), color='g')
        fill_between(fs, percentile(wns, 84) + 0*fs, percentile(wns, 16) + 0*fs, color='g', alpha=0.25)
        fill_between(fs, percentile(wns, 97.5) + 0*fs, percentile(wns, 2.5) + 0*fs, color='g', alpha=0.25)
    except:
        pass
    
    axis(ymin=np.min(ls)/1000.0)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

def plot_residuals(logpost, chain, Nmcmc=1000, Npts=1000):
    resid = []
    for p in permutation(chain)[:Nmcmc, :]:
        resid.append(logpost.standardised_residuals(p))
    resid = array(resid)
    errorbar(logpost.t, mean(resid, axis=0), yerr=std(resid, axis=0), color='k', fmt='.')

def plot_resid_distribution(logpost, chain, N=10):
    for p in permutation(chain)[:N, :]:
        r = logpost.standardised_residuals(p)
        pu.plot_kde_posterior(r, color='b', alpha=0.1, label='Resid')

    for i in range(N):
        pu.plot_kde_posterior(randn(r.shape[0]), color='k', alpha=0.1, label='N(0,1)')
        
def plot_resid_acf(logpost, chain, N=10):
    for p in permutation(chain)[:N,:]:
        r = logpost.standardised_residuals(p)
        acorr(r, maxlags=None, alpha=0.1)

    axhline(1.96/sqrt(r.shape[0]), color='b')
    axhline(-1.96/sqrt(r.shape[0]), color='b')

def plot_evidence_integrand(runner, fburnin=0.5):
    istart = int(round(fburnin*runner.chain.shape[2]))

    lnlikes = runner.lnlikelihood[:, :, istart:]
    mean_lnlikes = np.mean(lnlikes, axis=(1,2))

    mean_lnlikes = mean_lnlikes[:-1] # strip off beta=0
    betas = runner.sampler.betas[:-1] # strip off beta=0
    
    plot(betas, betas*mean_lnlikes)
    xlabel(r'$\beta$')
    ylabel(r'$\beta \left\langle \ln \mathcal{L} \right\rangle_\beta$')
    xscale('log')

def dic(runner):
    istart = int(round(0.5*runner.chain.shape[2]))
    lnlikes = runner.lnlikelihood[0,:,istart:]

    return -2.0*(np.mean(lnlikes) - np.var(lnlikes))

def residual_pvalues(logpost, chain, Nps=10):
    stats = []
    for p in permutation(chain)[:Nps,:]:
        r = logpost.standardised_residuals(p)
        stat, cv, sl = ss.anderson(r, 'norm')
        stats.append(stat)
    stats = np.array(stats)

    return stats, cv, sl

def plot_prediction(logpost, chain, Npred=100, Nts=1000):
    ts = linspace(np.min(logpost.t), np.max(logpost.t), Nts)
    ts = np.sort(np.concatenate((ts, logpost.t)))

    preds = []
    uls = []
    lls = []
    for p in permutation(chain)[:Npred,:]:
        pre = logpost.predict(p, ts)
        preds.append(pre[0])
        uls.append(pre[0] + sqrt(pre[1]))
        lls.append(pre[0] - sqrt(pre[1]))
    preds = np.array(preds)
    uls = np.array(uls)
    lls = np.array(lls)

    plot(ts, median(preds, axis=0), '-b')
    fill_between(ts, median(uls, axis=0), median(lls, axis=0), color='b', alpha=0.5)
        
    errorbar(logpost.t, logpost.y, logpost.dy, color='k', fmt='.')

def plot_simulation_psd(logpost, chain):
    fs, psd = ps.normalised_lombscargle(logpost.t, logpost.y, logpost.dy)

    loglog(fs, psd, '-k', alpha=0.5)
    
    for p in permutation(chain)[:10,:]:
        sim = logpost.simulate(p, logpost.t)
        fs, psd = ps.normalised_lombscargle(logpost.t, sim, logpost.dy)

        loglog(fs, psd, '-b', alpha=0.05)
