from pylab import *
import carma_pack_posterior as cpp
import pspec as ps
import run_carma_pack_posterior as rcp
from run_carma_pack_posterior import LL, LP
import bz2
import carmcmc as cm
import emcee 
import os
import os.path as op
import pickle
import plotutils.autocorr as ac
import plotutils.plotutils as pu
import plotutils.runner as pr
import scipy.stats as ss
import triangle as tri

def plot_psd(runner, xlabel=None, ylabel=None, Npts=1000, Nmcmc=1000):
    logpost = runner.sampler.logp.lp
    fs = exp(linspace(log(1.0/(2.0*logpost.T)), log(2.0/logpost.dt_min), Npts))
    ls = ps.normalised_lombscargle(logpost.t, logpost.y, fs)
    psds = []
    wns = []
    for p in permutation(runner.burnedin_chain[0,...].reshape((-1, logpost.nparams)))[:Nmcmc, :]:
        psds.append(logpost.power_spectrum(fs, p))
        wns.append(logpost.white_noise(p, fs[-1] - fs[0]))
    psds = array(psds)
    wns = array(wns)
    ar_psds = psds
    psds = wns.reshape((-1, 1)) + psds

    loglog(fs, ls, '-k')
    
    loglog(fs, median(psds, axis=0), '-b')
    fill_between(fs, percentile(psds, 84, axis=0), percentile(psds, 16, axis=0), color='b', alpha=0.25)
    fill_between(fs, percentile(psds, 97.5, axis=0), percentile(psds, 2.5, axis=0), color='b', alpha=0.25)
    
    loglog(fs, median(ar_psds, axis=0), '-r')
    fill_between(fs, percentile(ar_psds, 84, axis=0), percentile(ar_psds, 16, axis=0), color='r', alpha=0.25)
    fill_between(fs, percentile(ar_psds, 97.5, axis=0), percentile(ar_psds, 2.5, axis=0), color='r', alpha=0.25)
    
    axhline(median(wns), color='g')
    fill_between(fs, percentile(wns, 84) + 0*fs, percentile(wns, 16) + 0*fs, color='g', alpha=0.25)
    fill_between(fs, percentile(wns, 97.5) + 0*fs, percentile(wns, 2.5) + 0*fs, color='g', alpha=0.25)
    
    axis(ymin=percentile(wns, 2.5)/1000.0)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

def plot_residuals(runner, Nmcmc=1000, Npts=1000):
    logpost = runner.sampler.logp.lp
    resid = []
    for p in permutation(runner.burnedin_chain[0,...].reshape((-1, logpost.nparams)))[:Nmcmc, :]:
        resid.append(logpost.standardised_residuals(p))
    resid = array(resid)
    errorbar(logpost.t, mean(resid, axis=0), yerr=std(resid, axis=0), color='k', fmt='.')

def plot_resid_distribution(runner, N=10, Npts=1000):
    logpost = runner.sampler.logp.lp
    for p in permutation(runner.burnedin_chain[0,...].reshape((-1, logpost.nparams)))[:N, :]:
        r = logpost.standardised_residuals(p)
        pu.plot_kde_posterior(r, color='b', alpha=0.1)
    xs = linspace(-5, 5, Npts)
    plot(xs, ss.norm.pdf(xs), '-k')

def process_output_dir(dir, runner=None, return_runner=False):
    if runner is None:
        with bz2.BZ2File(op.join(dir, 'runner.pkl.bz2'), 'r') as inp:
            runner = pickle.load(inp)

    runner.sampler.logp.lp._mean_variance()

    figure()
    loglog(1/runner.sampler.beta_history.T)
    axvline(runner.sampler.chain.shape[2]*0.2)
    savefig(op.join(dir, 'temp-history.pdf'))
    
    figure()
    pu.plot_emcee_chains_one_fig(runner.burnedin_chain[0,...])
    savefig(op.join(dir, 'chains.pdf'))

    figure()
    plot_psd(runner)
    savefig(op.join(dir, 'psd.pdf'))

    figure()
    plot_residuals(runner)
    savefig(op.join(dir, 'resid.pdf'))

    figure()
    plot_resid_distribution(runner)
    savefig(op.join(dir, 'resid-distr.pdf'))

    if return_runner:
        return runner
