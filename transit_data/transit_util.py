import numpy as np
import pickle
import numpy.ma as ma
from scipy import signal
from scipy.stats import multivariate_normal
from scipy.linalg import toeplitz
from astropy.io import fits
import sys
import os


cbv_names = ['tess2018206045859-s0001-%s-%s-0120-s_cbv.fits','tess2018234235059-s0002-%s-%s-0121-s_cbv.fits','tess2018263035959-s0003-%s-%s-0123-s_cbv.fits','tess2018292075959-s0004-%s-%s-0124-s_cbv.fits'\
,'tess2018319095959-s0005-%s-%s-0125-s_cbv.fits','tess2018349182459-s0006-%s-%s-0126-s_cbv.fits','tess2019006130736-s0007-%s-%s-0131-s_cbv.fits','tess2019032160000-s0008-%s-%s-0136-s_cbv.fits'\
,'tess2019058134432-s0009-%s-%s-0139-s_cbv.fits','tess2019085135100-s0010-%s-%s-0140-s_cbv.fits','tess2019112060037-s0011-%s-%s-0143-s_cbv.fits','tess2019140104343-s0012-%s-%s-0144-s_cbv.fits', \
'tess2019169103026-s0013-%s-%s-0146-s_cbv.fits',\
'tess2019198215352-s0014-%s-%s-0150-s_cbv.fits', 'tess2019226182529-s0015-%s-%s-0151-s_cbv.fits', 'tess2019253231442-s0016-%s-%s-0152-s_cbv.fits','tess2019279210107-s0017-%s-%s-0161-s_cbv.fits',\
'tess2019306063752-s0018-%s-%s-0162-s_cbv.fits','tess2019331140908-s0019-%s-%s-0164-s_cbv.fits','tess2019357164649-s0020-%s-%s-0165-s_cbv.fits','tess2020020091053-s0021-%s-%s-0167-s_cbv.fits',\
'tess2020049080258-s0022-%s-%s-0174-s_cbv.fits','tess2020078014623-s0023-%s-%s-0177-s_cbv.fits','tess2020106103520-s0024-%s-%s-0180-s_cbv.fits','tess2020133194932-s0025-%s-%s-0182-s_cbv.fits',\
'tess2020160202036-s0026-%s-%s-0188-s_cbv.fits',\
'tess2020186164531-s0027-%s-%s-0189-s_cbv.fits','tess2020212050318-s0028-%s-%s-0190-s_cbv.fits','tess2020238165205-s0029-%s-%s-0193-s_cbv.fits','tess2020266004630-s0030-%s-%s-0195-s_cbv.fits',\
'tess2020294194027-s0031-%s-%s-0198-s_cbv.fits','tess2020324010417-s0032-%s-%s-0200-s_cbv.fits','tess2020351194500-s0033-%s-%s-0203-s_cbv.fits','tess2021014023720-s0034-%s-%s-0204-s_cbv.fits',\
'tess2021039152502-s0035-%s-%s-0205-s_cbv.fits','tess2021065132309-s0036-%s-%s-0207-s_cbv.fits','tess2021091135823-s0037-%s-%s-0208-s_cbv.fits','tess2021118034608-s0038-%s-%s-0209-s_cbv.fits',\
'tess2021146024351-s0039-%s-%s-0210-s_cbv.fits',\
'tess2021175071901-s0040-%s-%s-0211-s_cbv.fits', 'tess2021204101404-s0041-%s-%s-0212-s_cbv.fits', 'tess2021232031932-s0042-%s-%s-0213-s_cbv.fits', 'tess2021258175143-s0043-%s-%s-0214-s_cbv.fits']

#===========================

def check_symmetric(a, rtol=1e-05):
    return (np.sum(a-a.T) < rtol)

def median_normal(data):
    a = (data-np.nanmedian(data))
    return a/np.nanmedian(np.abs(a)) 

def safe_div(n, d):
    return n / d if d else 0

#============================
# simulate ar1 noise
def noise_ar1(size, sigma=.001, a1=.6):
    wn = np.random.normal(0, sigma, size=n)
    noise = np.zeros(n)
    noise[0] = wn[0]
    for i in range(n-1):
        noise[i+1] = (a1 * noise[i]) + wn[i+1]         
    return noise

def ar1_cov(n, sigma=.001, a1=.6):
    autocorr = [a1**i for i in range(n)]
    autocorr = np.array(autocorr)*(1/1-(a1**2))*(sigma**2)
    return scipy.linalg.toeplitz(autocorr)

def ar1_power(n, sigma=.001, a1=.6):
    comp = [a1*np.exp(-1j * 2 * np.pi *(k/(float(n)))) for k in range(n)]
    power = [(sigma**2)*(np.abs(1-comp[i])**(-2)) for i in range(n)]
    return power

#=============================
# computes sample autocovariance matrix (in practise similar to the ifft of smoothed periodogram)
def sample_acf_missing(noise, ind):
    N = len(noise)
    acf = np.zeros(N)
    for i in range(N):
        no_samples = 0
        for j in range(N-i):
            if (j+i in ind) and (j in ind):
                no_samples +=1
                acf[i] += noise[j+i]*noise[j]
        if no_samples ==0: print (i)
        acf[i]*=(1/no_samples)
    return acf

#=============================

def smooth_p(noise, K=3):
    N = len(noise)
    p_noise = (1/float(N)) * (np.abs(np.fft.fft(noise))**2)
    integ_periodogram = np.zeros(N)
    for i in range(N):
        if i<K: integ_periodogram[i] = np.sum(p_noise[:i+K])
        elif i>N-K: integ_periodogram[i] = np.sum(p_noise[i-K:])
        else: integ_periodogram[i] = np.sum(p_noise[i-K:i+K])
    integ_periodogram *= (1/float(2*K))
    return integ_periodogram

#=============================

def threshold_data(data):
    std_ = np.nanstd(data)
    diff = np.ediff1d(data)
    thresh = 4*std_
    mask = np.ones(len(data), dtype=bool)
    for j in range(len(data)-1):
        if np.abs(diff[j]) > thresh: mask[j+1] = 0
        if np.abs(data[j]) > thresh: mask[j] = 0
    std_ = np.nanstd(data[mask])
    diff = np.ediff1d(data)
    thresh = 4*std_
    for j in range(len(data)-1):
        if np.abs(diff[j]) > thresh: data[j+1] = np.random.normal(0, std_)
        if np.abs(data[j]) > thresh: data[j] = np.random.normal(0, std_)
    return data

def threshold_positive(data):
    std_ = np.nanstd(data)
    diff = np.ediff1d(data)
    thresh = 3*std_
    mask = np.ones(len(data), dtype=bool)
    for j in range(len(data)-1):
        if np.abs(diff[j]) > thresh: mask[j+1] = 0
        if np.abs(data[j]) > thresh: mask[j] = 0
    std_ = np.nanstd(data[mask])
    thresh =  3*std_
    for j in range(len(data)):
        if data[j] > thresh: data[j] = np.random.normal(0, std_)
    return data
