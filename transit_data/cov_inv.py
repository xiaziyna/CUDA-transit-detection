import numpy as np
import pickle
from scipy.stats import multivariate_normal
from scipy.linalg import toeplitz
from astropy.io import fits
import sys
import os
from transit_util import *

year = 3
N = 1

def generate_cov_inv(sector, lc_index):
    (t_id, t_mag, detrend, time, time_bjd, ind_bad, transit_sim, cam_ccd) = pickle.load( open("y%s_lightcurves/sector%s/lc_%s.p" % (year, sector, lc_index), "rb" ))
    cam = int(cam_ccd[0])
    ccd = int(cam_ccd[1])

    cbvfilename = cbv_names[sector-1] % (cam, ccd)
    evecs = []

    with fits.open('other/cbv/cbv_sector'+str(sector)+'/'+cbvfilename, memmap=True) as hdulist2:
        for j in range(30):
            k = j+1
            try:
                ev = hdulist2[1].data['VECTOR_%s' % k]
                if np.any(ev): evecs.append(ev)
            except:
                continue
    evecs = np.array(evecs, dtype='float32')

    #======================================
    # load data

    ind_good = ~ind_bad
    good_data = np.where(ind_good)[0]
    detrend[good_data] = threshold_data(detrend[good_data])
    good_detrend = detrend[good_data]
    detrend[np.where(ind_bad)] = np.random.normal(0, np.std(good_detrend), np.sum(ind_bad)) #good_detrend[:np.sum(ind_bad)]

    #=============================
    #gaussian integrated test

    # H1: y = z + t + n, H0: y = z + n
    # z - N(V*mu_c, V cov_c V.T + cov_*)
    # Am using the LSfit as an estimate of V*mu_c, therefore do test on clean signal (detrend)

    cov_c = pickle.load( open("other/priors/%s/cov_c_diag%s_%s_%s.p" % (sector, sector, cam, ccd), "rb" )) #cov_c has off diagonal entries

    _lc_fill = detrend # - np.mean(detrend) #shouldn't be doing this step, however needed for spectral estimation? detrend should already be zero mean (the filling is making it non-zero mean)
    len_lc = len(detrend)
    zp_lc = np.zeros((2*len_lc) - 1)
    zp_lc[:len_lc] = _lc_fill
    p_noise = smooth_p(zp_lc, K=3)
    ac = np.real(np.fft.ifft(p_noise)).astype('float32')
    ac = ac[:len_lc]#*2

    block1 = np.dot(evecs.T, np.dot(cov_c, evecs)) + toeplitz(ac, r = ac)
    masked_cov_z1 = block1[good_data]
    masked_cov_z1 = masked_cov_z1[:, good_data]
    block1 = None

    cov_inv_z = np.linalg.inv(masked_cov_z1)
    masked_cov_z1 = None

    print ('symmetry check', check_symmetric(cov_inv_z))
    print ('nan check',  np.sum(np.isnan(cov_inv_z)))

    filename_cov = 'y%s_covariance/sector%s/c_inv_%s.dat' % (year, sector, lc_index)
    arr = np.memmap(filename_cov, dtype='float32',mode='w+',shape=(np.shape(cov_inv_z)))
    arr[:] = cov_inv_z[:]
    arr.flush()

for i in range(N):
    for sector in range(27, 30):
        print (i, sector)
        generate_cov_inv(sector, i)


