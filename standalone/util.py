import numpy as np
from scipy.linalg import toeplitz
from lightkurve.correctors import load_tess_cbvs
import matplotlib.pyplot as plt
import pickle

sector_years = [[1, 13], [14, 26], [27, 39], [40, 43]] #13 sectors per year

def check_symmetric(a, rtol=1e-05):
    return (np.sum(a-a.T) < rtol)

def median_normal(lightcurve):
    """
    Median normalize lightcurve
    
    Args:
    lightcurve : lightcurve to be normalized.
        
    Returns:
    Median-normalized lightcurve.
    """
    lightcurve -= np.nanmedian(lightcurve)
    return lightcurve / np.nanmedian(np.abs(lightcurve))

def mag_normal_med(lightcurve):
    """
    Normalize lightcurve by magnitude (to be used if calculating pairwise correlation with corr_comp)
    
    Args:
    lightcurve :lightcurve to be normalized.
        
    Returns:
    magnitude-normalized data.
    """
    lightcurve -= np.nanmedian(lightcurve) #Subtract median as slightly more robust to outliers
    return lightcurve / np.linalg.norm(lightcurve)

def mag_normal_mean(lightcurve):
    """
    Normalize lightcurve by magnitude (to be used if calculating pairwise correlation with corr_comp)
    
    Args:
    lightcurve :lightcurve to be normalized.
        
    Returns:
    magnitude-normalized data.
    """
    lightcurve -= np.nanmean(lightcurve)
    return lightcurve / np.linalg.norm(lightcurve)

def calc_CDPP(lightcurve, scale, offset):
    """
    CDPP in PPM
    """
    lightcurve *= scale
    smooth_reg = lightcurve - savgol_filter(lightcurve, 97, 2)
    smooth_reg = threshold_data(smooth_reg)
    mean_bin = np.zeros(len(smooth_reg)-14)
    for j in range(len(mean_bin)):
        mean_bin[j] = np.mean(smooth_reg[j:j+13])
    cdpp_reg = ( np.std(mean_bin)*1.168 / np.median(offset) ) / (1e-6) # CDPP in PPM
    return cdpp_reg

def linear_detrend(lightcurve):
    """
    Linearly detrend an individual lightcurve

    Args:
    lightcurve : Length N time series
    
    Returns:
    lightcurve with linear trend removed
    
    """
    z = np.polyfit(range(len(lightcurve)), lightcurve, 1)
    p = np.poly1d(z)
    return lightcurve - p(range(len(lightcurve)))

def threshold_data(data, base_data = None, level=4):
    """
    Threshold outliers (flux samples) at level*std dev and replace with Gaussian random samples.
    
    The filtering is performed in a two-step procedure by first applying a coarse threshold
    to remove extremal values and using this data to calculate the std dev. 
    
    Args:
    data : 1D array containing the data to be thresholded.
    base_data (optional) : base the calculation of points to be thresholded on base_data if supplied (thresholding applied to data)
    level (optional) : Factor by which the standard deviation is multiplied to set the threshold level. Default is 5.
        
    Returns:
    1D array containing the thresholded data.
        
    """
    if base_data is None: base_data = data
    std_ = np.nanstd(base_data)
    diff = np.diff(base_data, prepend=base_data[0])
    thresh = level*std_
    mask = np.ones(len(base_data), dtype=bool)

    mask[np.abs(base_data) > thresh] = False
    mask[np.abs(diff) > thresh] = False

    std_clean = np.nanstd(base_data[mask])
    thresh = level*std_clean

    mask = np.zeros(len(data), dtype=bool)    
    mask[np.abs(base_data) > thresh] = True
    mask[np.abs(diff) > thresh] = True

    data[mask] = np.random.normal(0, std_clean, size=mask.sum())
    return data

def nan_linear_gapfill(data):
    """
    Fill NaN gaps in data using linear interpolation.
    
    The function identifies groups of consecutive NaNs in the data and fills them using 
    a linear interpolation approach based on the values immediately adjacent to the gaps.
    
    Args:
    data : 1D array containing the data with NaN gaps to be filled.
        
    Returns:
    1D array where NaN gaps have been filled using linear interpolation.
        
    """
    goodind = np.where(~np.isnan(data))
    badind = np.where(np.isnan(data))
    gaps = [list(group) for group in mit.consecutive_groups(badind[0])]
    for g in gaps:
        if len(g) == 1:
            data[g[0]] = data[g[0]-1]
            continue
        else:
            grad = (data[g[len(g)-1]+1]-data[g[0]-1])/(len(g)+2)
            data[g] = (np.arange(len(g))*grad) + data[g[0]-1]
    return data


def cbv_matrix(lc_cadence, sector, cam, ccd, model_order = None):
    '''
    Load and mask the TESS cotrending basis vectors (otherwise fit own basis)
    '''
    cbvs = load_tess_cbvs(sector=sector, camera=cam, ccd=ccd, cbv_type='SingleScale')
    if model_order == None: model_order = 16
    cbv_dm = cbvs.to_designmatrix(cbv_indices=np.arange(1, model_order+1))
    V = cbv_dm.values
    V_masked = V[np.in1d(cbvs.cadenceno, lc_cadence)]
    return V_masked.T

def covariance_stellar(lc, lc_cadence):
    '''
    Estimate stellar covariance model (stationary toeplitz model), using spectral estimator on detrended light curve
    '''
    lc_thresh = threshold_data(lc, base_data = None, level=3)
    lc_cadence_zero = lc_cadence - lc_cadence[0]
    cadence_len = lc_cadence_zero[-1]
    structured_lc = np.zeros(cadence_len*2 - 1)
    structured_lc[:cadence_len] = np.random.normal(0, np.std(lc), cadence_len)  # gap filling
    structured_lc[lc_cadence_zero] = lc_thresh

    p_noise = smooth_p(structured_lc, K=3)
    ac = np.real(np.fft.ifft(p_noise)).astype('float32')
    ac = ac[:cadence_len+1]
    cov_stellar = toeplitz(ac, r = ac)
    masked_cov_stellar = cov_stellar[lc_cadence_zero]
    masked_cov_stellar = masked_cov_stellar[:, lc_cadence_zero]
    return masked_cov_stellar

def covariance_model(lc, lc_cadence, sector, cam, ccd, model_order = None, full=True):
    '''
    H1: y = z + t + n, H0: y = z + n
    z ~ N(V*mu_c, V cov_c V.T + cov_*)
    Estimate joint covariance of z: Cov_stellar (stationary) + Cov_systematics (low rank)

    args
    model_order : systematics model order
    full: return the covariance and light curve, with uniform time sampling and zero's at masked/missing samples
    '''
    V = cbv_matrix(lc_cadence, sector, cam, ccd, model_order = model_order)
    ls_fit = np.dot(V, lc.T).dot(V) 
    lc_detrend = lc - ls_fit

    # load the systematic noise covariance (estimated from collection of light curves on the same sensor)
    #cov_c = pickle.load( open("cov_c_diag%s_%s_%s.p" % (sector, cam, ccd), "rb" ))
    cov_c = pickle.load( open("cov_c%s_%s_%s.p" % (sector, cam, ccd), "rb" ))

    cov_s = covariance_stellar(np.copy(lc_detrend), lc_cadence)

    cov_z = np.dot(V.T, np.dot(cov_c, V)) + cov_s
    #cov_inv_z = jax.numpy.linalg.pinv(cov_z)
    
    cov_inv_z = np.linalg.inv(cov_z)

    print ('symmetry check', check_symmetric(cov_inv_z))
    print ('nan check',  np.sum(np.isnan(cov_inv_z)))
    print (np.sum(cov_inv_z))
    if full:
        lc_cadence_zero = lc_cadence - lc_cadence[0]
        cadence_len = lc_cadence_zero[-1]
        lc_detrend_full = np.zeros(cadence_len+1)
        lc_detrend_full[lc_cadence_zero] = lc_detrend
        cov_inv_z_full = np.zeros((cadence_len+1, cadence_len+1))
        cov_inv_z_full[np.ix_(lc_cadence_zero, lc_cadence_zero)] = cov_inv_z
        print (np.sum(cov_inv_z_full))
        return lc_detrend_full, cov_inv_z_full
    else:
        return lc_detrend, cov_inv_z

def smooth_p(noise, K=3):
    '''
    Computes the smoothed periodogram
    '''
    N = len(noise)
    p_noise = (1/float(N)) * (np.abs(np.fft.fft(noise))**2)
    integ_periodogram = np.zeros(N)
    for i in range(N):
        if i<K: integ_periodogram[i] = np.sum(p_noise[:i+K])
        elif i>N-K: integ_periodogram[i] = np.sum(p_noise[i-K:])
        else: integ_periodogram[i] = np.sum(p_noise[i-K:i+K])
    integ_periodogram *= (1/float(2*K))
    return integ_periodogram

def box_transit(times_, period, dur, t0, alpha=1):
    """
    Generate a transit signal time-series with box function evaluated at given times.
    
    Args:
    times_ :  Array of time points at which to evaluate the transit time-series
    period : Period of the transit
    dur : Duration of the transit.
    t0 : Epoch.
    alpha : Transit depth. Default is 1.
        
    Returns:
    Transit time series evaluated at `times_`.
    """

    return np.piecewise(times_, [((times_-t0+(dur/2))%period) > dur, ((times_-t0+(dur/2))%period) <= dur], [0, 1])*(-alpha)

def nd_argsort(x):
    return np.array(np.unravel_index(np.argsort(x, axis=None), x.shape)).T[::-1]

