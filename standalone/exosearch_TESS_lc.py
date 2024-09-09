import lightkurve as lk
import numpy as np
import jax
import jax.numpy as jnp
from util import *
import scipy

def transit_num(y_d, num_period):
    ''''
    Arg:
    y_d : y.Cov_inv * transit_profile_d the size of this is ~ N_full / delta

    Returns:
    num_det: returns numerator of likelihoods as a 2D array indexed by period/delta and epoch/delta
    '''
    num_det = jnp.zeros((num_period, num_period)) 
    for p in range(num_period):
        for t0 in range(p+1):
            num_det = num_det.at[p, t0].set(jnp.sum(y_d[t0 :: p+1]))
    return num_det

def transit_den(K_d, num_period):
    '''
    Arg:
    K_d : Cov_inv * (transit_profile_d.transit_profile_d^T) the size of this is ~ (N_full / delta, N_full / delta)

    Returns:
    den_det: returns denominator of likelihoods as a 2D array indexed by period/delta and epoch/delta
    '''
    den_det = jnp.zeros((num_period, num_period))
    for p in range(num_period):
        for t0 in range(p+1): 
            den_det = den_det.at[p, t0].set(jnp.sum(K_d[t0 :: p+1, t0 :: p+1]))
    return den_det


# Download and clean light curve data
# ====================================

tic_id = 'TIC 382435735'  # Replace with your target TIC ID
search_result = lk.search_lightcurve(tic_id, mission='TESS', author='SPOC', cadence='short')
lightcurve = search_result.download_all()

# For now just look at a single sector (to do multi-sector need to save numerator/denom of detector tests separately and recombine)
sector = 27
ind_sector = np.argwhere(lightcurve.sector == sector)[0][0]

lc_sap = lightcurve[ind_sector].SAP_FLUX
quality = lc_sap.quality
time = lc_sap.time
cam = lc_sap.camera
ccd = lc_sap.ccd
len_lc = len(lc_sap)
cadence_to_min = 2
day_to_cadence = 720

# Mask bad quality points and normalize the light curve
lc_cadence = lc_sap.cadenceno[~quality.astype(bool)]
lc_flux = lc_sap.flux.to_value()[~quality.astype(bool)]
median_flux = np.nanmedian(lc_flux)
lc_normalized = median_normal(lc_flux)

# ====================================

lc_detrend, cov_inv = covariance_model(lc_normalized, lc_cadence, sector, cam, ccd, model_order = 8, full=True)
#pickle.dump((lc_detrend, cov_inv), open('cov_%s_%s.p' % (tic_id, sector), 'wb'))
#(lc_detrend, cov_inv) = pickle.load(open('cov_%s_%s.p' % (tic_id, sector), 'rb'))

# Defining transit parameter search space (period, epoch, duration)
# Period ranges from 0 to N/2
# epoch ranges from 0 to P
delta = 50 # period and epoch step size
durations = jnp.array([1, 2, 3, 4, 6, 8, 10, 12, 14, 16])*30 
N_full = len(lc_detrend)
lc_cov_inv = cov_inv.dot(lc_detrend) 
# Compute transit likelihoods over parameter space

num_period = int((N_full - durations[-1]) // (2 * delta))# number of periods to search in stepsize of delta
transit_likelihood_stats = np.zeros((len(durations), num_period, num_period))

jit_lik_num = jax.jit(transit_num, static_argnums=(1))
jit_lik_den = jax.jit(transit_den, static_argnums=(1))

for i in range(len(durations)):
    print (i)
    if i != 5: continue # only compute the LRT for one of the trial transit durations
    d = durations[i]
    transit_profile = jnp.ones(d)
    transit_kernel = jnp.outer(transit_profile, transit_profile)

    y_d = jax.scipy.signal.convolve(lc_cov_inv, transit_profile)[int(d/2)-1:N_full-int(d/2)-1][::delta]

    # commented this out as I get a memory error
    #K_d = jax.scipy.signal.convolve2d(cov_inv, transit_kernel)[int(d/2)-1:N_full-int(d/2)-1,int(d/2)-1:N_full-int(d/2)-1][::delta,::delta]

    # different way to calculate K_d
    K_d = np.zeros((np.shape(y_d)[0], np.shape(y_d)[0]))
    for l in range(num_period):
        for m in range(num_period):
            K_d[l,m] = np.sum(transit_kernel*cov_inv[(l*delta):(l*delta) + d, (m*delta):(m*delta) + d])    
    K_d = jnp.array(K_d)

    likelihoods_num = transit_num(y_d, num_period)
    likelihoods_den = transit_den(K_d, num_period)

    # Output transit detection tests, indexed as [P/delta, t_0/delta]
    transit_likelihood_stats[i] = np.divide(likelihoods_num, np.sqrt(likelihoods_den), out=np.zeros_like(likelihoods_num), where=likelihoods_den!=0.)
    
top_detections = nd_argsort(transit_likelihood_stats)

for i in range(5):
    print (top_detections[i], transit_likelihood_stats[top_detections[i][0], top_detections[i][1], top_detections[i][2]])
    print ('LRT (SNR): ', np.round(transit_likelihood_stats[top_detections[i][0], top_detections[i][1], top_detections[i][2]],2), 'duration (hr): ', np.round(top_detections[i][0]/30, 2), 'period (day): ', np.round(delta*(top_detections[i][1]+1)/(day_to_cadence),2), 'epoch(day): ',  np.round((delta/day_to_cadence)*top_detections[i][2], 2) )
