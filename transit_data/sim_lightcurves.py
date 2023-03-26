import transit
import sys
import pandas as pd
import numpy as np
import pickle
from astropy.io import fits
from transit_util import *
import matplotlib.pyplot as plt

year = 3
N = 1000
#y1 and y3 are south
#y2 and y4 are north
# Generate injected lightcurves for 3 sectors worth of data

_dir = '/media/ielo/Extreme SSD/projects/TESS_y%s/' % (year)
u_dir = '/media/ielo/Extreme SSD/scratch/tess_code/y%s/' % (year)

sector_years = [[1, 13], [14, 26], [27, 39]]
pers_lc = [1828, 2046, 1922]

# first generate the lightcurves, keep them broken down by sector 

def create_sim_lc(lc_index):
    print (lc_index)
    pds = np.random.uniform(low=.5, high=20., size=1)
    t0_ = np.random.uniform(0., pds, size=1)
    rp_rs = np.random.uniform(low=0.01, high=0.2, size=1)
    w = np.random.uniform(low=-np.pi, high=np.pi, size=1)
    impact_param = np.random.uniform(0., 1., size=1)
    q = np.random.uniform(low=0., high=1., size=2)
    s = transit.System(transit.Central(q1=q[0], q2=q[1]))
    pickle.dump((pds, t0_, rp_rs, w, impact_param, q), open("transit_sim/transit_params_%s.p" % (lc_index), "wb" ) )
    body = transit.Body(radius=rp_rs, period=pds, t0 = t0_, b = impact_param, omega = w)
    s.add_body(body)

    for sector in range(27, 30):
        filenames = pickle.load( open(_dir+"filenames%s.p" % (sector), "rb" ))
        loc = pickle.load( open("other/arg_pers%s.p" % (sector), "rb" ))
        filename = filenames[loc[lc_index]]
        with fits.open(_dir+'sector'+str(sector)+'/'+filename, memmap=True) as hdulist:
            t_id = hdulist[0].header['TICID']
            t_mag = hdulist[0].header['TESSMAG']
            sap_fluxes = hdulist[1].data['SAP_FLUX']
            time = hdulist[1].data['CADENCENO']
            time_bjd = hdulist[1].data['TIME']
            quality = hdulist[1].data['QUALITY']
            cam = hdulist[0].header['CAMERA']
            ccd = hdulist[0].header['CCD']
            cbvfilename = cbv_names[sector-1] % (cam, ccd)
            spike_evecs = []
            evecs = []
            with fits.open(_dir+'cbv_sector'+str(sector)+'/'+cbvfilename, memmap=True) as hdulist2:
                for j in range(30):
                    k = j+1
                    try:
                        ev = hdulist2[1].data['VECTOR_%s' % k]
                        if np.any(ev): evecs.append(ev)
                    except:
                        continue
                    try:
                        spike_evecs.append(hdulist2[2].data['VECTOR_%s' % k])
                    except:
                        continue
            spike_evecs = np.array(spike_evecs, dtype='float32')
            evecs = np.array(evecs, dtype='float32')
        cam_ccd = np.array([cam, ccd])
        # Calculate bad quality data points for masking
        ind_bad = np.logical_or((quality).astype(bool), (np.isnan(sap_fluxes)).astype(bool))
        mom_dump = (np.bitwise_and(quality, 2**5) >= 1)
        for mt in np.where(mom_dump)[0]:
            if mt <= 30: continue
            if mt >= len(quality)-30: continue
            ind_bad[mt-30:mt] = np.ones(30)
        ind_good = ~ind_bad
        good_data = np.where(ind_good)[0]

#==============================================
        # Inject simulated transit
        newx = time_bjd.byteswap().newbyteorder() # force native byteorder
        time_bjd = pd.Series(newx)
        transit_sim = s.light_curve(time_bjd[good_data])
        if np.isnan(transit_sim).any(): print ('NAN found')

        transit_lc = np.copy(sap_fluxes)
        transit_lc[good_data] *= transit_sim

#======================
        # Clean up the data
        lc = median_normal(transit_lc)

        evecs_mask = [ma.masked_where(ind_bad, ev, copy=True) for ev in evecs]
        sample_lc_mask = ma.masked_where(ind_bad, lc, copy=True)
        coeff = ma.dot(evecs_mask, sample_lc_mask.T)
        lsfit = np.dot(coeff, evecs)
        detrend = lc-lsfit

        s_evecs_mask = [ma.masked_where(ind_bad, ev, copy=True) for ev in spike_evecs]
        sample_lc_mask = ma.masked_where(ind_bad, detrend, copy=True)
        coeff = ma.dot(s_evecs_mask, sample_lc_mask.T)
        lsfit = np.dot(coeff, spike_evecs)
        detrend -= lsfit
        detrend[good_data] = threshold_positive(detrend[good_data])
        pickle.dump((t_id, t_mag, detrend, time, time_bjd, ind_bad, transit_sim, cam_ccd), open('y%s_lightcurves/sector%s/lc_%s.p' % (year, sector, lc_index), 'wb'))

for i in range(N):
    try:
        create_sim_lc(i)
    except: print ('alerrrt')

#durs = np.array([1, 2, 3, 4, 6, 8, 10, 12, 14, 16])
#durs *= 30
