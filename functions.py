import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob

from astropy.table import Table
from astropy import units as u
import astropy.constants
from astropy.io import ascii,fits

from scipy.optimize import curve_fit
from astropy import  convolution

import scipy.ndimage as scipynd
import scipy.stats as stats
from scipy.stats import chisquare

DEIMOS_DATA = '/Users/taracampbell/thesis/'


def read_sci_files(file_folder):
    folder = []
    for name in file_folder:
        df = pd.read_csv(name, sep='\s+')
        df['S/N'] = df['FLUX'] * np.sqrt(df['IVAR'])
        mean_SN = np.mean(df['S/N'])
        ivar = df['IVAR']
        df['VAR'] = np.sqrt(1/(ivar))
        mean_SN = np.mean(df['S/N'])
        folder.append(df)
    return folder

def make_highSN_folder(file_folder):
    highSN_folder = []
    for name in file_folder:
        df = pd.read_csv(name, sep='\s+')
        ivar = df['IVAR']
        df['S/N'] = df['FLUX'] * np.sqrt(df['IVAR'])
        mean_SN = np.mean(df['S/N'])
        if mean_SN > 10:
            highSN_folder.append(name)
    return highSN_folder

def get_tel(telluric):
    df = pd.read_csv(telluric, skiprows=2, sep='\s+', names=['WAVE','FLUX'])
    waves = df['WAVE']
    flux = df['FLUX']
    
    return waves, flux

def read_tel_files(file_folder):
    tel_spectra = []

    for name in file_folder:
        df = pd.read_csv(name, skiprows=2, sep='\s+', names=['WAVE','FLUX'])
        tel_spectra.append(df)
        
    return tel_spectra
    
def gaussian(x,*p) :
    # A gaussian peak with:
    #   Constant Background          : p[0]
    #   Peak height above background : p[1]
    #   Central value                : p[2]
    #   Standard deviation           : p[3]
    return p[0]+p[1]*np.exp(-1.*(x-p[2])**2/(2.*p[3]**2))

def gauss_guess(x,y):
    norm = np.median(np.percentile(y,50))
    ind = y.index(max(y))
    w = x[ind]
    N_guess = np.max(y) - np.min(y)
    sig_guess = 0.5
    p0 = [norm,N_guess,w,sig_guess]

    return p0

def fit_poly(fit_data, apply_fit_data, xrange, order):
    fit_data = np.array(fit_data)
    xrange = np.array(xrange)
    z = np.polyfit(xrange,fit_data,order)
    p = np.poly1d(z)
    nsci = apply_fit_data/p(xrange)
    return nsci

def chi_interp(chi,o2,h2o):
   
    mn    = np.argmin(chi)
    n1  = tmp_h2o == tmp_h2o[mn]
    n2  = tmp_o2  == tmp_o2[mn]
   
    # H2O
    c = chi[n1]
    o = o2[n1]
    h = h2o[n1]
    argchi = np.argsort(c)
    arg = argchi[0:3]
    p_o2 = np.polyfit(o[arg],c[arg],2)
    r_o2 = np.roots(p_o2)
   
    # O2
    c = chi[n2]
    o = o2[n2]
    h = h2o[n2]
    argchi = np.argsort(c)
    arg = argchi[0:3]
    p_h2o = np.polyfit(h[arg],c[arg],2)
    r_h2o = np.roots(p_h2o)
   
    best_h2o = r_h2o.real[0]
    best_o2  = r_o2.real[0]
    best_chi = chi[mn]
   
    return best_o2,best_h2o,best_chi

def calc_chi2(data_flux,data_wave,data_ivar,final_fit,no_aband = 0):

    if no_aband == 0:
        # SET DATA_IVAR TO 0.1 ?
        red_chi2 = np.sum((data_flux - final_fit)**2 * data_ivar)/np.size(data_flux)
        chi2 = np.sum((data_flux - final_fit)**2)/(final_fit)

    if no_aband == 1:
        m =~ (data_wave > 7500) & (data_wave < 7700)
        chi2 = np.sum((data_flux[m] - final_fit[m])**2 * data_ivar[m])/np.size(data_flux[m])

    return red_chi2

def parse_tfile(tfile):
    
    spl = tfile.split('_')
    h2o = np.float(spl[3])
    o2  = np.float(spl[5])
    return o2,h2o

def SelectRange(tel_waves, sci_waves, tel_flux, sci_flux, sci_ivar, choose_range):
    
    data = [[tel_waves, tel_flux], [sci_waves, sci_flux, sci_ivar]]
    new = []

    # Now have different masks for science & telluric spectra
    
    # TELLURIC
    telluric_data = data[0]
    twaves = telluric_data[0]
    tflux = telluric_data[1]
    
    trange = np.ma.masked_inside(twaves, choose_range[0], choose_range[1])
    tmsk = trange.mask
    
    trange_data = [twaves[tmsk], tflux[tmsk]]
    
    # SCIENCE
    science_data = data[1]
    swaves = science_data[0]
    sflux = science_data[1]
    sivar = science_data[2]
    
    srange = np.ma.masked_inside(swaves, choose_range[0], choose_range[1])
    smsk = srange.mask
    
    srange_data = [swaves[smsk], sflux[smsk], sivar[smsk]]
    
    sciwvNEW = srange_data[0]
    sciflxNEW = srange_data[1]
    sciivarNEW = srange_data[2]
    
    telwvNEW = trange_data[0]
    telflxNEW = trange_data[1]
    
    return telwvNEW, sciwvNEW, telflxNEW, sciflxNEW, sciivarNEW

def filter_error(one_sci):
        
        wv = one_sci['WAVE']
        flx = one_sci['FLUX']
        ivar = one_sci['IVAR']
        err = list(one_sci['VAR'])

        # Filter out points with large error

        new_waves = []
        new_fluxes = []
        new_ivar = []
        new_errs = []

        for e in err:
            if e < 55:
                #had at 30 before
                ind = err.index(e)
                new_waves.append(wv[ind])
                new_fluxes.append(flx[ind])
                new_ivar.append(ivar[ind])
                new_errs.append(e)

        new_waves = np.array(new_waves)
        new_fluxes = np.array(new_fluxes)
        new_ivar = np.array(new_ivar)

        return new_waves, new_fluxes, new_ivar