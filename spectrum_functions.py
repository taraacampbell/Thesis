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
import numpy.polynomial.polynomial as poly

import functions as fn

DEIMOS_DATA = '/Users/taracampbell/thesis/'

def RunOneTelluric_FullOLD(telluric, index, swaves, sflux, sivar, choose_polyfit, kernel_means):
    wrange = np.arange(-0.25,0.25,0.025)/0.01
    
    twaves, tflux = fn.get_tel(telluric)
    
    los_pix = kernel_means[index] / 0.01

    normal_sci = fn.fit_poly(sflux, sflux, swaves, choose_polyfit)
    smooth_tflux = scipynd.gaussian_filter1d(tflux,los_pix)

    chi_wshift = []
    for wshift in wrange:

        # Shift wavelength range
        shift_wave = twaves + wshift*0.01

        # Interpolate
        interp_shift_tel = np.interp(swaves, shift_wave, smooth_tflux)

        # Fit continuum in range        
        telluric_new = fn.fit_poly(normal_sci, interp_shift_tel, swaves, 1)  

        # Calculate chi2
        chi2 = fn.calc_chi2(normal_sci,swaves,0.1,telluric_new)
        chi_wshift = np.append(chi_wshift,chi2)
    
    return chi_wshift

def RunOneTelluric_RangeOLD(telluric, index, swaves, sflux, sivar, choose_range, choose_polyfit, kernel_means):
    wrange = np.arange(-0.25,0.25,0.025)/0.01
    
    twaves, tflux = fn.get_tel(telluric)
    
    los_pix = kernel_means[index] / 0.01
    
    use_twaves, use_swaves, use_tflux, use_sflux, use_sivar = fn.SelectRange(twaves, swaves, tflux, sflux, sivar,\
                                                                             choose_range)
    normal_sci = fn.fit_poly(use_sflux, use_sflux, use_swaves, choose_polyfit)
    smooth_tflux = scipynd.gaussian_filter1d(use_tflux,los_pix)

    chi_wshift = []
    for wshift in wrange:

        # Shift wavelength range
        shift_wave = use_twaves + wshift*0.01

        # Interpolate
        interp_shift_tel = np.interp(use_swaves, shift_wave, smooth_tflux)

        # Fit continuum in range        
        telluric_new = fn.fit_poly(normal_sci, interp_shift_tel, use_swaves, 1)  

        # Calculate chi2
        chi2 = fn.calc_chi2(normal_sci,use_swaves,0.1,telluric_new)
        chi_wshift = np.append(chi_wshift,chi2)
    
    return chi_wshift


## FORWARD FITTING MODEL

def RunOneTelluric_Full(telluric, index, swaves, sflux, kernel_means):
    wrange = np.arange(-0.25,0.25,0.025)/0.01
    
    twaves, tflux = fn.get_tel(telluric)
    
    # Convolve telluric
    los_pix = kernel_means[index] / 0.01
    smooth_tflux = scipynd.gaussian_filter1d(tflux,los_pix)

    chi_wshift = []
    for wshift in wrange:

        # Shift wavelength range
        shift_wave = twaves + wshift*0.01

        # Interpolate
        interp_shift_tel = np.interp(swaves, shift_wave, smooth_tflux)
        
        # Fit telluric to data
        coefs = poly.polyfit(swaves, sflux, 3)
        ffit = poly.polyval(swaves, coefs)
        telluric_new = ffit*interp_shift_tel

        # Calculate chi2
        chi2 = fn.calc_chi2(sflux,swaves,0.1,telluric_new)
        chi_wshift = np.append(chi_wshift,chi2)
    
    return chi_wshift


def RunOneTelluric_Range(telluric, index, swaves, sflux, sivar, choose_range, kernel_means):
    wrange = np.arange(-0.25,0.25,0.025)/0.01

    twaves, tflux = fn.get_tel(telluric)
    
    los_pix = kernel_means[index] / 0.01
    
    use_twaves, use_swaves, use_tflux, use_sflux, use_sivar = fn.SelectRange(twaves, swaves, tflux, sflux, sivar,\
                                                                             choose_range)
    
    smooth_tflux = scipynd.gaussian_filter1d(use_tflux,los_pix)
    
    chi_wshift = []
    for wshift in wrange:

        # Shift wavelength range
        shift_wave = use_twaves + wshift*0.01

        # Interpolate
        interp_shift_tel = np.interp(use_swaves, shift_wave, smooth_tflux)

        # Fit telluric to data
        coefs = poly.polyfit(use_swaves, use_sflux, 3)
        ffit = poly.polyval(use_swaves, coefs)
        telluric_new = ffit*interp_shift_tel


        # Calculate chi2
        
        chi2 = fn.calc_chi2(use_sflux,use_swaves,0.1,telluric_new)
        chi_wshift = np.append(chi_wshift,chi2)
    
    return chi_wshift