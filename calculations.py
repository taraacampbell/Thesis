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

import functions as fn

DEIMOS_DATA = '/Users/taracampbell/thesis/'


def line_centers(files, wavelengths):
    
    bad_slits_center = []
    mean_centers = []

    for f in files:

        line_centers = []
        line_center_err = []
        kernel = []
        kernel_err = []
        bad_slits = []

        cent_use_waves = []
        kern_use_waves = []

        for w in wavelengths:

            one_file = pd.read_csv(f, sep='\s+')

            fname = str(f)
            slit_name = fname[-17:]

            ivar = one_file['IVAR']
            one_file['VAR'] = np.sqrt(1/(ivar))

            sline = w
            plus = sline + 6
            minus = sline - 6

            msk_10A = one_file['WAVE'].between(minus, plus, inclusive=True)
            gauss_range = one_file[msk_10A]
            gauss_range['WAVE'] -= sline

            if gauss_range.empty:
                pass

            else:
                try:
                    waves = list(gauss_range['WAVE'])
                    sky = list(gauss_range['SKY'])
                    var = list(gauss_range['VAR'])

                    std = gauss_range['SKY'].std()
                    mean = gauss_range['SKY'].mean()
                    mx = np.max(sky)

                    guesses = gauss_guess(waves, sky)

                    popt, pcov = curve_fit(gaussian, waves, sky, guesses, sigma=var)

                except (RuntimeError, TypeError):
                    bad_slits_center.append(slit_name)

                param_err = np.sqrt(np.diag(pcov))

                center = popt[2]
                center_err = param_err[2]

                if center_err < 1:
                    line_centers.append(center)
                    line_center_err.append(center_err)
                    cent_use_waves.append(w)

        mean_centers.append(np.mean(line_centers))

        plt.errorbar(cent_use_waves,line_centers, yerr=line_center_err, fmt='.')

        fit = np.polyfit(cent_use_waves,line_centers,1)
        fit_fn = np.poly1d(fit) 
        plt.plot(cent_use_waves, fit_fn(cent_use_waves), '--r')

        plt.title(slit_name)
        plt.show()


def kernel_widths(files, wavelengths):
    bad_slits_width = []
    slit_kernel_means = []

    for f in files:

        kernel = []
        kernel_err = []

        kern_use_waves = []

        for w in wavelengths:

            one_file = pd.read_csv(f, sep='\s+')

            fname = str(f)
            slit_name = fname[-17:]

            ivar = one_file['IVAR']
            one_file['VAR'] = np.sqrt(1/(ivar))

            sline = w
            plus = sline + 6
            minus = sline - 6

            msk_10A = one_file['WAVE'].between(minus, plus, inclusive=True)
            gauss_range = one_file[msk_10A]
            gauss_range['WAVE'] -= sline

            if gauss_range.empty:
                pass

            else:
                try:
                    waves = list(gauss_range['WAVE'])
                    sky = list(gauss_range['SKY'])
                    var = list(gauss_range['VAR'])

                    std = gauss_range['SKY'].std()
                    mean = gauss_range['SKY'].mean()
                    mx = np.max(sky)

                    guesses = fn.gauss_guess(waves, sky)

                    popt, pcov = curve_fit(fn.gaussian, waves, sky, guesses, sigma=var)

                except (RuntimeError, TypeError):
                    bad_slits_width.append(slit_name)

                param_err = np.sqrt(np.diag(pcov))

                width = popt[3]
                width_err = param_err[3]

                if width_err < 1:
                    kernel.append(width)
                    kernel_err.append(width_err)
                    kern_use_waves.append(w)

        slit_kernel_means.append(np.mean(kernel))    
    return slit_kernel_means