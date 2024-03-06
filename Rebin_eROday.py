import matplotlib.pyplot as plt
import numpy as np
import scipy.stats, scipy.special, scipy.optimize
from astropy.table import Table
import astropy.io.fits as fits
import os
import sys
import matplotlib
from matplotlib import ticker
import warnings
import time
import argparse
import multiprocessing as mp
from functools import partial
from astropy.io import ascii
warnings.filterwarnings("ignore")

rebin_time_threshold=100 #very conservative time [s] used as a limit for consecutive bins belonging to the same eroday

def estimate_source_cr_marginalised(log_src_crs_grid, src_counts, bkg_counts, bkg_area, rate_conversion, flag_src="SRC"):
    """ Compute the Poisson PDF in a grid of log(source count rate) [log_src_crs_grid], given
    the observed src_counts, and bkg_counts in the background region of size bkg_area. rate_conversion contains fracexp and timedel [time bin]
    By David Bogensberger and Johannes Buchner
    Adapted by Riccardo Arcodia: the flag_src asks for source only rates, source+bkg rates, or background only rates
    """
    u = np.linspace(0, 1, len(log_src_crs_grid))[1:-1]
    if flag_src=="SRC" :
        def prob(log_src_cr):
            src_cr = 10 ** log_src_cr * rate_conversion
            bkg_cr = scipy.special.gammaincinv(bkg_counts + 1, u) / bkg_area
            like = scipy.stats.poisson.pmf(src_counts, src_cr + bkg_cr).mean()
            return like
    elif flag_src=="SRC+BKG" :
        def prob(log_src_cr):
            src_cr = 10 ** log_src_cr * rate_conversion
            bkg_cr = 0
            like = scipy.stats.poisson.pmf(src_counts, src_cr + bkg_cr).mean()
            return like
    elif flag_src=="BKG" :
        def prob(log_src_cr):
            src_cr = 10 ** log_src_cr * rate_conversion * bkg_area
            bkg_cr = 0
            like = scipy.stats.poisson.pmf(src_counts, src_cr + bkg_cr).mean()
            return like
    weights = np.array([prob(log_src_cr) for log_src_cr in log_src_crs_grid])
    weights /= weights.sum()
    return weights

def rate_errors(cts, rate_conv, Bcts, Bratio, flag_src="SRC"):
    """ Compute errors of count rates. Some comments below.
    By David Bogensberger and Johannes Buchner, adapted by Riccardo Arcodia
    """
    n_gp=200 #grid points of count rates for the "estimate_source_cr_marginalised" function above
    #this if cycle defines the count rates grid to compute the Poisson PDF [from David Bogensberger]
    if flag_src=="SRC" :
        a, b = scipy.special.gammaincinv((cts/rate_conv)+1, 0.001), scipy.special.gammaincinv(((Bcts * Bratio) / rate_conv) + 1, 0.999)
        if a - b < 0:
            m0 = -1
        else:
            m0 = np.log10(a - b)
        m1 = np.log10(scipy.special.gammaincinv((cts/rate_conv)+1, 0.999))
    elif flag_src=="SRC+BKG" :
        a, b = scipy.special.gammaincinv(((cts+Bcts * Bratio)/rate_conv)+1, 0.001), scipy.special.gammaincinv(((Bcts * Bratio) / rate_conv) + 1, 0.999)
        if a - b < 0:
            m0 = -1
        else:
            m0 = np.log10(a - b)
        m1 = np.log10(scipy.special.gammaincinv(((cts+Bcts * Bratio)/rate_conv)+1, 0.999))
    elif flag_src=="BKG" :
        a, b = scipy.special.gammaincinv(((Bcts * Bratio)/rate_conv)+1, 0.001), scipy.special.gammaincinv(((Bcts * Bratio) / rate_conv) + 1, 0.999)
        if a - b < 0:
            m0 = -1
        else:
            m0 = np.log10(a - b)
        m1 = np.log10(scipy.special.gammaincinv(((Bcts * Bratio)/rate_conv)+1, 0.999))
    if m0 - 0.05 * (m1-m0) < -1:
        log_src_crs_grid = np.linspace(-3, m1 + 0.05*(m1 - m0), n_gp)
    else:
        log_src_crs_grid = np.linspace(m0 - 0.05 * (m1 - m0), m1 + 0.05*(m1 - m0), n_gp)
    #once the grid in count rates is defined, the pdf is computed with the "estimate_source_cr_marginalised" function (see above)
    if flag_src=="BKG" :
        pdf = estimate_source_cr_marginalised(log_src_crs_grid, Bcts, Bcts, (1/(Bratio)), rate_conv, flag_src=flag_src)
    else :
        pdf = estimate_source_cr_marginalised(log_src_crs_grid, cts, Bcts, (1/(Bratio)), rate_conv, flag_src=flag_src)

    #Determine error bars
    #Still a valid option, it's just not preferred with respect to the one below [ask David Bogensberger for more details]
    #this was computing errors from the 1sigma percentiles of the CDF
    """
    quantiles = scipy.stats.norm().cdf([-1, 0, 1])
    cdf = np.cumsum(pdf)
    lo, mid, hi = 10 ** np.interp(quantiles, cdf, log_src_crs_grid)
    return mid, mid-lo, hi-mid
    """

    #This bit below computes the errors on the count rates [ask David Bogensberger for more details, https://github.com/DavidBogensberger/eROSITA_SEP_Variability/tree/main]
    #With this method you also keep 68% of the posterior, but also satisfy the additional condition that the value of the posterior at the upper and lower errorbar limit is identical."
    mid = np.argmax(pdf)
    upper = mid + np.where(pdf[mid:] < pdf[mid] / np.exp(0.5))[0]
    lower = mid - np.where(pdf[:mid][::-1] < pdf[mid] / np.exp(0.5))[0]
    if len(upper) > 0 :
        if len(lower) > 0 :
            lower_value, mid_value, upper_value = 10**log_src_crs_grid[[lower[0], mid, upper[0]]]
        else:
            lower_value = 0
            mid_value, upper_value = 10**log_src_crs_grid[[mid, upper[0]]]
    else :
        if m0 - 0.05 * (m1-m0) < -1:
            log_src_crs_grid = np.linspace(-3, m1 + 0.2*(m1 - m0), int(1.2*n_gp) )
        else:
            log_src_crs_grid = np.linspace(m0 - 0.05 * (m1 - m0), m1 + 0.2*(m1 - m0), int(1.2*n_gp) )
        if flag_src=="BKG" :
            pdf = estimate_source_cr_marginalised(log_src_crs_grid, Bcts, Bcts, (1/(Bratio)), rate_conv, flag_src=flag_src)
        else :
            pdf = estimate_source_cr_marginalised(log_src_crs_grid, cts, Bcts, (1/(Bratio)), rate_conv, flag_src=flag_src)
        mid = np.argmax(pdf)
        upper = mid + np.where(pdf[mid:] < pdf[mid] / np.exp(0.5))[0]
        lower = mid - np.where(pdf[:mid][::-1] < pdf[mid] / np.exp(0.5))[0]
        if len(upper) > 0 :
            if len(lower) > 0 :
                lower_value, mid_value, upper_value = 10**log_src_crs_grid[[lower[0], mid, upper[0]]]
            else:
                lower_value = 0
                mid_value, upper_value = 10**log_src_crs_grid[[mid, upper[0]]]
        else :
            quantiles = scipy.stats.norm().cdf([-1, 0, 1])
            cdf = np.cumsum(pdf)
            lower_value, mid_value, upper_value = 10 ** np.interp(quantiles, cdf, log_src_crs_grid)

    return mid_value, mid_value-lower_value, upper_value-mid_value

def rebin_lc(time, counts, fracexp, bkg_counts, backratio, timedel_old):
    """
    Rebinning Srctool lightcurves in erodays.
    From Riccardo Arcodia
    """
    i = 0
    #this while runs over the fits files and rebins counts and computes errors from the functions above
    while i + 1 < len(time):
        #if there is a single bin with 0 counts in an eroday, this ensures it's correctly carried over as an "empty" eroday
        if counts[i]<=0 :
            if time[i+1]-time[i] >= rebin_time_threshold :
                Rs, Rslow, Rsup = rate_errors(counts[i], fracexp[i]*timedel_old[i], bkg_counts[i], backratio[i], flag_src="SRC")
                R, Rlow, Rup = rate_errors(counts[i], fracexp[i]*timedel_old[i], bkg_counts[i], backratio[i], flag_src="SRC+BKG")
                BR, BRlow, BRup = rate_errors(counts[i], fracexp[i]*timedel_old[i], bkg_counts[i], backratio[i], flag_src="BKG")
                yield (time[i], Rs, Rslow, Rsup, R, Rlow, Rup,
                        fracexp[i], timedel_old[i], BR, BRlow, BRup, backratio[i], counts[i], bkg_counts[i])
                i = i + 1
        for j in range(i, len(time)):
            #proceed until the eroday (100s) ends
            #print(i,j, time[i], time[j], len(time))
            if time[j]-time[i] >= rebin_time_threshold or (time[j]-time[i] < rebin_time_threshold and j + 1 == len(time)):
                if time[j]-time[i] < rebin_time_threshold and j + 1 == len(time) :
                    time_mask = np.logical_and(time >= time[i], time <= time[j])
                else :
                    time_mask = np.logical_and(time >= time[i], time < time[j])
                #averaging the time, fracexp, backratio in the eroday
                average_time=np.average(time[time_mask])
                average_fracexp=np.average(fracexp[time_mask])
                average_backratio=np.average(backratio[time_mask])
                #taking the correct timedel as the sum in the new eroday bin
                timedel=np.sum(timedel_old[time_mask])
                Rs, Rslow, Rsup = rate_errors(counts[time_mask].sum(),
                                           average_fracexp*timedel,
                                           bkg_counts[time_mask].sum(),
                                           average_backratio, flag_src="SRC")
                R, Rlow, Rup = rate_errors(counts[time_mask].sum(),
                                           average_fracexp*timedel,
                                           bkg_counts[time_mask].sum(),
                                           average_backratio, flag_src="SRC+BKG")
                BR, BRlow, BRup = rate_errors(counts[time_mask].sum(),
                                              average_fracexp*timedel,
                                              bkg_counts[time_mask].sum(),
                                              average_backratio, flag_src="BKG")
                yield (average_time, Rs, Rslow, Rsup, R, Rlow, Rup,
                        average_fracexp, timedel, BR, BRlow, BRup, average_backratio, counts[time_mask].sum(), bkg_counts[time_mask].sum())
                if j + 1 != len(time) : # go to the next eroday
                    break
                #extra condition to handle the last bin of the light curve
                #not elegant but working
                if time[j]-time[i] >= rebin_time_threshold and j + 1 == len(time) :
                    #print(i, j+1, time[j]-time[i])
                    time_mask = np.logical_and(time >= time[j], time <= time[j])
                    average_time=np.average(time[time_mask])
                    average_fracexp=np.average(fracexp[time_mask])
                    average_backratio=np.average(backratio[time_mask])
                    timedel=timedel_old[j]
                    Rs, Rslow, Rsup = rate_errors(counts[time_mask].sum(),
                                               average_fracexp*timedel,
                                               bkg_counts[time_mask].sum(),
                                               average_backratio, flag_src="SRC")
                    R, Rlow, Rup = rate_errors(counts[time_mask].sum(),
                                               average_fracexp*timedel,
                                               bkg_counts[time_mask].sum(),
                                               average_backratio, flag_src="SRC+BKG")
                    BR, BRlow, BRup = rate_errors(counts[time_mask].sum(),
                                                  average_fracexp*timedel,
                                                  bkg_counts[time_mask].sum(),
                                                  average_backratio, flag_src="BKG")
                    yield (average_time, Rs, Rslow, Rsup, R, Rlow, Rup,
                                average_fracexp, timedel, BR, BRlow, BRup, average_backratio, counts[time_mask].sum(), bkg_counts[time_mask].sum())
                    break
        i = j

def plot_rebinned_lightcurve(lc_name, indir, outdir, whichrates, time_unit = "h", log = False, overwrite_flag = False, debug = False) :

    #skip light curve if already rebinned unless you want to overwrite everything
    if os.path.exists(outdir + "/" + lc_name.replace(".fits", "_rebinned.fits")) and not overwrite_flag :
        if debug:
            print("   Skipping light curve already rebinned:", lc_name)
        return None
    else :
        if debug :
            print("   Rebinnning light curve:", lc_name)
        pass

    lc_path = indir + "/" + lc_name
    lc_all = Table.read(lc_path, format='fits')
    lc_header = fits.open(lc_path)[1].header
    
    try :
        len_bands = lc_all['COUNTS'].shape[1]
        refband_for_fracexp_cut = 1
        lc_temp_4saving = lc_all[lc_all['FRACEXP'][:, refband_for_fracexp_cut] >= 0.1]
    except :
        len_bands = 1
        lc_temp_4saving = lc_all[lc_all['FRACEXP'] >= 0.1]
    lc_tosave = lc_temp_4saving.copy()

    for band_i in range(0, len_bands) :
        fig = plt.figure(figsize = (12,7))
        ax0 = fig.add_subplot(111)

        ax0.set_ylabel(r'Count rate [cts s$^{-1}$]', fontsize = 18)
        ax0.set_xlabel('Time $-t_0$ [%s]' %time_unit, fontsize = 18)

        scale = 1
        if time_unit=="h" :
            scale = 3600
            
        ax0.yaxis.set_tick_params(labelsize=16)
        ax0.xaxis.set_tick_params(labelsize=16)
    
        try :
            lc_temp = lc_all[lc_all['FRACEXP'][:, refband_for_fracexp_cut] >= 0.1]
            bc = lc_temp['BACK_COUNTS'][:, band_i]
            fracexp = lc_temp['FRACEXP'][:, band_i]
            rate_tot_err = lc_temp['RATE_ERR'][:,band_i]
            c = lc_temp['COUNTS'][:, band_i]
        except :
            lc_temp = lc_all[lc_all['FRACEXP'] >= 0.1]
            bc = lc_temp['BACK_COUNTS']
            fracexp = lc_temp['FRACEXP']
            rate_tot_err = lc_temp['RATE_ERR']
            c = lc_temp['COUNTS']
            
        if len(lc_temp['TIME'])<2 :
            return None
        
        x = lc_temp['TIME'] - lc_temp['TIME'][0]
        bgarea = 1. / lc_temp['BACKRATIO']
        rate_conversion = fracexp * lc_temp['TIMEDEL']
        rate_tot = c / rate_conversion
        bkg_rate= bc / bgarea / rate_conversion

        x_binned, rate_src_binned, rate_src_errm_binned, rate_src_errp_binned, rate_tot_binned, rate_tot_errm_binned, rate_tot_errp_binned, fracexp_binned, timedel_binned, bkg_rate_binned, bkg_rate_errm_binned, bkg_rate_errp_binned, backratio_binned, c_binned, bc_binned = np.transpose(list(rebin_lc(x, c, fracexp, bc, 1./ bgarea, lc_temp['TIMEDEL'])))                       

        #plot new data
        ax0.errorbar(x_binned/scale, rate_tot_binned,
                     yerr=[rate_tot_errm_binned, rate_tot_errp_binned],
                     marker='.',
                     markersize=30,capsize=5.0,capthick=2.5, lw=1.5, alpha=0.95,
                     mfc='darkgreen', mec='black',ecolor='black',ls='None', zorder=2)
        
        #plot background as lines
        ax0.fill_between(x_binned/scale, bkg_rate_binned - np.array(bkg_rate_errm_binned),
                        y2 = bkg_rate_binned + np.array(bkg_rate_errp_binned),
                        facecolor='lightgrey', edgecolor='lightgrey',
                        alpha=0.6, zorder=0)
        ax0.plot(x_binned/scale, bkg_rate_binned, marker='None', linestyle='-',lw=2, color='darkgray',
                zorder=0)
        
        ax0.set_ylim(-0.01)
        if log :
            ax0.set_yscale('log')
            ax0.set_ylim(9e-4)
        emin = lc_header["E_MIN%s" % str(band_i+1)]
        emax = lc_header["E_MAX%s" % str(band_i+1)]
        if not os.path.exists(outdir + "/Images/") :
            os.mkdir(outdir + "/Images/")
        fig.savefig(outdir + "/Images/%s_emin%s_emax%s.png" % (lc_name.partition(".fits")[0], emin, emax), format='png', bbox_inches = "tight")
        
        if whichrates == "SRC+BKG" :
            rates_tosave = rate_tot_binned
            rates_tosave_errm = rate_tot_errm_binned
            rates_tosave_errp = rate_tot_errp_binned
        elif whichrates == "SRC" : 
            rates_tosave = rate_src_binned
            rates_tosave_errm = rate_src_errm_binned
            rates_tosave_errp = rate_src_errp_binned
        #save fits file, in the loop as it can be done once per energy range of course
        if band_i==0 :
            lc_tosave.remove_rows(slice(len(x_binned), len(lc_temp["TIME"])))
            lc_tosave["TIME"][:] = x_binned + lc_temp['TIME'][0]
            lc_tosave["TIMEDEL"][:] = timedel_binned
            lc_tosave["BACKRATIO"][:] = backratio_binned
            #if we want the following, adapt rebinning function to save them
            lc_tosave.remove_column('FRACFLATAREA')
            lc_tosave.remove_column('RATE_ESTPOSERR')
            lc_tosave.remove_column('RATE_ESTNEGERR')
            lc_tosave.remove_column('FRACFLATEXP')
            lc_tosave.remove_column('FRACAREA')
            lc_tosave.remove_column('FRACTIME')
            lc_tosave.remove_column('OFFAXIS')
            newcol=lc_tosave["RATE_ERR"].copy()
            newcol.name="RATE_ERRP"
            lc_tosave.add_column(newcol, index=lc_tosave.colnames.index("RATE_ERR")+1)
            lc_tosave["RATE_ERR"].name = "RATE_ERRM"
        if len_bands>1 :
            lc_tosave["COUNTS"][:, band_i] = c_binned
            lc_tosave["BACK_COUNTS"][:, band_i] = bc_binned
            lc_tosave["RATE"][:, band_i] = rates_tosave
            lc_tosave["RATE_ERRM"][:, band_i] = rates_tosave_errm
            lc_tosave["RATE_ERRP"][:, band_i] = rates_tosave_errp
            lc_tosave["FRACEXP"][:, band_i] = fracexp_binned
        else :
            lc_tosave["COUNTS"] = c_binned
            lc_tosave["BACK_COUNTS"] = bc_binned
            lc_tosave["RATE"] = rates_tosave
            lc_tosave["RATE_ERRM"] = rates_tosave_errm
            lc_tosave["RATE_ERRP"] = rates_tosave_errp
            lc_tosave["FRACEXP"] = fracexp_binned
    lc_tosave.write(outdir + "/" + lc_name.replace(".fits", "_rebinned.fits"), overwrite=True)
    if debug: 
        print("   ...done!")

if __name__ == '__main__':

    start_time=time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("-indir", help="Path to the input directory where 1+ unbinned light curves are", type=str)
    parser.add_argument("-names", help="Name of the ascii list of light curve names to be rebinned, or name of the fits file of a single light curve to be rebinned", type=str)
    parser.add_argument("-outdir", help="Path to the list of output directory for the rebinned light curves", type=str)
    parser.add_argument("-which_rates", help="Do you want source-only ('SRC') or source+bkg ('SRC+BKG') count rates in the rebinned light curve file, together with bkg-only? Default is 'SRC+BKG'", default="SRC+BKG", type=str)
    parser.add_argument("-overwrite", help="Do you want to overwrite light curves already rebinned? y/n. Default is n.", default="n", type=str)
    parser.add_argument("-debug", help="Do you want to print which light curve is being rebinned, for debugging?", default="n", type=str)
    args = vars(parser.parse_args())

    if args["which_rates"]!="SRC" and args["which_rates"]!="SRC+BKG":
        parser.error("at least one of 'SRC' and 'SRC+BKG' required for 'which_rates'")

    singlesource = False
    if args["names"].endswith(".fits") or args["names"].endswith(".fits.gz") :
        singlesource = True
        lc_name = args["names"]
    else :
        filename = args["names"]

    input_dir = args["indir"]
    output_base_dir = args["outdir"]
    whichrates  = args["which_rates"]

    overwrite_flag = False
    if args["overwrite"] == "y" or args["overwrite"] == "yes" or args["overwrite"] == "True" :
        overwrite_flag = True

    debug = False
    if args["debug"] == "y" or args["debug"] == "yes" or args["debug"] == "True" :
        debug = True


    if not os.path.exists(output_base_dir) :
        os.mkdir(output_base_dir)

    #implement multiprocessing here and test
    print("\n", "Rebinning the light curves..")
    if singlesource :
        plot_rebinned_lightcurve(lc_name = lc_name,
                                       indir = input_dir,
                                       outdir = output_base_dir,
                                       whichrates = whichrates, 
                                       time_unit = "h",
                                       log = False,
                                       overwrite_flag = overwrite_flag,
                                       debug = debug)
    else :
        n_proc = mp.cpu_count()

        plot_rebinned_lightcurve_partial = partial(plot_rebinned_lightcurve,
                                            indir = input_dir,
                                            outdir = output_base_dir, 
                                            whichrates = whichrates,
                                            time_unit = "h",
                                            log = False,
                                            overwrite_flag = overwrite_flag,
                                            debug = debug)

        lc_names = ascii.read(input_dir + "/" + filename, format='no_header')["col1"].value  

        with mp.Pool(processes=n_proc) as pool:
                pool.map(plot_rebinned_lightcurve_partial, lc_names)

    #-------------We are done!-----------------
    print("\n", "All done!")
    elapsed=(time.time() - start_time)/60
    print("\n","Total time elapsed is %s min." %str(elapsed),"\n")
