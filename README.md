# eRebin
Code for rebinning eROSITA all-sky survey (eRASS) light curves per eroday.

An 'eroday' for eROSITA is a fixed interval of 4 hours (also corresponding to the duration of one revolution of the spacecraft in all-sky survey mode). The on-target exposure of an eroday is ~40s. By default eRASS light curves are provided in 10s bins, so with a handful of data points per eroday. However, for most cases one would want to study the variability across these 40s scans.

This code helps you to produce rebinned light curves, with asymmetric Poisson uncertainties, in all the default energy ranges.

For more details on eROSITA, the eRASS exposure and strategy etc, check [this paper](https://ui.adsabs.harvard.edu/abs/2024A%26A...682A..34M/abstract).

For an example on the outcome of this rebinning code, check the eRASS light curves in [this paper](https://ui.adsabs.harvard.edu/abs/2021Natur.592..704A/abstract) and [this too](https://ui.adsabs.harvard.edu/abs/2024arXiv240117275A/abstract). Please cite these two papers if you use this code!

## Basic usage

By running 

```
python3 Rebin_eROday.py -h
```

you'll see the required and optional arguments:

```
  -indir INDIR          Path to the input directory where 1+ unbinned light curves are
  -names NAMES          Name of the ascii list of light curve names to be rebinned, or name of the
                        fits file of a single light curve to be rebinned
  -outdir OUTDIR        Path to the list of output directory for the rebinned light curves
  -which_rates WHICH_RATES
                        Do you want source-only ('SRC') or source+bkg ('SRC+BKG') count rates in
                        the rebinned light curve file, together with bkg-only? Default is 'SRC+BKG'
  -overwrite OVERWRITE  Do you want to overwrite light curves already rebinned? y/n. Default is n.
  -debug DEBUG          Do you want to print which light curve is being rebinned, for debugging?
```

So a simple case in which you want to rebin data which is in a ~/data/input/ directory:

```
python3 Rebin_eROday.py -indir "~/data/input/" -names "list.txt" -outdir "~/data/output/"
```

Where for convenience, you ideally have listed the light curve names in a file "list.txt" on /input/, so you don't need to waste time in case you bulk downloaded tens of thousands of them.

## Output

eRASS light curves come by default in three energy bands: 0.2-0.6keV, 0.6-2.3keV and 2.3-5.0keV.
The "RATE" array is infact 3D for this reason.

The code rebins the light curves in the three energy ranges independently, so the same structure is kept.

The counts are rebinned in each eroday, and the count rate is numerically computed from the Poisson mass funciton in a grid of count rate values. The median and 1 sigma equivalent uncertainties are computed from this function.
Rates are computed for background only, source only and source+background. By defauls, source+background and background only are saved in the rebinned light curve. All ucnertainties are therefore asymmetric. A non detection would correspond to a case in which the source+background and background data points are compatible within their uncertainties. 

With '-which_rates "SRC"' you can save source-only count rates, with the caveat what the Poisson function is zero-bound for non-detections, which you'll see as data points in which median and/or lower bound touch zero.

In the output directory, there is a subdirectory called "Images" where you find the plotted rebinned light curves, one per energy range in the input light curve.

The file test.zip contains an example with ~50 eRASS1 light curves in input and the rebinned output plus images.

## How to cite

If you make use of this code, please link to this github repo and cite [this paper](https://ui.adsabs.harvard.edu/abs/2021Natur.592..704A/abstract) and [this too](https://ui.adsabs.harvard.edu/abs/2024arXiv240117275A/abstract).

Contact me here or at rarcodia@mit.edu for questions/bugs/changes!
