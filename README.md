# shiftes
Implimentation of Wilcox shift effect size with command line interface,
"A Robust Nonparametric Measure of Effect Size Based on an Analog of Cohen's d...", R. Wilcox (2018). https://dx.doi.org/10.22237/jmasm/1551905677

Run from command line, `python calculate_effectsize.py -h` for help.

Input files should be structured with a column of values per sample, e.g.
```Samp1  Samp2 Samp3
4.3 7.5 3.4
4.5 8.3 2.4
etc...```

To test every sample against every other sample in a table saved as `data.csv` use:  
`python calculate_effectsize.py data.csv ALL ALL -o results.csv`

To use within Python `from shiftES import effectsize, effectsize_ci, difference_dist` and see inline documentation.
