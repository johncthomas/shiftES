# shiftES
Implimentation of Wilcox shift effect size with command line interface,
"A Robust Nonparametric Measure of Effect Size Based on an Analog of Cohen's d...", R. Wilcox (2018). https://dx.doi.org/10.22237/jmasm/1551905677


Run from command line, `shift_effectsize.py -h` for help.

Input files should be structured with a column of values per sample, e.g.
```
Samp1 Samp2 Samp3
4.3   7.5   3.4
4.5   8.3   2.4
```
Files can be comma/tab separated values, or a Excel .xlsx. File type will be automatically detected.


To test every sample against every other sample, with 95% confidence intervals, use:  
`shift_effectsize.py input_file_name.csv ALL ALL -o shiftes_results.csv --ci`

Results will be saved in a table saved as `shiftes_results.csv`.

The given effect size is Ω, which ranges between -1 and +1 and is described in Wilcox's paper. As a guide: small 0.1; medium 0.3; large 0.4

To use within Python `from shiftES import effectsize, effectsize_ci, difference_dist` and see inline documentation.