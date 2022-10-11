"""Implimentation of some functions described in:

Wilcox, Rand. "A robust nonparametric measure of effect size based on an
analog of Cohen's d, plus inferences about the median of the typical difference."
Journal of Modern Applied Statistical Methods 17.2 (2019): 1.
https://dx.doi.org/10.22237/jmasm/1551905677

Functions:
    difference_dist(a, b): All of the difference between values in a, b
    Q(a, b): Calculate Q value.
"""
import itertools
import multiprocessing
import typing

import numpy as np
import pandas as pd
import argparse
import csv
from typing import Union
from multiprocessing import Pool
import logging
logging.basicConfig()




def difference_dist(X1, X2) -> np.ndarray:
    """All of the difference between values in X1, X2"""
    a_m = np.tile(X1, (X2.shape[0], 1))
    b_m = np.tile(X2, (X1.shape[0], 1))
    D = np.ravel(a_m.T - b_m)
    return D


def _clean_np_arrays(a, b):
    a, b = np.array(a), np.array(b)
    a, b = [ab[~np.isnan(ab)] for ab in (a, b)]
    return a, b

def effectsize(X1, X2, max_obsv=1000, report_Q=False):
    """Calculate Wilcox Ω effect size. This is a transformation of Q
    described in Wilcox (2018) ranging -1 to 1 with 0 indicating no effect.

    Guide to effect sizes (equiv to Cohen's d small/medium/large):
        |Ω|: small 0.1; medium 0.3; large 0.4
        Q: small 0.55; medium 0.65; large 0.70

    Args:
        X1, X2: 1D collections of numbers
        report_Q: Report Q value not Ω. Ω=(Q-0.5)/0.5
        max_obsv: If X1 or X2 contain more than max_obsv, they are resampled down
            to max_obsv. Set to False to always use all observations. (note
            an array of size len(a)*len(b) is generated)
    """


    def sampdown(ab):
        """sample down to max_obsv if we have more than that already"""
        if ab.shape[0] > max_obsv:
            return ab[np.random.randint(0, ab.shape[0], size=max_obsv)]
        return ab

    X1, X2 = _clean_np_arrays(X1, X2)
    X1, X2 = [sampdown(ab) for ab in (X1, X2)]

    D = difference_dist(X1, X2)
    # Y = D, were the null hyp true
    median = D.dtype.type(np.median(D))

    Y = D - median

    # Q: proportion of Y that are less than equal to actual median
    Q = np.sum(Y < median) / Y.shape[0]


    if not report_Q:
        return (Q-0.5)/0.5
    else:
        return Q

def _load_table(fn, filetype, header=True, sheet=0):
    """Load table using arguments given at command line.
    Headerless tables will have header made up of string integers
    starting from one, as the CL args assume one indexed and are str by default."""

    valid_types = ('a', 'c', 't', 'x')
    if filetype not in valid_types:
        raise ValueError(f"{filetype} not a valid file type token, use one of {', '.join(valid_types)}")

    # autodetect filetype
    if filetype=='a':
        lowfn = fn.lower()
        if lowfn.endswith('.tsv') or lowfn.endswith('.txt'):
            filetype = 't'
        elif lowfn.endswith('.csv'):
            filetype = 'c'
        elif lowfn.endwith('.xlsx'):
            filetype = 'x'
        else:
            raise ValueError(f"Couldn't automatically detect type of file, {fn}")


    header = [None, 0][header]

    if filetype in 'ct':
        sep = dict(c=',', t='\t')[filetype]
        table = pd.read_csv(fn, sep=sep, header=header)
    elif filetype == 'x':
        sheet = int(sheet)
        table = pd.read_excel(fn, header=header, sheet_name=sheet)
    # the else is covered above, no other values should make it this far

    if not header:
        table.columns = table.columns.map(lambda x: str(x+1))

    return table


def effectsize_ci(X1, X2, ci=0.05, max_obsv=1000, nboot=500, report_Q=False) \
        -> typing.Tuple[float, float, float]:
    """Calculate confidence intervals of effect size. """

    es_kwargs = dict(report_Q=report_Q, max_obsv=max_obsv)

    X1, X2 = _clean_np_arrays(X1, X2)
    qs = np.empty(shape=nboot, dtype=np.float32)

    na, nb = [min(len(ab), max_obsv) for ab in (X1, X2)]

    es = effectsize(X1, X2, **es_kwargs)

    for i in range(nboot):
        resampled_ab = []
        for ab, n_ab in [(X1, na), (X2, nb)]:
            resampled_ab.append(
                ab[np.random.randint(0, len(ab), size=n_ab)]
            )
        qs[i] = np.float32(effectsize(*resampled_ab, **es_kwargs))
    lower = ci
    upper = 1-ci
    return es, np.quantile(qs, lower), np.quantile(qs, upper)



def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Calculate robust nonparametric effect size, described by R. Wilcox, "
            "https://dx.doi.org/10.22237/jmasm/1551905677\n\n"
            "Effect sizes. Small: 0.55. Medium: 0.6. Large: 0.7\n"

        )
    )

    parser.add_argument(
        'file',
        help='Path of input file, an CSV (comma or tab sep) or Excel file.',
    )
    parser.add_argument(
        'columnA', metavar="list,of,columns", type=str,
        help="Name(s) of column (if file has headers) or number(s) of first column(s) containing data to be tested. "
             "Column numbers are 1-indexed. Putting 'ALL' for both columnA/B will test every pair of ",
    )
    parser.add_argument(
        'columnB', metavar="list,of,columns", type=str,
        help="Name of column/number of second column.",
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help="File to write the results to, if none just prints"
    )
    parser.add_argument(
        '-c', '--ci',default=False,
        action='store_true',
        help="Set flag to calculate confidence intervals using the bootstrap method. Use -b to change"
             " number of boostrap iterations (default: 500)"
    )
    parser.add_argument(
        '-b', '--nboot', type=int, default=500,
        help="Change number of bootstrap iterations when calculating confidence intervals. Default: 500."
    )
    parser.add_argument(
        '-t', '--type', default='a',
        help="Input file type, can be 'a' for autodetect, 'c' for comma-separated-values, 't' for tab-sep-vals, "
             "'x' for .xlsx . If 'a' the script will guess based on the file name."
    )
    parser.add_argument(
        '-n', '--no-header',
        action='store_true',
        default=False,
        help="Set to indicate that first row contains data and columns will give numerical position."
    )
    parser.add_argument(
        '--sheet-number', default=1,
        help="Excel only: if the sheet is not the first one, give its number here. 1-indexed"
    )
    # parser.add_argument(
    #     '-q'
    # )

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = parse_args()

    colA, colB = args.columnA, args.columnB

    table = _load_table(
        fn=args.file,
        filetype=args.type,
        header=(not args.no_header),
        sheet=args.sheet_number-1,
    )


    if (colA == 'ALL') and (colB == 'ALL'):
        cols = table.columns
        cols = cols[~table.columns.str.startswith('Unnamed:')]
        column_pairs = list(itertools.combinations(cols, 2))
    else:
        column_pairs = list(
            [ab for ab in zip(colA.split(','), colB.split(','), )]
        )

    if args.no_header:
        column_pairs = [(ab[0], ab[1]) for ab in column_pairs]
        aw, bw = 3,3
    else:
        # for printing, width of largest sample name
        widths = [1, 1]
        for pair in column_pairs:
            for i in (0, 1):
                l = len(pair[i])
                if l > widths[i]:
                    widths[i] = l
        aw, bw = widths

    if args.ci:
        results = [('S1', 'S2', 'Omega', 'Lower 95% CI', 'Upper 95% CI')]
    else:
        results = [('S1', 'S2', 'Omega')]

    for k1, k2 in column_pairs:
        X1, X2 = table[k1], table[k2]

        # produce formatable string with word widths written in
        if args.ci:
            stats = effectsize_ci(X1, X2)
            fstr = f"{{:>{aw}}} : {{:<{bw}}} {{: .3}} [{{: .3}} {{: .3}}]"

        else:
            stats = (effectsize(X1, X2),)
            fstr = f"{{:>{aw}}} : {{:<{bw}}} {{: .3}}"

        print(fstr.format(k1, k2, *stats))
        results.append((k1,k2, *stats))

    if args.output:
        results = pd.DataFrame(results)
        fn = args.output
        kwargs = dict(index=False, header=False)
        if fn.endswith('.tsv') or fn.endswith('.txt'):
            results.to_csv(fn, sep='\t', **kwargs)
        elif fn.lower().endswith('.xlsx'):
            results.to_excel(fn, **kwargs)
        else:
            results.to_csv(fn, **kwargs)




