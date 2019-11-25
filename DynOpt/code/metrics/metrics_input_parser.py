'''
Used to create benchmark file on the server.

Created on Nov 19, 2018

@author: ameier
'''

import os
import sys
import numpy as np

# otherwise import error when script is run
path_to_code = os.path.abspath(os.pardir)
sys.path.append(path_to_code)


def int_list_type(string):
    '''
    Input type for argparse.
    Otherwise it is not possible to read in a list of int.
    https://www.tuxevara.de/2015/01/pythons-argparse-and-lists/ (12.5.18)

    @param string: a list as string, e.g.: '2,5,7,50'
    @return the string as numpy array of integers
    '''
    strings = string.split(',')
    return np.array([int(i) for i in strings])


def float_list_type(string):
    strings = string.split(',')
    return np.array([float(i) for i in strings])


def string_list_type(string):
    strings = string.split(',')
    return np.array([i for i in strings])


def define_parser_arguments():
    import argparse
    parser = argparse.ArgumentParser()

    # for all benchmarks
    parser.add_argument("-benchmarkfunctionfolderpath", type=str)
    parser.add_argument("-outputpath", type=str)
    parser.add_argument("-benchmarkfunctions", type=string_list_type)
    parser.add_argument("-poschgtypes", type=string_list_type)
    parser.add_argument("-fitchgtypes", type=string_list_type)
    parser.add_argument("-dims", type=int_list_type)
    parser.add_argument("-noises", type=float_list_type)
    parser.add_argument("-pathaddition", type=str)
    parser.add_argument("-metricfilename", type=str)
    parser.add_argument("-onlyforpreds", type=str)
    parser.add_argument("-arrwithabs", type=str)
    parser.add_argument("-rcswithabs", type=str)
    return parser


def read_input_values(parser):
    args = parser.parse_args()
    return (args.benchmarkfunctionfolderpath,
            args.outputpath,
            args.benchmarkfunctions,
            args.poschgtypes,
            args.fitchgtypes,
            args.dims,
            args.noises,
            args.pathaddition,
            args.metricfilename,
            args.onlyforpreds == 'True',
            args.arrwithabs == 'True',
            args.rcswithabs == 'True')


def run_parser():
    parser = define_parser_arguments()
    values = read_input_values(parser)  # returns a tupel

    start_computing_metrics(*values)


if __name__ == '__main__':
    from metrics.metric_calculator import start_computing_metrics
    run_parser()
