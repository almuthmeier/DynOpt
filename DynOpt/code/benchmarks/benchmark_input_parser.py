'''
Used to create benchmark file on the server.

Created on Nov 19, 2018

@author: ameier
'''

import os
import sys


# otherwise import error when script is run
path_to_code = os.path.abspath(os.pardir)  # .split('/')
#path_to_dynopt = ('/').join(path_to_code[:-1])
path_to_benchmarks = path_to_code + "/benchmarks"
sys.path.append(path_to_benchmarks)  # path to benchmark directory


def define_parser_arguments():
    import argparse
    parser = argparse.ArgumentParser()

    # for all benchmarks
    parser.add_argument("-benchmarkfunction", type=str)
    parser.add_argument("-benchmarkfunctionfolderpath", type=str)
    return parser


def read_input_values(parser):
    args = parser.parse_args()
    return (args.benchmarkfunction,
            args.benchmarkfunctionfolderpath)


def run_parser():
    parser = define_parser_arguments()
    values = read_input_values(parser)  # returns a tupel
    # pass values of tupel as separate parameters
    if parser.benchmarkfunction == "mpbrand" or parser.benchmarkfunction == "mpbnoisy" or \
            parser.benchmarkfunction == "mpbcorr":
        start_creating_problem(*values)  # use all values
    elif parser.benchmarkfunction == "srr":  # sphere, rastrigin, rosenbrock are created at once
        create_problems(values[1])  # use only path


if __name__ == '__main__':
    from benchmarks.dynposbenchmark import create_problems
    from benchmarks.mpb import start_creating_problem
    run_parser()
