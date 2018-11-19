'''
Used to create benchmark file on the server.

Created on Nov 19, 2018

@author: ameier
'''

import os
import sys


# otherwise import error when script is run
path_to_code = os.path.abspath(os.pardir)
sys.path.append(path_to_code)


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
    benchmark_fct = values[0]
    benchmark_path = values[1]

    # pass values of tupel as separate parameters
    if benchmark_fct == "mpbrand" or benchmark_fct == "mpbnoisy" or \
            benchmark_fct == "mpbcorr":
        start_creating_problem(*values)  # use all values
    elif benchmark_fct == "srr":  # sphere, rastrigin, rosenbrock are created at once
        create_problems(benchmark_path)  # use only path


if __name__ == '__main__':
    from benchmarks.dynposbenchmark import create_problems
    from benchmarks.mpb import start_creating_problem
    run_parser()
