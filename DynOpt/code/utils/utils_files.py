'''
Functionality to print the result files.
'''
import datetime


def print_to_file(file_name, values):
    '''
    Prints a list of values as one line into a csv file.
    @param values: list of scalar values containing the values for this metrics (in this specific order):
    file_name, problem, dim, pos_chng_type, fit_chng_type, avg_bog, arr_avg, arr_stddev, bebc_avg, bebc_stddev
    @param file_name: file name including its path and format
    '''
    # convert list of values to string of comma separated values
    # https://stackoverflow.com/questions/44778/how-would-you-make-a-comma-separated-string-from-a-list
    # (16.10.17)
    line = ','.join(map(str, values)) + "\n"
    print_line(file_name, line)


def print_line(file_name, line):
    '''
    Appends line to result csv-file.
    @param file_name: file name including its path and format
    '''
    # https://stackoverflow.com/questions/31948879/using-explict-predefined-validation-set-for-grid-search-with-sklearn
    file = open(file_name, 'a')
    file.write(line)
    file.close()


def get_current_day_time():
    current_date = datetime.datetime.now()
    day = str(current_date.year) + '-' + \
        str(current_date.month).zfill(2) + '-' + str(current_date.day).zfill(2)
    time = str(current_date.hour).zfill(2) + ':' + \
        str(current_date.minute).zfill(2)
    return day, time


def get_metrics_file_name(metrics_file_path, predictor_name, benchmarkfunction, day, time):
    return metrics_file_path + predictor_name + "_" + benchmarkfunction + "_" + \
        day + '_' + time + ".csv"


def get_logs_file_name(logs_file_path, predictor_name, benchmarkfunction, day, time):
    return logs_file_path + predictor_name + "_" + benchmarkfunction + "_" + \
        day + '_' + time + ".txt"