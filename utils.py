import numpy as np
import csv
import pickle
import gzip
from constants import *


def flatten(lists):
    """Returns a flattened version of the given list of lists"""
    flattened = []
    for token_list in lists:
        for lst in token_list:
            flattened += lst
    return flattened


def load_data_file(file_name):
    """Returns a list of strings where each string represents a line in the given file"""
    dataset = []
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            dataset.append(line.split())
    return dataset


def append_to_file(lst, file_name):
    """Appends each element of the given list to the given file."""
    with open(file_name, 'a') as file:
        for text in lst:
            file.write(text + '\n')


def remove_whitespace(lst):
    """Removes whitespace from every element in the given list."""
    return list(filter(lambda x: len(x.strip()) > 0, lst))


def value_if_non_empty(val, default):
    """Returns the default value if val is None or empty. Otherwise returns val."""
    if val is None or len(val) == 0:
        return default
    return val


def lst_equal(lst1, lst2):
    """Returns true if the two lists are equal element-by-element."""
    if len(lst1) != len(lst2):
        return False
    for i in range(0, len(lst1)):
        if (lst1[i] != lst2[i]):
            return False
    return True


def log_record(file_name, record):
    """Appends the record to the given log CSV file."""
    with open(file_name, 'a+') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|')
        csv_writer.writerow(record)


def write_methods_to_file(file_name, method_arr):
    """
    Writes the array of method bodies to the given file.
    These methods are wrapped in a class to enable formatting using the Google JAVA formatter.
    """
    with open(file_name, 'w') as file:
        file.write('public class Results {\n\n')
        for method in method_arr:
            file.write(method + '\n\n')
        file.write('}\n')


def add_slash_to_end(dir_path):
    """
    Returns the given directory path with a slash at the end if
    the given string does not already end in a slash.
    """
    if dir_path[-1] != '/':
        return dir_path + '/'
    return dir_path


def get_index(key):
    """Returns the ID as specified in the Redis key."""
    return key.split(':')[1].replace('\'', '')


def load_parameters(restore_dir):
    """Returns a Parameter object serialized in the restore directory."""
    path = restore_dir + META_NAME
    with gzip.GzipFile(path, 'rb') as in_file:
        meta_data = pickle.load(in_file)
    return meta_data['parameters']


def get_ranking_in_array(arr, x):
    """Returns the ranking (1 + index) of element x in the array. Returns -1 if x is not present."""
    for i, elem in enumerate(arr):
        if elem == x:
            return i + 1
    return -1
