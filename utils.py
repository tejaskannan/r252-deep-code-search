import numpy as np
import csv
import pickle
import gzip
import heapq
from constants import *


def flatten(lists):
    flattened = []
    for token_list in lists:
        for lst in token_list:
            flattened += lst
    return flattened


def load_data_file(file_name):
    dataset = []
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            dataset.append(line.split())
    return dataset


def append_to_file(lst, file_name):
    with open(file_name, 'a') as file:
        for text in lst:
            file.write(text + '\n')


def remove_whitespace(lst):
    return list(filter(lambda x: len(x.strip()) > 0, lst))


def cosine_similarity(u, v):
    dot_prod = np.dot(u, v)
    u_norm = np.linalg.norm(u) + SMALL_NUMBER
    v_norm = np.linalg.norm(v) + SMALL_NUMBER
    return dot_prod / (u_norm * v_norm)


def value_if_non_empty(val, default):
    if val is None or len(val) == 0:
        return default
    return val


def lst_equal(lst1, lst2):
    if len(lst1) != len(lst2):
        return False
    for i in range(0, len(lst1)):
        if (lst1[i] != lst2[i]):
            return False
    return True


def log_record(file_name, record):
    with open(file_name, 'a+') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|')
        csv_writer.writerow(record)


def write_methods_to_file(file_name, method_arr):
    with open(file_name, 'w') as file:
        for method in method_arr:
            file.write(method + '\n')
            file.write(LINE + '\n')


def add_slash_to_end(dir_path):
    if dir_path[-1] != '/':
        return dir_path + '/'
    return dir_path


def get_index(key):
    return key.split(':')[1].replace('\'', '')


def load_parameters(restore_dir):
    path = restore_dir + META_NAME
    with gzip.GzipFile(path, 'rb') as in_file:
        meta_data = pickle.load(in_file)
    return meta_data['parameters']


# heap is a min PQ
def get_ranking_in_heap(heap, x):
    rank = 1
    num_elems = len(heap)
    while len(heap) > 0:
        elem = heapq.heappop(heap)
        if int(get_index(elem[2])) == x:
            return (num_elems - rank) + 1
        rank += 1
    return -1

def get_ranking_in_array(arr, x):
    for i, elem in enumerate(arr):
        if elem == x:
            return i + 1
    return -1


def softmax(arr):
    max_elem = np.max(arr)
    arr_exp = np.exp(arr - max_elem)
    return arr_exp / np.sum(arr_exp)
