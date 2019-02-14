import numpy as np
import csv

def pad(text, max_seq_length):
    if len(text) > max_seq_length:
        return text[:max_seq_length]
    return np.pad(text, (0, max_seq_length - len(text)),
                  'constant',
                  constant_values=0)

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
    with open(file_name, "a") as file:
        for text in lst:
            file.write(text + "\n")

def remove_whitespace(lst):
    return list(filter(lambda x: len(x.strip()) > 0, lst))

def cosine_similarity(u, v):
    dot_prod = np.dot(u, v)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    return 1.0 - (dot_prod / (u_norm * v_norm))

def try_parse_int(val, default):
    try:
        return int(val)
    except ValueError:
        return default

def value_if_non_empty(val, default):
    if val == None or len(val) == 0:
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
    with open(file_name, "a+") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",", quotechar="|")
        csv_writer.writerow(record)