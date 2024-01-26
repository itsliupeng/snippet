import os
import pickle
import requests
import numpy as np
import random
import re


def list_to_string(a):
    a = str(a)
    return a.replace(' ', '')

def num_to_list(num):
    return [int(x) for x in str(num)]

def get_input_string(x,y):
    x, y = str(x), str(y)
    input_str = f'Input:\n{x}+{y}\n'

    return input_str


def get_output_string(x, y):
    x, y = str(x), str(y)

    len_x, len_y = len(x), len(y)
    list_x, list_y = num_to_list(x), num_to_list(y)

    output_str = f'Target:\n<scratch>\n'

    output_str += f'{list_to_string(list_x)} has {len_x} digits.\n'
    output_str += f'{list_to_string(list_y)} has {len_y} digits.\n'

    C = 0
    A = []
    for i in range(max(len_x, len_y)):
        a = list_x[-1] if i < len_x else 0
        b = list_y[-1] if i < len_y else 0
        c = a + b + C

        output_str += f'{list_to_string(list_x)} + {list_to_string(list_y)} , A={list_to_string(A)} , C={C} , {a}+{b}+{C}={c} , A->{c % 10} , C->{c // 10}\n'

        A.insert(0, c % 10)
        C = c // 10

        list_x = list_x[:-1]
        list_y = list_y[:-1]

    output_str += f'{list_to_string(list_x)} + {list_to_string(list_y)} , A={list_to_string(A)} C={C} , END\n</scratch>\n'
    if C == 1:
        A.insert(0, 1)
    for a in A:
        output_str += f'{a} '

    return output_str[:-1] + '\n'


def make_addition_examples(pad=True):
    print('making examples of a+b=c')
    input_file_path = os.path.join(os.path.dirname(__file__), 'add_examples.txt')
    # if not os.path.exists(input_file_path):
    with open(input_file_path, 'w+') as f:
        for i in range(10000):
            a, b = random.randint(0,999), random.randint(0,999)
            c = a + b
            if pad:
                f.write(f"{a:03}+{b:03}={c}\n")
            else:
                f.write(f"{a}+{b}={c}\n")

def create_add_data(total_num_examples=10000):
    digit_set = set()
    result_list = []
    for i in range(total_num_examples):
        a, b = random.randint(0, 9999), random.randint(0, 9999)
        c = a + b

        if (a, b) in digit_set:
            continue

        digit_set.add((a, b))
        # y = math.sin(x_trunc)
        # y_trunc = truncate_to_4_digit(y)

        input_str = get_input_string(a, b)
        output_str = get_output_string(a, b)
        result_str = f"{input_str}{output_str}"
        result_list.append(result_str)

    return result_list

gen_list = create_add_data(300000)
j_list = [{'id': idx, 'content': x} for idx, x in enumerate(gen_list)]
