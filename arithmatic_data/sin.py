import os
import pickle
import requests
import numpy as np
import random
import re

import math
import random
import os


def truncate_to_4_digit(x):
    return math.floor(x * 10000) / 10000


def truncate_to_n_digit(x, n=4):
    return math.floor(x * (10 ** n)) / (10 ** n)


def get_input_string(x: float, operator='sin'):
    x_trunc = truncate_to_n_digit(x)
    input_str = f'Input:\n{operator}({x_trunc})\n'
    input_str += f'Target:\n'

    return input_str


def get_output_string(x, y=0, n=4):
    output_str = f'<scratch>\n'

    x_true = truncate_to_n_digit(x, 4)
    this_x = x_true

    output_str += f'x_0={this_x}\n'

    for i in range(1, n + 1):
        k = 2 * i + 1

        x_i = this_x

        this_x = this_x + (-1) ** i * (x ** k) / (math.factorial(k))
        this_x = truncate_to_n_digit(this_x, n)

        plus_minus = '+ 1' if i % 2 == 0 else '- 1'

        output_str += f'x_{i}: x_{i - 1} {plus_minus}/{k}! * (x^{k}) , x_{i}={this_x}'

        if not i == n:
            output_str += '\n'

    output_str += ' , END\n</scratch>\n'

    output_str += f'{this_x}\n'

    return output_str[:-1] + '\n'


def create_sin_data(total_num_examples=10000):
    digit_set = set()
    result_list = []
    for i in range(total_num_examples):
        x = random.uniform(-math.pi / 2, math.pi / 2)
        x_trunc = truncate_to_4_digit(x)
        if x_trunc in digit_set:
            continue

        digit_set.add(x_trunc)
        # y = math.sin(x_trunc)
        # y_trunc = truncate_to_4_digit(y)

        input_str = get_input_string(x_trunc)
        output_str = get_output_string(x_trunc)
        result_str = f"{input_str}{output_str}"
        result_list.append(result_str)

    return result_list

gen_list = create_sin_data(1000000)
j_list = [{'id': idx, 'content': x} for idx, x in enumerate(gen_list)]

