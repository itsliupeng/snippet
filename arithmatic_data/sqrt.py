import math
import random
import os

def truncate_to_4_digit(x):
    return math.floor(x * 10000) / 10000


def get_input_string(x: float, operator='sqrt'):
    x_trunc = truncate_to_4_digit(x)
    input_str = f'Input:\n{operator}({x_trunc})\n'
    input_str += f'Target:\n'

    return input_str

def truncate_to_n_digit(x, n=4):
    return math.floor(x * (10 ** n)) / (10 ** n)

def get_output_string(x, y=0, n=5):
    output_str = f'<scratch>\n'

    a = x
    x_true = truncate_to_n_digit(math.sqrt(a), 4)
    this_x = x_true

    if this_x >= 1:
        this_x = int(this_x)
    else:
        this_x = 0.1
    output_str += f'x_0={this_x}\n'

    for i in range(1, n + 1):
        x_i = this_x

        this_x = 0.5 * (this_x + a / this_x)
        this_x = truncate_to_n_digit(this_x, 4)

        output_str += f'x_{i}: 1/2*({x_i}+{a}/{x_i})={this_x}, x_{i}={this_x}'

        if not i == n:
            output_str += '\n'

    output_str += ' , END\n</scratch>\n'

    output_str += f'{this_x}\n'

    return output_str[:-1] + '\n'


def create_sqrt_data(total_num_examples=10000):
    result_list = []
    for i in range(total_num_examples):
        x = random.uniform(0, 100)
        x_trunc = truncate_to_4_digit(x)
        # y = math.sqrt(x_trunc)
        # y_trunc = truncate_to_4_digit(y)
        # f.write(f'sqrt({x_trunc})={y_trunc}\n')
        input_str = get_input_string(x_trunc)
        output_str = get_output_string(x_trunc)
        result_str = f"{input_str}{output_str}"
        result_list.append(result_str)
    return result_list

gen_list = create_sqrt_data(300000)
j_list = [{'id': idx, 'content': x} for idx, x in enumerate(gen_list)]

