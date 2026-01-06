import sys,os
sys.path.append(os.path.dirname(sys.path[0]))
import json
import numpy as np
import os
from root import rootpath
import random

map_dic = {'X': 0, 'S': 1, '-': 2, '?': 3, 'Q': 4, 'E': 5, '<': 6, '>': 7, '[': 8, ']': 9, 'o': 10}
phareser = ['X', 'S', '-', '?', 'Q', 'E', '<', '>', '[', ']', 'o']

# number with string
def arr_to_str(level):
    height = len(level)
    width = len(level[0])
    str = ''
    for i in range(height):
        for j in range(width):
            str += phareser[level[i][j]]
        if i < height - 1:
            str += '\n'
    return str

def arr_to_num_str(level):
    height = len(level)
    width = len(level[0])
    result = ''
    for i in range(height):
        for j in range(width):
            result += str(level[i][j])
        if i < height - 1:
            result += '\n'
    return result

# def arr_to_str(level):
#     height = len(level)
#     width = len(level[0])
#     result = ""
#     for i in range(height):
#         for j in range(width):
#             result += str(level[i][j])
#         if i < height - 1:
#             result += "\n"
#     return result

def numpy_level(string):
    data = string.split('\n')

    height = len(data)
    if len(data[height - 1]) == 0: height -= 1
    width = len(data[0])  # '\n' is not included
    whole_level = np.empty((height, width), dtype=int, order='C')
    for i in range(height):
        for j in range(width):
            whole_level[i][j] = map_dic[data[i][j]]
    return whole_level

def little_level(level, size):
    height = len(level)
    width = len(level[0])
    litte_level = np.empty((height // size, width // size), dtype=int, order='C')
    cnt = [0] * len(map_dic.keys())
    for i in range(height // size):
        for j in range(width // size):
            for k in range(len(map_dic.keys())):
                cnt[k] = 0
            for k in range(size):
                for l in range(size):
                    cnt[level[i * size + k][j * size + l]] += 1
            litte_level[i][j] = random.sample(list(np.where(cnt == np.max(cnt))[0]), 1)[0]
    return litte_level

def addLine(lv):
    n = len(lv)
    return np.concatenate([lv[0:1], lv[0:n], lv[n-1:n]], axis=0)

def calculate_broken_pipes(data):
    rule_file = json.load(open(rootpath + '//CNet//data//legal_rule.json'))
    rule = set()
    for e in rule_file:
        rule.add(tuple(e))
    height = len(data)
    width = len(data[0])
    cnt = 0
    for i in range(height):
        for j in range(width):
            flag = False
            info = [i]
            for i1 in range(-1, 2):
                for j1 in range(-1, 2):
                    ni = i + i1
                    nj = j + j1
                    if ni == i and nj == j:
                        continue
                    if ni < 0 or nj < 0 or ni >= height or nj >= width:
                        info.append(11)
                    else:
                        info.append(data[ni][nj])
                        if 6 <= data[ni][nj] <= 9:
                            flag = True
            info.append(data[i][j])
            info = np.array(info)
            if flag and tuple(info) not in rule:
                cnt += 1
    return cnt
