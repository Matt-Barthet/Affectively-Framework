import sys,os
sys.path.append(os.path.dirname(sys.path[0]))
from utils.level_process import *

# import keyboard

def save_level_as_text(level, name):
    with open(name+".txt", 'w') as f:
        f.write(arr_to_str(level))
    # keyboard.wait("space")

def save_level_as_num_text(level, name):
    with open(name+".txt", 'w') as f:
        f.write(arr_to_num_str(level))
    # keyboard.wait("space")
