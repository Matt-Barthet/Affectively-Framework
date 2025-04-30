import subprocess
import sys
import os


if __name__ == "__main__":
    runs = range(4,7)
    weights = [0]
    environments = ["pirates"]

    script_path = 'Train_DQN.py'
    cwd = "./Agents/DQN/"
    parent = os.getcwd()

    for run in runs:
        for weight in weights:
            for environment in environments:
                command = f'cd {parent}/../.. && conda activate unity_gym && python {cwd}{script_path} {run} {weight} {environment}'
                subprocess.run(['wt', '-p', 'Command Prompt', 'cmd', '/c', command], shell=True)
