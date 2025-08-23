import os
import shutil
import subprocess
import itertools
import time
import platform

# Define parameters
runs = 5  
weights = [0.5]  
clusters = [0] 
targetArousals = [1]
period_ra = 0
headless = 1
output_dir = "./results/"
grayscale = 0
discretize = 0
use_gpu = 0

preferences = [0, 1]
classifiers = [0, 1]

game = "solid"
algorithm = "PPO"
policy="MlpPolicy"

cwd = os.getcwd()
script_path = "./train.py"
conda_env = "affect-envs"
system = platform.system()
cv = 0

for weight, cluster, targetArousal, classifier, preference in itertools.product(weights, clusters, targetArousals, classifiers, preferences):
    command = (
        f"cd {cwd} && conda activate {conda_env} && "
        f"python {script_path} --run={runs} --use_gpu={use_gpu} --weight={weight} --cluster={cluster} --target_arousal={targetArousal} --preference={preference} --classifier={classifier} --game={game} --periodic_ra={period_ra} --cv={cv} --headless={headless} --discretize={discretize if cv == 0 else 0} --grayscale={grayscale if cv == 1 else 0} --logdir={output_dir} --algorithm={algorithm} --policy={policy}"
    )

    if system == "Linux":
        terminals = ["gnome-terminal", "konsole", "xfce4-terminal", "x-terminal-emulator", "lxterminal",
                     "mate-terminal"]
        terminal = next((t for t in terminals if shutil.which(t)), None)
        subprocess.Popen([
            terminal,
            "--",
            "bash", "-c", f"source ~/miniconda3/bin/activate && {command}; exec bash"
        ])
    elif system == "Windows":
        cwd = os.getcwd()
        drive, path = os.path.splitdrive(cwd)

        subprocess.Popen([
            "wt", "new-tab", "cmd.exe", "/K",
            f"{drive} && cd {path} && call {command}"
        ])
    elif system == "Darwin":  # macOS
        subprocess.Popen(
            ["osascript", "-e", f'tell app "Terminal" to do script "{command}"']
        )
    time.sleep(10)