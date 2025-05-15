import os
import shutil
import subprocess
import itertools
import time
import platform

# Define parameters
runs = [2]  
weights = [0]  
clusters = [0] 
targetArousals = [1]
period_ra = 0
game = "platform"

cwd = os.getcwd()
script_path = "./Main.py"
conda_env = "affect-envs"
system = platform.system()
cv = 0

for run, weight, cluster, targetArousal in itertools.product(runs, weights, clusters, targetArousals):
    command = (
        f"cd {cwd} && conda activate {conda_env} && "
        f"python {script_path} --run={run} --weight={weight} --cluster={cluster} --target_arousal={targetArousal} --game={game} --periodic_ra={period_ra} --cv={cv}"
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
        subprocess.Popen([
            "wt", "new-tab", "cmd.exe", "/K",
            f'call {command}'
        ])
    elif system == "Darwin":  # macOS
        subprocess.Popen(
            ["osascript", "-e", f'tell app "Terminal" to do script "{command}"']
        )
    time.sleep(10)