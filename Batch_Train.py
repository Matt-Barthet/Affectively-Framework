import os
import shutil
import subprocess
import itertools
import time
import platform

# Define parameters
runs = [1,2,3]  # Example run IDs
weights = [0]  # Example weight values
clusters = [0]  # Cluster indices
targetArousals = [1]
period_ra = 0
game = "fps"

cwd = os.getcwd()
script_path = "./Main.py"
conda_env = "affect-envs"
system = platform.system()

# Iterate over all combinations of parameters
for run, weight, cluster, targetArousal in itertools.product(runs, weights, clusters, targetArousals):
    command = (
        f"cd {cwd} && conda activate {conda_env} && "
        f"python {script_path} --run={run} --weight={weight} --cluster={cluster} --target_arousal={targetArousal} --game={game} --periodic_ra={period_ra}"
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
        # Use Windows Terminal (wt) and open a new tab
        subprocess.Popen([
            "wt", "new-tab", "cmd.exe", "/K",
            f'call {command}'
        ])
    elif system == "Darwin":  # macOS
        # Use Terminal on macOS
        subprocess.Popen(
            ["osascript", "-e", f'tell app "Terminal" to do script "{command}"']
        )
    time.sleep(10)