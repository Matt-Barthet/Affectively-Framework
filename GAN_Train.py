import os
import shutil
import subprocess
import time
import platform
import yaml
import sys


# --- Helper function to load the configuration file ---
def load_config(config_file_path):
    """
    Loads parameters from a YAML configuration file.

    Args:
        config_file_path (str): The full path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the loaded configuration parameters.
    """
    try:
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file '{config_file_path}': {exc}")
        return None


# --- Main script logic ---
def main():
    """
    Main function to load configurations from a directory and run experiments.
    """
    # Define the directory containing the configuration files
    configs_directory = "configs"

    # Get current working directory and system info
    cwd = os.getcwd()
    script_path = "./train.py"
    system = platform.system()

    # Create the output directory if it doesn't exist
    output_dir = "./results/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if the configs directory exists
    if not os.path.isdir(configs_directory):
        print(f"Error: The configuration directory '{configs_directory}' does not exist.")
        sys.exit(1)

    # Find all YAML files in the directory
    config_files = [
        f for f in os.listdir(configs_directory)
        if f.endswith(('.yaml', '.yml'))
    ]

    if not config_files:
        print(f"Warning: No YAML configuration files found in '{configs_directory}'.")
        return

    print(f"Found {len(config_files)} configuration files. Starting experiments...")

    # Iterate through each configuration file and run an experiment
    config_file_path = os.path.join(configs_directory, config_files[0])
    config = load_config(config_file_path)

    # Extract parameters from the loaded config
    runs = config.get("runs")
    game = config.get("game")
    algorithm = config.get("algorithm")
    policy = config.get("policy")
    conda_env = config.get("conda_env")
    use_gpu = config.get("use_gpu")
    headless = config.get("headless")

    weight = config.get("weight")
    cluster = config.get("cluster")
    target_arousal = config.get("target_arousal")
    classifier = config.get("classifier")
    preference = config.get("preference")

    period_ra = config.get("periodic_ra")
    cv = config.get("cv")
    grayscale = config.get("grayscale")
    discretize = config.get("discretize")

    # Construct the command using the parameters from the current file
    command = (
        f"cd {cwd} && conda activate {conda_env} && "
        f"python {script_path} --run={runs} --use_gpu={use_gpu} --weight={weight} "
        f"--cluster={cluster} --target_arousal={target_arousal} --preference={preference} "
        f"--classifier={classifier} --game={game} --periodic_ra={period_ra} --cv={cv} "
        f"--headless={headless} --discretize={discretize if cv == 0 else 0} "
        f"--grayscale={grayscale if cv == 1 else 0} --logdir={output_dir} "
        f"--algorithm={algorithm} --policy={policy}"
    )

    # Execute the command based on the operating system
    print(f"Starting run from config file: {config_files[0]}")
    if system == "Linux":
        terminals = ["gnome-terminal", "konsole", "xfce4-terminal", "x-terminal-emulator", "lxterminal",
                        "mate-terminal"]
        terminal = next((t for t in terminals if shutil.which(t)), None)
        if terminal:
            subprocess.Popen([
                terminal,
                "--",
                "bash", "-c", f"source ~/miniconda3/bin/activate && {command}; exec bash"
            ])
        else:
            print("Warning: No supported terminal found. Running command in the current terminal.")
            subprocess.Popen(["bash", "-c", f"source ~/miniconda3/bin/activate && {command}"])

    elif system == "Windows":
        drive, path = os.path.splitdrive(cwd)
        subprocess.Popen([
            "wt", "new-tab", "cmd.exe", "/K",
            f"{drive} && cd {path} && call {command}"
        ])
    elif system == "Darwin":  # macOS
        subprocess.Popen(
            ["osascript", "-e", f'tell app "Terminal" to do script "{command}"']
        )

    print(f"Finished processing {config_files[0]}. Waiting 10 seconds before next run.")


if __name__ == "__main__":
    main()
