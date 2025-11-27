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

def find_conda_sh():
    """Locate conda.sh dynamically from the conda binary."""
    try:
        # Find where the `conda` executable lives
        conda_path = subprocess.check_output(["which", "conda"], text=True).strip()
        if not os.path.exists(conda_path):
            return None

        # Go two levels up and look for conda.sh
        base_dir = os.path.abspath(os.path.join(os.path.dirname(conda_path), ".."))
        candidate = os.path.join(base_dir, "etc", "profile.d", "conda.sh")

        if os.path.exists(candidate):
            return candidate

        # As a fallback, try searching the base dir
        for root, dirs, files in os.walk(base_dir):
            if "conda.sh" in files:
                return os.path.join(root, "conda.sh")

    except Exception as e:
        print(f"Error locating conda.sh: {e}")
        return None

    return None

def get_conda_activation_command(conda_env):
    system = platform.system()

    if system in ["Linux", "Darwin"]:  # macOS = "Darwin"
        try:
            # Find conda binary
            conda_path = subprocess.check_output(["which", "conda"], text=True).strip()
            base_dir = os.path.abspath(os.path.join(os.path.dirname(conda_path), ".."))
            conda_sh = os.path.join(base_dir, "etc", "profile.d", "conda.sh")

            if not os.path.exists(conda_sh):
                # fallback: search for conda.sh
                for root, _, files in os.walk(base_dir):
                    if "conda.sh" in files:
                        conda_sh = os.path.join(root, "conda.sh")
                        break

            if not os.path.exists(conda_sh):
                print("❌ Could not find conda.sh; please verify your Conda installation.")
                sys.exit(1)

            return f"source {conda_sh} && conda activate {conda_env}"

        except subprocess.CalledProcessError:
            print("❌ Conda not found on PATH.")
            sys.exit(1)

    elif system == "Windows":
        # On Windows, conda.bat handles activation
        conda_exe = subprocess.check_output(["where", "conda"], text=True).splitlines()[0]
        conda_bat = os.path.join(os.path.dirname(conda_exe), "conda.bat")
        if not os.path.exists(conda_bat):
            print("❌ Could not find conda.bat; please check your Conda installation.")
            sys.exit(1)
        return f'call "{conda_bat}" activate {conda_env}'

    else:
        print(f"Unsupported platform: {system}")
        sys.exit(1)

# --- Main script logic ---
def main():
    """
    Main function to load configurations from a directory and run experiments.
    """
    # Define the directory containing the configuration files
    configs_directory = "configs"

    conda_sh = find_conda_sh()
    if not conda_sh and platform.system() != "Windows":
        print("❌ Could not find conda.sh. Please check your conda installation.")
        sys.exit(1)

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
    for config_file_name in config_files:
        config_file_path = os.path.join(configs_directory, config_file_name)
        config = load_config(config_file_path)

        if not config:
            continue  # Skip to the next file if loading failed

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
        decision_period = config.get("decisionPeriod")
        
        linux_only = f"source {conda_sh}" if system == "darwin" else ""
        # Construct the command using the parameters from the current file
        command = (
            f"cd {cwd} && {linux_only}"
            f"conda activate {conda_env} && "
            f"python {script_path} --run={runs} --use_gpu={use_gpu} --weight={weight} "
            f"--cluster={cluster} --target_arousal={target_arousal} --preference={preference} "
            f"--classifier={classifier} --game={game} --periodic_ra={period_ra} --cv={cv} "
            f"--headless={headless} --discretize={discretize if cv == 0 else 0} "
            f"--grayscale={grayscale if cv == 1 else 0} --logdir={output_dir} "
            f"--algorithm={algorithm} --policy={policy} --decision_period={decision_period}"
        )

        # Execute the command based on the operating system
        print(f"Starting run from config file: {config_file_name}")
        if system == "Linux":
            terminals = ["konsole", "xfce4-terminal", "x-terminal-emulator", "lxterminal",
                         "mate-terminal", "gnome-terminal", ]
            terminal = next((t for t in terminals if shutil.which(t)), None)


            if terminal:
                # Some terminals (like GNOME Terminal) use different argument styles
                if terminal == "gnome-terminal":
                    subprocess.Popen([
                        terminal,
                        "--",  # separates gnome-terminal args from the command
                        "bash", "-c", f"source {command}; exec bash"
                    ])
                elif terminal == "konsole":
                    subprocess.Popen([
                        terminal,
                        "-e", f"bash -c 'source {command}; exec bash'"
                    ])
                else:
                    subprocess.Popen([
                        terminal,
                        "-e", f"bash -c 'source {command}; exec bash'"
                    ])
            else:
                print("No compatible terminal emulator found.")

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

        print(f"Finished processing {config_file_name}. Waiting 10 seconds before next run.")
        time.sleep(10)


if __name__ == "__main__":
    main()
