import os
import subprocess
import time
import platform
import yaml
import sys
import shlex


def load_config(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def find_conda_sh():
    try:
        conda_path = subprocess.check_output(["which", "conda"], text=True).strip()
        base = os.path.abspath(os.path.join(os.path.dirname(conda_path), ".."))
        candidate = os.path.join(base, "etc", "profile.d", "conda.sh")
        if os.path.exists(candidate):
            return candidate
        for root, _, files in os.walk(base):
            if "conda.sh" in files:
                return os.path.join(root, "conda.sh")
    except Exception:
        pass
    return None


def main():
    configs_dir = "configs"
    script_path = "./train.py"
    output_dir = "./results"
    cwd = os.getcwd()
    system = platform.system()

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(configs_dir):
        sys.exit(f"Config directory not found: {configs_dir}")

    configs = [f for f in os.listdir(configs_dir) if f.endswith((".yml", ".yaml"))]
    if not configs:
        sys.exit("No config files found")

    conda_sh = None
    if system in ("Darwin", "Linux"):
        conda_sh = find_conda_sh()
        if not conda_sh:
            sys.exit("conda.sh not found")

    quoted_cwd = shlex.quote(cwd)
    quoted_script = shlex.quote(script_path)
    quoted_conda_sh = shlex.quote(conda_sh) if conda_sh else ""

    for cfg in configs:
        c = load_config(os.path.join(configs_dir, cfg))
        if not c:
            continue

        train_cmd = (
            f"python {quoted_script} "
            f"--run={c.get('runs')} "
            f"--use_gpu={c.get('use_gpu')} "
            f"--weight={c.get('weight')} "
            f"--cluster={c.get('cluster')} "
            f"--target_arousal={c.get('target_arousal')} "
            f"--preference={c.get('preference')} "
            f"--classifier={c.get('classifier')} "
            f"--game={c.get('game')} "
            f"--periodic_ra={c.get('periodic_ra')} "
            f"--cv={c.get('cv')} "
            f"--headless={c.get('headless')} "
            f"--discretize={c.get('discretize') if c.get('cv') == 0 else 0} "
            f"--grayscale={c.get('grayscale') if c.get('cv') == 1 else 0} "
            f"--logdir={shlex.quote(output_dir)} "
            f"--algorithm={c.get('algorithm')} "
            f"--policy={c.get('policy')} "
            f"--decision_period={c.get('decisionPeriod')}"
        )

        if system == "Darwin":
            shell_cmd = (
                f"cd {quoted_cwd} && "
                f"source {quoted_conda_sh} && "
                f"conda activate {c.get('conda_env')} && "
                f"{train_cmd}"
            )
            apple_cmd = shell_cmd.replace('"', '\\"')
            subprocess.Popen([
                "osascript",
                "-e",
                f'tell application "Terminal" to do script "{apple_cmd}"'
            ])

        elif system == "Linux":
            shell_cmd = (
                f"cd {quoted_cwd} && "
                f"source {quoted_conda_sh} && "
                f"conda activate {c.get('conda_env')} && "
                f"{train_cmd}"
            )
            subprocess.Popen(["bash", "-c", shell_cmd])

        elif system == "Windows":
            drive, path = os.path.splitdrive(cwd)
            cmd = (
                f"{drive} && cd {path} && "
                f"call conda activate {c.get('conda_env')} && "
                f"{train_cmd}"
            )
            subprocess.Popen(["wt", "new-tab", "cmd.exe", "/K", cmd])

        time.sleep(10)


if __name__ == "__main__":
    main()
