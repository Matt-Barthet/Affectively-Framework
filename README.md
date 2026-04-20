# Affectively Framework

## Building the project from source
Clone the repository to your local storage. `Builds` are available separately, download them [here](https://drive.google.com/file/d/1hoNjlVUj9Yh7vacSjnwM7_iaXFChlK1d/view?usp=sharing).

Create conda environment
```bash
conda create -n affect-envs python==3.9
```
Activate the environment
```bash
conda activate affect-envs
```
Downgrade `pip` and `setuptools`:
```bash
python -m pip install setuptools==69.5.1 pip==24.0
```
Install dependencies
```bash
pip install -r requirements.txt
```

If you have a CUDA-capable GPU, the requirements file now installs the CUDA-enabled PyTorch wheel (`torch==2.8.0+cu118`) using PyTorch's extra index URL.

If on MacOS (Tested on an Apple Silicon machine)
```
pip install stable_baselines3==1.8.0 sb3_contrib==1.8.0 --no-deps
```
