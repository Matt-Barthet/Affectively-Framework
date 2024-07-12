# Affectively Environments

# Building the project from source
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

# Documentation
A "nice" documentation of the project is available in `docs\build\html\index.html`.

You can rebuild the documentation by first making sure you have `sphinx` installed:
```bash
pip install -r docs\requirements.txt
```
Then simply run
```bash
sh build_docs.sh
```
to update the `index.html` page.

# Check out examples
_TODO :)_