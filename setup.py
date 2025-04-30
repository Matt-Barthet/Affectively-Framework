from setuptools import setup, find_packages

setup(
    name='affectively_environments',
    author='Matthew Barthet',
    author_email='matthew.barthet@um.edu.mt',
    description='Affectively Environments',
    long_description=open('README.md').read(),
    version='0.0.1',
    python_requires='==3.9.21',
    install_requires=['gym==0.26.2', 'gym_unity==0.26.0', 'mlagents_envs==0.26.0', 'tensorboardX==2.6.2.2'],
    packages=find_packages(),  # This automatically finds all packages and subpackages
    include_package_data=True,
    package_data={'affectively_environments': ['datasets/*.csv']},
)