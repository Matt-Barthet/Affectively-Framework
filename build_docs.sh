cd docs
make clean
sphinx-apidoc -o source/ ../affectively_environments
make html
cd ..
