rm -rf dist jsonizer.egg-info
python3 setup.py sdist
twine check dist/*
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# ---PRODUCTION---
# twine upload dist/*
# ---PRODUCTION---
