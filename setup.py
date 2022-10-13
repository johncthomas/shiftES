from setuptools import setup

setup(
    name = 'shiftES',
    author = 'John C. Thomas',
    author_email = 'jcthomas000@gmail.com',
    version='0.1',
    install_requires = ['numpy', 'pandas'],
    scripts = ['shiftes/calculate_effectsize.py'],
    python_requires = '>=3.6',
    include_package_data=True, #uses MANIFEST.in
)
