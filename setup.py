from setuptools import setup

setup(
    name = 'shiftes',
    author = 'John C. Thomas',
    author_email = 'jcthomas000@gmail.com',
    version='0.1',
    install_requires = ['numpy', 'pandas'],
    scripts = ['shiftes/shiftes.py'],
    python_requires = '>=3.7',
    include_package_data=True,
)

