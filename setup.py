from setuptools import find_packages, setup

setup(name="pycheribenchplot",
      version="1.2",
      packages=find_packages(),
      scripts=["benchplot-cli.py"],
      install_requires=[
          "paramiko>=3.2.0",
          "marshmallow-dataclass[enum,union]>=8.5.8",
          "marshmallow-enum>=1.5.1",
          "isort>=5.10.0",
          "Jinja2>=3.0.2",
          "matplotlib>=3.4.3",
          "numpy>=1.21.3",
          "networkx>=2.8.5",
          "openpyxl>=3.0.9",
          "pandas>=1.3.4",
          "pandera>=0.15.1",
          "pyelftools>=0.27",
          "PyPika>=0.48.8",
          "sortedcontainers>=2.4.0",
          "seaborn>=0.12",
          "squarify>=0.4.3",
          "tabulate>=0.9",
          "termcolor>=1.1.0",
          "XlsxWriter>=3.0.2",
          "yapf>=0.31.0",
          "gitpython>=3.1.27",
          "typing_inspect>=0.5.0",
          "multi-await>=1.0.0",
          "pyserial>=3.5",
      ])
