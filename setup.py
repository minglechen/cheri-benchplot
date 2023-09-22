import os
import shutil
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """
    Custom extension for cmake projects
    """
    def __init__(self, name, module_name=None):
        super().__init__(name, sources=[])
        #: The resulting module name where the extension is placed
        self.module_name = module_name or Path(name).name


class CheriBenchplotBuild(build_ext):
    """
    Build C++ extensions with cmake in pycheribenchplot/ext
    """
    description = "Build C++ extensions with cmake in pycheribenchplot/ext"
    user_options = [
        ("build-dir=", None, "Path to build directory, use temporary directory otherwise."),
        ("cherisdk=", None, "Path to the Cheri SDK directory, this must have llvm installed."),
        ("cmake-options=", None, "Extra options to forward to cmake."),
    ]

    def _do_build(self, ext: CMakeExtension):
        """
        Build selected cmake extension.

        There are two cases:
        - During a normal installation, the target directory will be compressed into
        a python wheel archive.
        - During editable installations the target directory is a source tree directory,
        the build is installed in-tree.
        """
        target_dir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        target_dir = target_dir / ext.module_name

        cmake_args = [
            "-G",
            "Ninja",
            "-S",
            str(Path.cwd().absolute() / "pycheribenchplot" / "ext"),
            "-B",
            self.build_temp,
            f"-DCMAKE_INSTALL_PREFIX:PATH={target_dir}",
        ]
        if self.cherisdk:
            cmake_args += [f"-DCHERISDK={self.cherisdk}"]
        cmake_args += self.cmake_options

        self.spawn(["cmake"] + cmake_args)
        self.spawn(["cmake", "--build", self.build_temp])
        self.spawn(["cmake", "--install", self.build_temp])

    def initialize_options(self):
        super().initialize_options()
        # As a workaround for setuptools refusing to pass arbitrary
        # --config-settings down here, just read the value from the environment.
        # If it is not set, complain. One day this will be fixed upstream.
        # The alternative is to have a custom setuptools backend but it is
        # probably not worth it.
        self.cherisdk = os.getenv("CHERISDK")
        self.cmake_options = []

    def finalize_options(self):
        super().finalize_options()
        if self.cherisdk is None:
            raise ValueError(f"Missing CHERISDK env variable, please set CHERISDK=/path/to/cherisdk")
        self.ensure_dirname("cherisdk")
        self.ensure_string_list("cmake_options")

    def run(self):
        """
        Build extensions using cmake, other extensions are rejected.
        """
        if shutil.which("cmake") is None:
            raise RuntimeError("Missing system cmake")
        if shutil.which("ninja") is None:
            raise RuntimeError("Missing system ninja")

        for ext in self.extensions:
            if not isinstance(ext, CMakeExtension):
                raise TypeError("Unsupported extension type")
            self._do_build(ext)


setup(name="pycheribenchplot",
      summary="Task runner and analysis tool for CHERI-related projects",
      license="BSD 2-Clause",
      version="1.2",
      packages=find_packages(),
      ext_modules=[CMakeExtension("pycheribenchplot/ext", module_name="ext")],
      cmdclass={"build_ext": CheriBenchplotBuild},
      scripts=["benchplot-cli.py", "benchplot-gui.py"],
      install_requires=[
          "paramiko>=3.2.0", "marshmallow-dataclass[enum,union]>=8.5.8", "marshmallow-enum>=1.5.1", "isort>=5.10.0",
          "matplotlib>=3.4.3", "numpy>=1.21.3", "networkx>=2.8.5", "openpyxl>=3.0.9", "pandas>=1.3.4",
          "pandera>=0.15.1", "pyelftools>=0.27", "PyPika>=0.48.8", "sortedcontainers>=2.4.0", "seaborn>=0.12",
          "squarify>=0.4.3", "tabulate>=0.9", "termcolor>=1.1.0", "XlsxWriter>=3.0.2", "yapf>=0.31.0",
          "gitpython>=3.1.27", "typing_inspect>=0.5.0", "typing_extensions>=4.7.0", "multi-await>=1.0.0",
          "pyserial>=3.5", "hypothesis>=6.81.0", "cxxfilt>=0.3.0", "pyqt6>=6.5.1"
      ],
      extras_require={"dev": [
          "pytest",
          "pytest-mock",
          "pytest-timeout",
          "sphinx",
      ]})
