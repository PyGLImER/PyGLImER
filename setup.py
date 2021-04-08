'''
Author: Peter Makus (makus@gfz-potsdam.de)

Created: Friday, 3rd July 2020 10:46:45 am
Last Modified: Thursday, 25th March 2021 04:20:35 pm

Setup.py file that governs the installation process of
`PyGLImER it is used by
`conda install -f environment.yml` which will install the package in an
environment specified in that file.
'''
from setuptools import setup
from setuptools.command.test import test as testcommand

<<<<<<< HEAD
# Utility function to read the README.md file.
# Used for the long_description.  It's nice, because now 1) we have a top levelx
# README.md file and 2) it's easier to type in the README.md file than to put a raw
# string in below ...

# Function to read and output README into long decription


def read(fname):
    """From Wenjie Lei 2019"""
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except Exception as e:
        return "Can't open %s" % fname


long_description = "%s" % read("README.md")

=======
>>>>>>> 52513906183d5094c448384313dacb8a4608170f

# This installs the pytest command. Meaning that you can simply type pytest
# anywhere and "pytest" will look for all available tests in the current
# directory and subdirectories recursively
class PyTest(testcommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.tests")]

    def initialize_options(self):
        testcommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        import sys
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


<<<<<<< HEAD
setup(
    name="PyGLImER",
    description="A workflow to create a global database for Ps and Sp rece\
    eiver function imaging of crustal and upper mantle discontinuities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="0.0.1",
    author="Peter Makus",
    author_email="peter.makus@student.uib.no",
    license='GNU Lesser General Public License, Version 3',
    keywords="Seismology, Receiver Function, Earth Sciences,\
        Lithosphere, Imaging, converted wave",
    url='https://github.com/PeterMakus/PyGLImER',
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    install_requires=['numpy', 'geographiclib', 'h5py', 'joblib',
                      'matplotlib', 'numpy', 'obspy', 'pandas', 'pathlib',
                      'plotly', 'psutil', 'scipy', 'tqdm', 'flake8'],
    tests_require=['pytest'],
    cmdclass={'tests': PyTest},
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        ("License :: OSI Approved "
         ":: GNU General Public License v3 or later (GPLv3+)"),
    ],
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme"],
        "tests": ["pytest", "py"]
    },
    entry_points={
        'console_scripts': [
            'sample-bin = matpy.bins.sample_bin:main'
        ]
    }
)
=======
setup(cmdclass={'tests': PyTest})
>>>>>>> 52513906183d5094c448384313dacb8a4608170f
