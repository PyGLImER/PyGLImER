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


setup(cmdclass={'tests': PyTest})
