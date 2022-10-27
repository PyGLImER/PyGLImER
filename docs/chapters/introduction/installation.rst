
Installation
------------

From PyPi
=========
We recommend installing ``PyGLImER`` from PyPi.

.. code:: bash

    pip install pyglimer
    # if you want to use mpi
    pip install mpi4py


Installation from source code
=============================

This will mostly make sense for developers. Note that you should download
the ``dev`` branch if you should wish to make contributions.

.. code:: bash

    # Get the source code
    # Download via wget or web-browser
    wget https://github.com/PyGLImER/PyGLImER/archive/refs/heads/master.zip

    # For developers use
    #wget https://github.com/PyGLImER/PyGLImER/archive/refs/heads/dev.zip

    # unzip the package
    unzip master.zip  # or dev.zip, depending on branch
    # devlopers use dev.zip

    # Change directory to the same directory that this repo is in (i.e., same directory as setup.py)
    cd PyGLImER-master  # That's the standard name the folder should have

    # Create the conda environment and install dependencies
    conda env create -f environment.yml

    # Activate the conda environment
    conda activate pyglimer

    # Install your package, -e ensure that the package will be updated as you edit modules
    pip install -e .
