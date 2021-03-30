# PyGLImER  [![Build Status](https://travis-ci.com/PeterMakus/PyGLImER.svg?token=SDZj9ZcVZQ7jQ6YJZabv&branch=master)](https://travis-ci.com/PeterMakus/PyGLImER) [![Documentation Status](https://readthedocs.org/projects/pyglimer/badge/?version=latest)](https://pyglimer.readthedocs.io/en/latest/?badge=latest)

## A workflow to create a global database for Ps and Sp receiver function imaging of crustal and upper mantle discontinuties 

This project was created in the framework of a Master's thesis by Peter Makus.
It aims to **automate receiver function (RF) processing from download of raw waveform data to common conversion point (CCP) imaging with a minimum amount
of user interference.**
It is largely based on the [ObsPy](https://github.com/obspy/obspy) project and can be seen as a more powerful and user-friendly
successor of the [GLImER](http://stephanerondenay.com/glimer-web.html) project.


## Installation of this package

A few simple steps:

```bash
# Create the conda environment and install dependencies
conda env create -f environment.yml

# Activate the conda environment
conda activate PyGLImER

# Install your package
pip install  $InstallationDirectory$
```

## Getting started
Access PyGLImER's documentation [here](https://pyglimer.readthedocs.io/en/latest/).

PyGLImER comes with a few tutorials (Jupyter notebooks). You can find those in the `examples/` directory.

## Reporting Bugs / Contact the developers
This version is an early release. If you encounter any issues or unexpected behaviour, please open an issue here on GitHub or [contact the developers](mailto:makus@gfz-potsdam.de).

## Citing PyGLImER
If you use PyGLImER to produce content for your publication, please consider citing us. For the time being, please cite our [AGU abstract](https://www.essoar.org/doi/10.1002/essoar.10506417.1).
