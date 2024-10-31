# pybounds

Python implementation of BOUNDS: Bounding Observability for Uncertain Nonlinear Dynamic Systems.

<p align="center">
  <a href="https://pynumdiff.readthedocs.io/en/master/" target="_blank" >

[//]: # (    <img alt="Python for Numerical Differentiation of noisy time series data" src="docs/source/_static/logo_PyNumDiff.png" width="300" height="200" />)
  </a>
</p>

<p align="center">
    <a href="https://pypi.org/project/pybounds/">
        <img src="https://badge.fury.io/py/pynumdiff.svg" alt="PyPI version" height="18"></a>
</p>

## Introduction

This repository provides a minimal working example demonstrating how to empirically calculate the observability level of individual states for a nonlinear (partially observable) system, and accounts for sensor noise.

## Installing

The package can be installed by cloning the repo and running python setup.py install from inside the home pybounds directory.

Alternatively using pip
```bash
pip install pybounds
```

## Notebook examples
There is currently one simple example notebook. More to come.
*  Monocular camera with optic fow measurements: [mono_camera_example.ipynb](examples%2Fmono_camera_example.ipynb)

## Citation

If you use the code or methods from this package, please cite the following paper:

Benjamin Cellini, Burak Boyacıoğlu, Stanley David Stupski, and Floris van Breugel. Discovering and exploiting active sensing motifs for estimation with empirical observability. (2024) bioRxiv.

## Related packages
This repository is the evolution of the EISO repo (https://github.com/BenCellini/EISO), and is intended as a companion to the repository directly associated with the paper above.

## License

This project utilizes the [MIT LICENSE](LICENSE.txt).
100% open-source, feel free to utilize the code however you like. 
