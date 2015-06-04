Authors: Yoshi Hemzal, Gareth McFarlane

COMP3308 Assignment 2
=====================

This is a Python2 implementation of the Cloudy Day Bayesian Network for COMP3308 (Assignment 2, part 1).

The program is not intrinsically restricted to *just* the cloudyday network, as there is a general Node and
Network class built. However, there are no tests to ensure the Network is correctly initialised, so it may not
work with general implementations. As is, however, it works fine. Mean and variance of the posterior estimate
are calculated based on the number of samples and how many times the calculation is repeated.

Requirements
------------
* Python 2.7
* Numpy
* Scipy

Usage
-----
* Ensure bayesnet.py is executable (chmod +x on Unix systems)
* Run ./bayesnet.py &lt; number of samples &rt; &lt; number of runs &rt;
