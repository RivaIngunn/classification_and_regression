# classification_and_regression

## General info
Repository for second project in FYS-STK 3155.
All methods are found in code/.
Note that the plots in visuals/ may differ from those in report.

## How to run code
* install pipenv on your system
* clone the repository
*  in correct folder, type:
```
install pipenv
```
* enter shell:
```
pipenv shell
```
* run code file as normal

## Table of contents
* [Visuals](code/visuals)
* [Classes](code)
* [Example runs](code/example_runs)
* [Test runs](code/tests)
* [Benchmark results](benchmarks)
* [Report](report)

## Example use
To run SGD, Logistic Regression and Neural Network methods, run example files in example_runs/.

#### stochastic_run.py:
Study hyperparameters and find best set for SGD

#### logistic_run.py:
Find best set of hyperparameters for logistic regression

### neural_network_run:
Find best set of hyperparameters for the Feed Forward Neural Network.
Both for classification and linear regression.

### Test of codes
to test the function write
```
pytest -v
```
or run the program normally. this also serves as benchmarks to check that each part of the code is running as expected.
