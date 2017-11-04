##################################################
#                                                #
#                README - test.py                #
#                                                #
##################################################


HOW TO USE test.py

* SETUPS
Required files :
- test.py (obviously)
- train-data.csv : the train data from Kaggle
- test-data.csv : test data from Kaggle
- eval.csv : expected results for test-data's predictions

Generated file :
- output.csv : predictions obtained with our model for test-data

Configure the script's paths PATH_TRAIN, PATH_TEST, OUTPUT and EVAL_PATH with your corresponding paths on your computer.

* BASIC TEST
Directly launch the script. Make sure the paths are correct before running the following command line :

    > python test.py

* ADD PARAMETERS
Open Terminal, go to the script's directory and write the following command line :

    > python test.py absolute/path/to/train-data.csv absolute/path/to/test-data.csv absolute/path/to/output.csv

TODO : Pass a paramgrid dictionary as an argument. Still have some difficulties to parse the argument