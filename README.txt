CS578 - Statistical Machine Learning
Homework 3

By: Andres Bejarano <abejara@cs.purdue.edu>


For running the program type the following command in console:

	python template.py


By default, if no arguments are passed then GD is selected with:

- MaxIterations = 1000
- Regularization: L1
- Step size: 0.1
- Lambda: 0.1
- Feature Set: Unigrams


IMPORTANT: It is required to have the three data set files (training, validating and testing) in the /data folder of the python file and their names should also be train.csv, validation.csv and test.csv. Additionaly, a file of neutral words (neutral.csv) is also included in the /data folder. It is required for building the internal dictionary for training purposes.

Additional output for each experiment can be found in experiment1.txt, experiment2.txt, experiment3.txt, experiment4.txt, experiment5.txt and experiment6.txt

Note: Training time may take a while (around 5 minutes for both unigrams and bigrams feature set).
