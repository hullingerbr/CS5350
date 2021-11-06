This is a machine learning library developed by Brandon Hullinger (u0579548) for CS5350/6350 in University of Utah

--------------------------------
Decision Tree
--------------------------------
To use the decision tree code, first make sure that the Decision Tree file you are using is identical to the one in the repository. Then, simply run DecisionTree.py
The code will output data in the following order in the form
a & b & c & d & e & f & g

a = Depth of tree
b = Error with Information Gain on Training Data
c = Error with Majority Error on Training Data
d = Error with Gini Index on Training Data
e = Error with Information Gain on Test Data
f = Error with Majority Error on Test Data
g = Error with Gini Index on Trest Data

Formatted as such for easy copy-paste into laTeX.

Edit the string titled 'file' on line 11 to either bank or car, depending on which data you want to run on. Possible to use your own data, as long as the formatting is identical to what is found in the bank/car folders.
Edit the integer titled 'maxDepth' to change the maximum depth of the created decision trees.
Uncomment line 263 to print any created Decision Trees.
--------------------------------


--------------------------------
Adaboost
--------------------------------
Not Implemented
--------------------------------


--------------------------------
Linear Regression
--------------------------------
To use the linear regression code, first make sure that the Linear Regression file you are using is identical to the one in the repository. The code for this can be run in one of following ways

From the console in the :
$ python3 "Gradient Descent.py"
$ bash run.sh

If running in one of the above two ways, the code will output the weight vector and bias, cost, and a plot of the costs per epoch for Batch Gradient Descent and Stochastic Gradient Descent. It will then print the calculated optimal weight vector and bias.

$ jupyter notebook "Gradient Descent.ipynb"

This method will require more manual execution of the code, but allows walking through it step by step. Just click in the first cell, then press shift+enter to run each cell in order to get the results. The last 3 cells will print the same results as running the bash script or the python code directly.
--------------------------------
