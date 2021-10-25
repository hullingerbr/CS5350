This is a machine learning library developed by Brandon Hullinger (u0579548) for CS5350/6350 in University of Utah

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