import numpy as np
import pandas as pd
eps = np.finfo(float).eps
import pprint
from scipy import stats
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_bool_dtype

#Parameters that change.
file = "bank" #The training and test data file.
maxDepth = 5 #The maximum depth of the tree.

def getBins(data,cat):
    #Legacy code. Too lazy to remove right now.
    column = data[cat]
    values = data[cat].unique()
    #print(column)
    return column,values
    
def findEntropy(data):
    #Used to calculate Entropy of Target Category
    target = data.keys()[-1]
    column,values = getBins(data,target)
    entropy = 0
    for value in values:
            frac = column.value_counts()[value]/len(column)
            entropy += -frac*np.log2(frac+eps)
    return entropy

def findEntropyCat(data,cat):
    #Used to calculate fractional entropy of a category
    target = data.keys()[-1]
    tColumn,targetVals = getBins(data,target)
    column,values = getBins(data,cat)
    entropy2 = 0
    for value in values:
        entropy = 0
        for targetVal in targetVals:
            num = len(column[column==value][tColumn==targetVal])
            den = len(column[column==value])
            frac = num/(den+eps)
            entropy += -frac*np.log2(frac+eps)
        frac2 = den/len(data)
        entropy2 += -frac2*entropy
    #print(entropy2)
    return abs(entropy2)
    
def findME(data):
    #Same as findEntropy, but using Majority Error
    target = data.keys()[-1]
    column,values = getBins(data,target)
    ME = 0
    values = data[target].unique()
    for value in values:
        frac = column.value_counts()[value]/len(column)
        ME = max(frac,ME)
    return 1-ME

def findMECat(data,cat):
    #Same as findEntropyCat, but using Majority Error
    target = data.keys()[-1]
    tColumn,targetVals = getBins(data,target)
    column,values = getBins(data,cat)
    ME2 = 0
    for value in values:
        ME = 0
        for targetVal in targetVals:
            num = len(column[column==value][tColumn==targetVal])
            den = len(column[column==value])
            frac = num/(den+eps)
            ME = max(ME,frac)
        frac2 = den/len(data)
        ME2 += frac2*ME
    return abs(1-ME2)

def findGI(data):
    #Same as findEntropy, but using gini index
    target = data.keys()[-1]
    column,values = getBins(data,target)
    GI = 0
    for value in values:
        frac = column.value_counts()[value]/len(column)
        GI += frac**2
    return 1-GI

def findGICat(data,cat):
    #Same as findEntropyCat, but using gini index
    target = data.keys()[-1]
    tColumn,targetVals = getBins(data,target)
    column,values = getBins(data,cat)
    GI2 = 0
    for value in values:
        GI = 0
        for targetVal in targetVals:
            num = len(column[column==value][tColumn==targetVal])
            den = len(column[column==value])
            frac = num/(den+eps)
            GI += frac**2
        frac2 = den/len(data)
        GI2 += frac2*GI
    return abs(1-GI2)
    
def winner(data,func1,func2):
    #Calculates information gain of all categories, then returns category with largest gain
    #Also prints Errors and Gains calculated
    gains = []
    ents = []
    for value in data.keys()[:-1]:
        dEnt = func1(data)
        catEnt = func2(data,value)
        gains.append(dEnt-catEnt)
        ents.append(catEnt)
        #print(dEnt,catEnt)
    ents.append(dEnt)
    #print("Errors:",ents)
    #print("Gains:",gains)
    return data.keys()[:-1][np.argmax(gains)]

def getSubtable(data, node, value):
    #Used to split the table down to smaller subsets
    return data[data[node] == value].reset_index(drop=True)

def buildTree(data, func1, func2, depth = 100000, d = 1, tree=None):
    #Builds a decision tree.
    #func 1 and func2 are the error calc methods.
    target = data.keys()[-1]
    node = winner(data,func1,func2)
    catValue = np.unique(data[node])  
    if tree is None:                    
        tree={}
        tree[node] = {}
    for value in catValue:
        subtable = getSubtable(data,node,value)
        #print(subtable)
        clValue,counts = np.unique(subtable[target],return_counts=True)
        if len(counts)==1 or d == depth:
            tree[node][value] = subtable[target].mode()[0]                               
        else:        
            tree[node][value] = buildTree(subtable,func1,func2,depth,d+1)   
    return tree

def predict(instance, tree):
    #Given an input, traverses a decision tree and returns the result.
    #Returns "Unable to Traverse Tree With Given Input" if the input cannot follow the tree
    if(not(type(tree) is dict)):
        #print(tree)
        return tree
    for nodes in tree.keys():

        value = instance[nodes]
        #print(value)
        try:
            nextNode = tree[nodes][value]
        except KeyError:
            return "Unable to Traverse Tree With Given Input"
        prediction = 0

        if type(tree) is dict:
            prediction = predict(instance, nextNode)
        else:
            prediction = tree
            break;
    return prediction
    
def genNumBuckets(data,cats):
    #Used to get the buckets for numerical data
    buckets = []
    for cat in cats:
        bucket = []
        if(not(is_string_dtype(data[cat]))):
            median = data[cat].median()-eps
            bucket = [float('-inf'),median,float('inf')]
        buckets.append(bucket)
    return buckets

def cleanNumeric(data,cats,buckets):
    #Replaces numerical data with given bucket categories
    for i in range(len(cats)):
        cat = cats[i]
        if(not(is_string_dtype(data[cat]))):
            data[cat] = pd.cut(data[cat],bins=buckets[i],include_lowest=True,duplicates='drop')
    return data
    
def replaceUnkMC(data,cat,unknown):
    #Replaces an unknown with the most common value
    MC = data[cat].mode()[0]
    data[cat] = data[cat].replace(unknown,MC)
    return data

def replaceUnkMM(data,cat,unknown):
    #Replaces an unknown with the value which most commonly matches the output
    subset = data[data[cat] != unknown]
    MM = data[cat].mode()[0]
    data[cat] = data[cat].replace(unknown,MM)
    return data

def replaceUnkFrac(data,cat,unknown):
    #Changes data table to use fractional counts. Does this by expanding size of table.
    subTable = data[data[cat] == unknown]
    #print(subTable)
    data = data[data[cat] != unknown]
    #print(data)
    values = data[cat].unique()
    valCounts = []
    data2 = pd.concat([data]*len(data), ignore_index=True)
    for value in values:
        valCount = data[cat].value_counts()[value]
        #print(valCount)
        subT2 = subTable
        subT2[cat] = value
        subT2 = pd.concat([subT2]*valCount, ignore_index=True)
        #print(subT2)
        data2 = pd.concat([data2,subT2], ignore_index=True)
    return data2
    
def main():
    with open(file+'/categories.txt') as f:
        categories = f.readlines()
    categories = categories[0].strip().replace(' ','').split(',')
    tData = pd.read_csv(file+'/train.csv', names = categories)
    numCats = len(categories)
    buckets = genNumBuckets(tData,categories)
    tData = cleanNumeric(tData,categories,buckets)
    
    testData = pd.read_csv(file+'/test.csv',names=categories)
    testData = cleanNumeric(testData,categories,buckets)
    
    #Replacing Unknown Data. Comment out if not needed.
    for cat in categories:
        testData = replaceUnkMC(testData,cat,"unknown")
    #print(testData)
    
    #Generate trees using training data, then test on training and test data. 
    #Print fraction of erros.
    #Comment out all print statements in winner function before running.
    for i in range(maxDepth):
        TE = []
        TestE = []
        TargetCalcs = [findEntropy,findME,findGI]
        CatCalcs = [findEntropyCat,findMECat,findGICat]
        for j in range(3):
            tree = buildTree(tData,TargetCalcs[j],CatCalcs[j],i+1)
            target = categories[len(categories)-1]
            num = 0
            den = len(tData)
            for k in range(den):
                goal = tData[target].iloc[k]
                prediction = predict(tData.iloc[k],tree)
                #print(prediction,"|",goal)
                if(prediction == goal):
                    num += 1
            TE.append(round(1-num/den,3))
            num = 0
            den = len(testData)
            for k in range(den):
                goal = testData[target].iloc[k]
                prediction = predict(testData.iloc[k],tree)
                #print(prediction,"|",goal)
                if(prediction == goal):
                    num += 1
            TestE.append(round(1-num/den,3))
        #pprint.pprint(tree)
        print(i+1,"&",TE[0],"&",TE[1],"&",TE[2],"&",TestE[0],"&",TestE[1],"&",TestE[0],"\\\\")
    
if __name__ == "__main__":
    main()
