#Gradient Descent
import numpy as np
from scipy import linalg as la
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
import random
from matplotlib import pyplot as plt

trainData = np.genfromtxt('concrete/train.csv', delimiter=',')
xTrain = np.transpose(trainData[:,:-1])
yTrain = trainData[:,-1]

testData = np.genfromtxt('concrete/test.csv', delimiter=',')
xTest= np.transpose(testData[:,:-1])
yTest = testData[:,-1]

#Batch Gradient Descent
r = 0.125
xLen = xTrain.shape[0]
m = xTrain.shape[1]

wCurr = np.zeros((1,xLen))
bCurr = 0

def anyGT(w,b,n):
    if(b > n):
        return True
    if((w[0,:] > n).any()):
        return True
    return False

dw = np.ones((1,xLen))
db = 1
costs = []
cost = 0
for i in range(m):
    x = xTrain[:,i]
    wx = wCurr.dot(x)
    cost += ((bCurr + wx - yTrain[i])**2)/(2*m)
costs.append(cost)
while(anyGT(dw,db,1e-6)):
    #print(dw[0,:],db)
    dw = np.zeros((1,xLen))
    db = 0
    for i in range(m):
        x = xTrain[:,i]
        wx = wCurr.dot(x)
        dw += (bCurr+wx-yTrain[i]) * x
        db += (bCurr+wx-yTrain[i])
    wCurr = wCurr - r*dw/m
    bCurr = bCurr - r*db/m
    cost = 0
    for i in range(m):
        x = xTrain[:,i]
        wx = wCurr.dot(x)
        cost += ((bCurr + wx - yTrain[i])**2)/(2*m)
    costs.append(cost)
    dw = abs(r*dw/m)
    db = abs(r*db/m)
    
    
print("Batch Gradient Descent")
print("w=",wCurr,"b=",bCurr)
m = xTest.shape[0]
cost = 0
for i in range(m):
    x = xTest[:,i]
    wx = wCurr.dot(x)
    cost += ((bCurr + wx - yTest[i])**2)/(2*m)
print("Cost=",cost)
plt.plot(costs)

#Stochastic Gradient Descent
random.seed()
r = 0.125
xLen = xTrain.shape[0]
m = xTrain.shape[1]

wCurr = np.zeros((1,xLen))
bCurr = 0

def anyGT(w,b,n):
    if(b > n):
        return True
    if((w > n).any()):
        return True
    return False

costs = []
for i in range(m):
    x = xTrain[:,i]
    wx = wCurr.dot(x)
    cost += ((bCurr + wx - yTrain[i])**2)/(2*m)
costs.append(cost)
dw = np.ones((1,xLen))
db = 1
while(anyGT(dw,db,1e-7)):
    #print(dw,db)
    i = random.randrange(m)
    x = xTrain[:,i]
    wx = wCurr.dot(x)
    dw = (bCurr+wx-yTrain[i]) * x/m
    db = (bCurr+wx-yTrain[i])/m
    wCurr = wCurr - r*dw
    bCurr = bCurr - r*db
    cost = 0
    for i in range(m):
        x = xTrain[:,i]
        wx = wCurr.dot(x)
        cost += ((bCurr + wx - yTrain[i])**2)/(2*m)
    costs.append(cost)
    dw = abs(r*dw)
    db = abs(r*db)
    
print("Stochastic Gradient Descent")
print("w=",wCurr,"b=",bCurr)
m = xTest.shape[0]
cost = 0
for i in range(m):
    x = xTest[:,i]
    wx = wCurr.dot(x)
    cost += ((bCurr + wx - yTest[i])**2)/(2*m)
print("Cost=",cost)
plt.plot(costs)

#Calculate Optimal Solution
m = xTrain.shape[1]
b = np.ones(m)
newX = np.transpose(np.insert(np.transpose(xTrain), 0, b, axis=1))
XTX = np.matmul(newX,np.transpose(newX))
XTXI = la.inv(XTX)
XTXIX = np.matmul(XTXI,newX)
W = np.matmul(XTXIX,yTrain)
print("Optimal")
print("w=",W[1:],"b=",W[0])
