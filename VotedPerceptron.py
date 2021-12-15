#Perceptron algorithm using logic gates
#(2-bit binary input)

import numpy as np
  
#define our unit step function
#necessary for perceptrons of all types
def unitStep(v):
    if v >= 0:
        return 1
    else:
        return 0
    
#design perceptron model
def perModel(x, w, b):
    v = np.dot(w, x) + b
    y = unitStep(v)
    return y

''' 

All logic functions are below.
Used to inform final logic of perceptron.
NOT, AND, OR, XOR.

'''

def NOT_logicFunction(x):
    wNOT = -1
    bNOT = 0.5
    return perModel(x, wNOT, bNOT)

def AND_logicFunction(x):
    w = np.array([1, 1])
    bAND = -1.5
    return perModel(x, w, bAND)
  
def OR_logicFunction(x):
    w = np.array([1, 1])
    bOR = -0.5
    return perModel(x, w, bOR)
  
#function calls in sequence
def XOR_logicFunction(x):
    y1 = AND_logicFunction(x)
    y2 = OR_logicFunction(x)
    y3 = NOT_logicFunction(y1)
    final_x = np.array([y2, y3])
    finalOutput = AND_logicFunction(final_x)
    return finalOutput


#Below is optional, just used for accuracy purposes.
#testing the perceptron model (good job lil' buddy!)
test1 = np.array([0, 1])
test2 = np.array([1, 1])
test3 = np.array([0, 0])
test4 = np.array([1, 0])
  
print("XOR({}, {}) = {}".format(0, 1, XOR_logicFunction(test1)))
print("XOR({}, {}) = {}".format(1, 1, XOR_logicFunction(test2)))
print("XOR({}, {}) = {}".format(0, 0, XOR_logicFunction(test3)))
print("XOR({}, {}) = {}".format(1, 0, XOR_logicFunction(test4)))



