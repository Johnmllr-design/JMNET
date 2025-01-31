from math import log
from math import exp

def dot(a1, a2) -> float:
    if len(a1) != len(a2):
        return -1
    else:
        sum = 0.0
        for i in range(0, len(a1)):
            sum += a1[i] * a2[i]
        return sum

def softplus(input: float) -> float:
        return log(1 + exp(input))
    
def softplus_derivative(input: float):
    return (1 / 1 + exp(input))
    

def SSR(a1, a2) -> float:
    sum = 0.0
    for i in range(0, len(a1)):
        sum += (a1[i] - a2[i]) * (a1[i] - a2[i])
    return (sum / len(a1))



def ReLU(input: float) -> float:
    return 0 if input < 0.0 else input
