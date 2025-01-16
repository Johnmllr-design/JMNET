#this class will hold the activation functions called by each node
from math import log
from math import exp

class softplus():

    def evaluate(self, input: float) -> float:
        return log(1 + exp(input))
    
    