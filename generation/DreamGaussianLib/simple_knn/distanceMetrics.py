import numpy as np
# **The following class contains methods to calculate distance between two points using various techniques**

# **Formula to calculate Eucledian distance:**
#
# <math>\begin{align}D(x, y) = \sqrt{ \sum_i (x_i - y_i) ^ 2 }\end{align}</math>

# **Formula to calculate Manhattan Distance:**
#
# <math>\begin{align}D(x, y) = \sum_i |x_i - y_i|\end{align}</math>

# **Formula to calculate Hamming Distance:**
#
# <math>\begin{align}D(x, y) = \frac{1}{N} \sum_i \delta_{x_i, y_i}\end{align}</math>

class distanceMetrics:
    '''
    Description:
        This class contains methods to calculate various distance metrics
    '''
    def __init__(self):
        '''
        Description:
            Initialization/Constructor function
        '''
        pass
        
    def euclideanDistance(self, vector1, vector2):
        '''
        Description:
            Function to calculate Euclidean Distance
                
        Inputs:
            vector1, vector2: input vectors for which the distance is to be calculated
        Output:
            Calculated euclidean distance of two vectors
        '''
        self.vectorA, self.vectorB = vector1, vector2
        if len(self.vectorA) != len(self.vectorB):
            raise ValueError("Undefined for sequences of unequal length.")
        distance = 0.0
        for i in range(len(self.vectorA)-1):
            distance += (self.vectorA[i] - self.vectorB[i])**2
        return (distance)**0.5
    
    def manhattanDistance(self, vector1, vector2):
        """
        Desription:
            Takes 2 vectors a, b and returns the manhattan distance
        Inputs:
            vector1, vector2: two vectors for which the distance is to be calculated
        Output:
            Manhattan Distance of two input vectors
        """
        self.vectorA, self.vectorB = vector1, vector2
        if len(self.vectorA) != len(self.vectorB):
            raise ValueError("Undefined for sequences of unequal length.")
        return np.abs(np.array(self.vectorA) - np.array(self.vectorB)).sum()
    
    def hammingDistance(self, vector1, vector2):
        """
        Desription:
            Takes 2 vectors a, b and returns the hamming distance
            Hamming distance is meant for discrete-valued vectors, though it is a
            valid metric for real-valued vectors.
        Inputs:
            vector1, vector2: two vectors for which the distance is to be calculated
        Output:
           Hamming Distance of two input vectors
        """
        self.vectorA, self.vectorB = vector1, vector2
        if len(self.vectorA) != len(self.vectorB):
            raise ValueError("Undefined for sequences of unequal length.")
        return sum(el1 != el2 for el1, el2 in zip(self.vectorA, self.vectorB))


