import operator
from .distanceMetrics import distanceMetrics

class kNNClassifier:
    '''
    Description:
        This class contains the functions to calculate distances
    '''
    def __init__(self,k = 3, distanceMetric = 'euclidean'):
        '''
        Description:
            KNearestNeighbors constructor
        Input    
            k: total of neighbors. Defaulted to 3
            distanceMetric: type of distance metric to be used. Defaulted to euclidean distance.
        '''
        pass
    
    def fit(self, xTrain, yTrain):
        '''
        Description:
            Train kNN model with x data
        Input:
            xTrain: training data with coordinates
            yTrain: labels of training data set
        Output:
            None
        '''
        assert len(xTrain) == len(yTrain)
        self.trainData = xTrain
        self.trainLabels = yTrain

    def getNeighbors(self, testRow):
        '''
        Description:
            Train kNN model with x data
        Input:
            testRow: testing data with coordinates
        Output:
            k-nearest neighbors to the test data
        '''
        
        calcDM = distanceMetrics()
        distances = []
        for i, trainRow in enumerate(self.trainData):
            if self.distanceMetric == 'euclidean':
                distances.append([trainRow, calcDM.euclideanDistance(testRow, trainRow), self.trainLabels[i]])
            elif self.distanceMetric == 'manhattan':
                distances.append([trainRow, calcDM.manhattanDistance(testRow, trainRow), self.trainLabels[i]])
            elif self.distanceMetric == 'hamming':
                distances.append([trainRow, calcDM.hammingDistance(testRow, trainRow), self.trainLabels[i]])
            distances.sort(key=operator.itemgetter(1))

        neighbors = []
        for index in range(self.k):
            neighbors.append(distances[index])
        return neighbors
        
    def predict(self, xTest, k, distanceMetric):
        '''
        Description:
            Apply kNN model on test data
        Input:
            xTest: testing data with coordinates
            k: number of neighbors
            distanceMetric: technique to calculate distance metric
        Output:
            predicted label 
        '''
        self.testData = xTest
        self.k = k
        self.distanceMetric = distanceMetric
        predictions = []
        for i, testCase in enumerate(self.testData):
            neighbors = self.getNeighbors(testCase)
            output= [row[-1] for row in neighbors]
            prediction = max(set(output), key=output.count)
            predictions.append(prediction)
        
        return predictions
