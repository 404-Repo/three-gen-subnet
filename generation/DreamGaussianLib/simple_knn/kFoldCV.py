from random import randrange
from .kNNClassifier import kNNClassifier

class kFoldCV:
    '''
    This class is to perform k-Fold Cross validation on a given dataset
    '''
    def __init__(self):
        pass
        
    def printMetrics(self, actual, predictions):
        '''
        Description:
            This method calculates the accuracy of predictions
        '''
        assert len(actual) == len(predictions)
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predictions[i]:
                correct += 1
        return (correct / float(len(actual)) * 100.0)

    
    def crossValSplit(self, dataset, numFolds):
        '''
        Description:
            Function to split the data into number of folds specified
        Input:
            dataset: data that is to be split
            numFolds: integer - number of folds into which the data is to be split
        Output:
            split data
        '''
        dataSplit = list()
        dataCopy = list(dataset)
        foldSize = int(len(dataset) / numFolds)
        for _ in range(numFolds):
            fold = list()
            while len(fold) < foldSize:
                index = randrange(len(dataCopy))
                fold.append(dataCopy.pop(index))
            dataSplit.append(fold)
        return dataSplit
    
    
    def kFCVEvaluate(self, dataset, numFolds, *args):
        '''
        Description:
            Driver function for k-Fold cross validation
        '''
        knn = kNNClassifier()
        folds = self.crossValSplit(dataset, numFolds)
        print("\nDistance Metric: ",*args[-1])
        print('\n')
        scores = list()
        for fold in folds:
            trainSet = list(folds)
            trainSet.remove(fold)
            trainSet = sum(trainSet, [])
            testSet = list()
            for row in fold:
                rowCopy = list(row)
                testSet.append(rowCopy)
                
            trainLabels = [row[-1] for row in trainSet]
            trainSet = [train[:-1] for train in trainSet]
            knn.fit(trainSet,trainLabels)
            
            actual = [row[-1] for row in testSet]
            testSet = [test[:-1] for test in testSet]
            
            predicted = knn.predict(testSet, *args)
            
            accuracy = self.printMetrics(actual, predicted)
            scores.append(accuracy)

        print('*'*20)
        print('Scores: %s' % scores)
        print('*'*20)
        print('\nMaximum Accuracy: %3f%%' % max(scores))
        print('\nMean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
