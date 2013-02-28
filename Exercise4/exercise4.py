# -*- coding: utf-8 -*-
import cv2, numpy as np

## 1. Train and use a naive Bayes classifier. ------------------------
def CallNaiveBayes(trainData, trainLabels, testData):

  ## Create a Naive (normal) Bayes Classifier
  ## TODO
  
  ## Use classifier to predict the test data
  ## TODO: (Overwrite the following line with your answer.)
  predictions = np.zeros((testData.shape[0], 1), np.float32)

  return predictions


## 3. Reduce the dimensionality of the data ---
def ReduceDimensionality(trainData, testData):

  ## Reduce the dimensionality of the data.
  ## TODO
  return trainData, testData



##------------------------------------------------------------------------
##------------------------------------------------------------------------
##---- No need to change anything below this point. ----------------------
##------------------------------------------------------------------------

letters = "abcdefghij"
def printConfusionMatrix(predictions, actual):
  
  numClasses = int(np.max(actual) - np.min(actual) + 1);
  print "Confusion matrix for %i classes:" % numClasses
  print "".join("\t%4s" % letters[i] for i in range(numClasses))
  
  for i in range(numClasses):
    print letters[i],
    
    for j in range(numClasses):
      predictedClass = predictions[actual == i]
      print "\t%4i" % sum(predictedClass == j),
    print

def calculateMeanF1(predictions, actual):
  numClasses = int(np.max(actual) - np.min(actual) + 1)
  f1s = np.zeros((numClasses, 1))
  
  for i in range(numClasses):
    predictedClass = predictions[actual == i]
    tp = sum(predictedClass == i)   # true positives
    fp = sum(predictions == i) - tp # false positives
    fn = sum(predictedClass != i)   # false negatives
    
    precision = tp / float(tp + fp) if tp + fp > 0 else 0
    recall    = tp / float(tp + fn) if tp + fn > 0 else 0
    
    if precision + recall != 0:  
      f1s[i] = 2 * precision * recall / (precision + recall)
  
  print "Mean F1 value: %.4f" % np.mean(f1s)


if __name__ == "__main__":
  
  ## Load training and test data.
  trainData2 = np.loadtxt('input/ocr2-train.txt', np.float32, delimiter=',')
  testData2  = np.loadtxt('input/ocr2-test.txt',  np.float32, delimiter=',')
  
  trainSamples2   = trainData2[:, :128]
  trainResponses2 = trainData2[:, 128]
  
  testSamples2    = testData2[:, :128]
  testResponses2  = testData2[:, 128]
  
  trainSamples2,testSamples2 = ReduceDimensionality(trainSamples2, testSamples2)
  
  ## Train a model
  predictions2 = CallNaiveBayes(trainSamples2, trainResponses2, testSamples2)
  
  print
  print 'Performance on a-b classification'
  print '---------------------------------'
  printConfusionMatrix(predictions2, testResponses2)
  calculateMeanF1(predictions2, testResponses2)
  print
  
  
  ##---- More challenging data set ---------------------------------------
  
  ## Load training and test data.
  trainData10 = np.loadtxt('input/ocr10-train.txt', np.float32, delimiter=',')
  testData10  = np.loadtxt('input/ocr10-test.txt',  np.float32, delimiter=',')
  
  trainSamples10   = trainData10[:, :128]
  trainResponses10 = trainData10[:, 128]
  
  testSamples10    = testData10[:, :128]
  testResponses10  = testData10[:, 128]
  
  trainSamples10, testSamples10 = ReduceDimensionality(trainSamples10, testSamples10)
  
  ## Train a model
  predictions10 = CallNaiveBayes(trainSamples10, trainResponses10, testSamples10)
  
  print
  print 'Performance on a-j classification'
  print '---------------------------------'
  printConfusionMatrix(predictions10, testResponses10)
  calculateMeanF1(predictions10, testResponses10)
  print
