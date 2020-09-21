import numpy as np;
import scipy.linalg as linalg

def filterOutliers (Features, Dataset, numStdDevs):
	NewData = Dataset[:,:]
	for i in Features:
		var = NewData[:,i]
		avgvar = round(np.mean(var),3)
		stdvar = round(np.std(var),3)
		NewData = NewData[:,:][((NewData[:,i] < avgvar + (stdvar*numStdDevs)) & (NewData[:,i] > avgvar - (stdvar*numStdDevs)))]
	return NewData

def estimateVars (dataset):
	[m,n] = dataset.shape
	mu = sum(dataset) /m
	sigma = sum(np.power(dataset-mu,2))/m
	return mu, sigma


def probabilitize(data,mu,sigma):
	k = len(mu)
	if sigma.ndim == 1:
		sigma = np.reshape(sigma,(-1,sigma.shape[0]))
	if sigma.shape[1] == 1 or sigma.shape[0] == 1:
		sigma = linalg.diagsvd(sigma.flatten(), len(sigma.flatten()), len(sigma.flatten()))
		
	X = data - mu.reshape(mu.size, order='F').T
	p = np.dot(np.power(2 * np.pi, - k / 2.0), np.power(np.linalg.det(sigma), -0.5) ) * \
	np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(sigma)) * X, axis=1))
    
	return p

def getThreshold(yval, pval):
	stepsize = (max(pval) - min(pval)) / 1000
	i = min(pval)
	F1 = 0
	bestF1 = 0
	bestEps = 0
	truepos = 0
	falsepos = 0
	falseneg = 0
	while i < max(pval):
		pred = pval < i;
		tp = sum((pred - yval == 0) & (yval ==1))
		fp = sum((pred-yval == 1) & (pred==1))
		fn = sum((yval - pred ==1) & (pred == 0))
		prec = tp /(tp + fp);
		rec = tp / (tp + fn);
		F1 = (2 * prec * rec) / (prec + rec)
		if F1 > bestF1:
			bestF1 = F1
			bestEps = i
			truepos = tp
			falsepos = fp
			falseneg = fn
		i += stepsize
	return bestEps, bestF1, truepos, falsepos, falseneg
	


def trainData(TrainData, ValidationData, yval, featureCols, standardDevs):
	X = filterOutliers(featureCols, TrainData,standardDevs)
	Xval = ValidationData[:,featureCols]
	Xtrain = X[:,featureCols]
	[mu, sigma] = estimateVars(Xtrain)
	pval = probabilitize(Xval, mu, sigma)
	[eps, bestF1, tp, fp, fn] = getThreshold(yval,pval)
	print("best epsilon = " + str(eps))
	print("best F1 = " + str(bestF1))
	print("True Positives = " + str(tp))
	print("False Positives = " + str(fp))
	print("False Negatives = " + str(fn))
	return eps, bestF1, tp, fp, fn, mu, sigma



def exportData(dataset, filename):
	path = '/yourpathhere/'
	np.savetxt(path + filename, dataset, delimiter = ',', fmt='%25.8f')



def predictNewData (freshDataset, featureCols, eps, mu, sigma):
	features = freshDataset[:,featureCols]
	p = probabilitize(features,mu,sigma)
	c = freshDataset.shape[1]
	predictions = p < eps
	predictions.shape = [predictions.size,1]
	predictedData = np.append(freshDataset,predictions,1)
	predictedData = predictedData[:,:][(predictedData[:,c]==1)]
	return p, predictedData
	



def testThreshold(testData, ytest, eps, mu, sigma, featureCols):
	Xval = testData[:,featureCols]
	[predScore, predData] = predictNewData(testData, featureCols, eps,mu,sigma)
	prediction = predScore < eps
	tp = sum((prediction - ytest == 0) & (ytest ==1))
	fp = sum((prediction-ytest == 1) & (prediction==1))
	fn = sum((ytest - prediction ==1) & (prediction == 0))
	prec = tp /(tp + fp);
	rec = tp / (tp + fn);
	F1 = (2 * prec * rec) / (prec + rec)
	print("Testing epsilon = " + str(eps))
	print("F1 = " + str(F1))
	print("True Positives = " + str(tp))
	print("False Positives = " + str(fp))
	print("False Negatives = " + str(fn))
	print("Precision = " + str(prec))
	print("Recall = " + str(rec))
	return F1, tp, fp, fn, prec, rec
##NEW VARIABLES
"""
loadedData = np.loadtxt("/yourpathhere/Dec10_WW.txt"
				, skiprows = 1, delimiter = ',');

validationData = np.loadtxt("/yourpathhere/Jan2020_validationSet.txt"
				, skiprows = 1, delimiter = ',');

testData = np.loadtxt("/yourpathhere/Jan2020_testSet.txt"
				, skiprows = 1, delimiter = ',');


yval = validationData[:,6]
ytest = testData[:,6]
featureCols = [1,2,3,4,5]
##Get Params
[eps, bestF1, tp, fp, fn, mu, sigma]= trainData(loadedData, validationData, yval, featureCols, 55)


testThreshold(testData, ytest, eps, mu, sigma, featureCols)

[p, predData] = predictNewData(loadedData, featureCols, eps,mu,sigma)
exportData(predData, 'processedFile_Dec30.txt')
"""






