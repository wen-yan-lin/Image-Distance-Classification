import numpy as np
import tensorflow as tf
import random

def trainTestSplit_oneClass(allFeat, allGt, target, numSamples=100):
    mask = np.where(allGt ==target)[0]
    trainInd = random.sample(list(mask), numSamples)    
    trainFeat = np.copy(allFeat[trainInd])
    testFeat = allFeat.copy()
    testFeat = np.delete(testFeat, trainInd, axis =0)
    testGt = allGt.copy()
    testGt = np.delete(testGt, trainInd, axis =0)
    return trainFeat, testFeat, testGt

def trainTestSplit_multiClass(allFeat, allGt, numSamples=100):
    numClass = int(np.max(allGt)+1)
    trainFeat = []
    testFeat = allFeat.copy()
    testGt = allGt.copy()
    for i in range(numClass):
        trainF, testFeat, testGt = trainTestSplit_oneClass(testFeat, testGt, i, numSamples=numSamples)
        trainFeat.append(trainF)   
    return trainFeat, testFeat, testGt
        

def unstackFeat(feat_):
    feat = np.concatenate(feat_, axis=0)
    gt = np.zeros(feat.shape[0], dtype=int)
    label = 0
    cur = 0
    for f in feat_:
        gt[cur:cur+f.shape[0]] = label
        label = label + 1
        cur = cur + f.shape[0]
    return feat, gt



def import_data(setIndex):

    folderNames = ['fashion-mnist',
                   'STL-10', 
                   'fake-stl10', 
                   'MIT-Places-Small',  
                   'dogvscat',
                   'mnist']

    print(folderNames[setIndex])
    if setIndex == 0:
        mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        trainFeat = x_train.reshape(x_train.shape[0], 28*28)
        testFeat = x_test.reshape(x_test.shape[0], 28*28)
        testGt = y_test
        trainGt = y_train

        trainFeat= trainFeat-255/2
        testFeat= testFeat-255/2


    elif setIndex == 1:
        allFeat = np.load('../STL-10/resNet50.npy')
        gtAll = np.load('../STL-10/gt.npy')
        trainFeat_, testFeat, testGt = trainTestSplit_multiClass(allFeat, gtAll, 1000)
        trainFeat, trainGt = unstackFeat(trainFeat_)

    elif setIndex == 2:
        testFeat = np.load('../STL-10/resNet50.npy')
        testGt = np.load('../STL-10/gt.npy')
        trainFeat = np.load('../fake-stl10/resNet50.npy')
        trainGt = np.load('../fake-stl10/gt.npy')

    elif setIndex == 3:
        allFeat = np.load('../MIT-Places-Small/resNet50.npy')
        gtAll = np.load('../MIT-Places-Small/gt.npy')
        mask = gtAll<5
        allFeat = allFeat[mask]
        gtAll = gtAll[mask]
        trainFeat_, testFeat, testGt = trainTestSplit_multiClass(allFeat, gtAll, 500)
        trainFeat, trainGt = unstackFeat(trainFeat_)

    elif setIndex == 4:
        testFeat = np.load('../dogvscat/cats_vs_dogs224feats/feats_test.npy')
        testGt = np.load('../dogvscat/cats_vs_dogs224feats/y_test.npy')

        trainFeat = np.load('../dogvscat/cats_vs_dogs224feats/feats_train.npy')
        trainGt = np.load('../dogvscat/cats_vs_dogs224feats/y_train.npy')


    elif setIndex == 5:
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        trainFeat = x_train.reshape(x_train.shape[0], 28*28)
        testFeat = x_test.reshape(x_test.shape[0], 28*28)
        testGt = y_test
        trainGt = y_train
        trainFeat= trainFeat-255/2
        testFeat= testFeat-255/2



    else:
        print('Data-set ' + setIndex + 'is not defined.')


    return trainFeat, trainGt, testFeat, testGt, folderNames[setIndex]


