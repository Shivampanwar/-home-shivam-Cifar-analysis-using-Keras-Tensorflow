import numpy as np
import os
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def load_data():
    folder = os.getcwd()+"/"+"cifar-10-batches-py"
    #add folder containing data
    testsubfolder = []
    pathofsubfolders = []
    for i in os.listdir(folder):
        name = folder + "/" + i
        if i.split('/')[-1].split('_')[0] == 'data':
            pathofsubfolders.append(name)
        if i.split('/')[-1].split('_')[0] == 'test':
            testsubfolder.append(name)
    traindatasetlistx = []
    traindatasetlabels = np.zeros((len(pathofsubfolders)*10000))
    testdict = unpickle(testsubfolder[0])
    testsetlistfeatures = []
    for i in range(testdict['data'].shape[0]):
        matrix = np.zeros((32, 32, 3))
        matrix[:, :, 0] = testdict['data'][i][:1024].reshape((32, 32))
        matrix[:, :, 1] = testdict['data'][i][1024:2048].reshape((32, 32))
        matrix[:, :, 2] = testdict['data'][i][2048:].reshape((32, 32))
        testsetlistfeatures.append(matrix)
    testsetfeatures = np.array(testsetlistfeatures)
    testsetlabels = np.array(testdict['labels'])
    testsetlabels = testsetlabels.reshape(10000, 1)

    init = 0
    for subbatch in pathofsubfolders:
        subbatch_dict = unpickle(subbatch)
        # traindatasetlisty.append(subbatch_dict['labels'])
        traindatasetlabels[init:init + 10000]=np.array(subbatch_dict['labels'])
        init += 10000
        for i in range(subbatch_dict['data'].shape[0]):
            matrix = np.zeros((32, 32, 3))
            matrix[:, :, 0] = subbatch_dict['data'][i][:1024].reshape((32, 32))
            matrix[:, :, 1] = subbatch_dict['data'][i][1024:2048].reshape((32, 32))
            matrix[:, :, 2] = subbatch_dict['data'][i][2048:].reshape((32, 32))
            traindatasetlistx.append(matrix)
    traindatasetfeatures = np.array(traindatasetlistx)

    return ((traindatasetfeatures,traindatasetlabels.reshape((traindatasetlabels.shape[0],1))),(testsetfeatures,testsetlabels))
# (x_train, y_train), (x_test, y_test) = load_data()
# print "x_train shape is: "+str(x_train.shape)
# print "x_test shape is: "+str(x_test.shape)
# print "y_train shape is: "+str(y_train.shape)
# print "y_test shape is: "+str(y_test.shape)
#



















