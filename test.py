import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras.callbacks import ModelCheckpoint



if __name__ == '__main__':
    timeSteps = 5000   # total number of samples for training, validation, and test
    batchSize = 200

    # offset between two samples
    offset = 5
    # past window
    length = 50
    # prediction
    predictionHorizon = 20

    # create total empty data sets
    # input data set

    dataSetX = np.zeros(shape=(batchSize, length))
    # output data set
    dataSetY = np.zeros(shape=(batchSize, predictionHorizon))
    timeVector = np.linspace(0, 50, timeSteps)
    print(timeVector)
    print(timeVector.shape)

    originalFunction = np.sin(50 * 0.6 * timeVector + 0.5) + np.sin(100 * 0.6 * timeVector + 0.5) + np.sin(
        100 * 0.05 * timeVector + 2) + 0.01 * np.random.randn(len(timeVector))

    # plot the function
    plt.plot(originalFunction)
    plt.xlabel('Time')
    plt.ylabel('Function')
    plt.show()

