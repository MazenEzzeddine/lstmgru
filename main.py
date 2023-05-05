import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import requests




def createFunction():
    timeVector = np.linspace(0,50, timeSteps)
    originalFunction = np.sin(50 * 0.6 * timeVector + 0.5) + np.sin(100 * 0.6 * timeVector + 0.5) + np.sin(
        100 * 0.05 * timeVector + 2) + 0.01 * np.random.randn(len(timeVector))

    # plot the function
    plt.plot(originalFunction[0:300])
    #plt.plot(originalFunction)

    plt.xlabel('Time')
    plt.ylabel('Function')
    plt.show()
    return originalFunction





# Press the green button in the gutter to run the script.
def createModel():
    model = keras.models.Sequential(
        [keras.layers.GRU(5, return_sequences=True, use_bias=False, activation='linear', input_shape=[None, 1]),
         keras.layers.GRU(5, return_sequences=False, use_bias=False, activation='linear'),
         keras.layers.Dense(predictionHorizon, activation='linear')])
    return model


def trainModel():
    filepath = "weights.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,
    #                              mode='auto')
    # callbacks_list = [checkpoint]
    #
    # # compile the model
    # model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mse'])
    # model.summary()
    # # fit the model
    # history = model.fit(Xtrain, Ytrain, batch_size=100, epochs=5000, validation_data=(Xvalid, Yvalid),
    #                     callbacks=callbacks_list, verbose=2)
    # load weights
    model.load_weights("weights.hdf5")

    # predict
    prediction = model.predict(Xtest)
    # plot the training and validation curves
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.figure()
    # plt.plot(epochs, loss, 'b', label='Training loss', linewidth=2)
    # plt.plot(epochs, val_loss, 'r', label='Validation loss', linewidth=2)
    # plt.title('Training and validation losses')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.legend()
    # plt.savefig('validation_and_loss.png')
    # plt.show()

    # visualize the prediction and test data
    plt.plot(prediction[1, :], color='k', label='Predicted', linewidth=2)
    plt.plot(Ytest[1, :], color='r', label='Test', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Function Value')
    plt.legend()
    plt.savefig('prediction1.png')
    plt.show()

    # visualize the prediction and test data
    plt.plot(prediction[0, :], color='k', label='Predicted', linewidth=2)
    plt.plot(Ytest[0,:], color='r', label='Test', linewidth=2)
    print("prediction[0]")
    print(prediction[0,:])
    #plt.plot(Ytest[2], color='r', label='Test', linewidth=2)

    plt.xlabel('Time')
    plt.ylabel('Function Value')
    plt.legend()
    plt.savefig('prediction2.png')
    plt.show()


def saveModel():
    tf.saved_model.save(model, "C:\\Users\\m.ezzeddine\\PycharmProjects\\lstmgru\\mdl\\1")


if __name__ == '__main__':
    timeSteps = 5000
    # total number of samples for training, validation, and test
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
    originalFunction = createFunction()

    for i in range(batchSize):
        dataSetX[i, :] = originalFunction[0 + (i) * offset:length + (i) * offset]
        dataSetY[i, :] = originalFunction[length + (i) * offset:length + predictionHorizon + (i) * offset]

    # reshape the data sets, such that we can use these data sets for recurrent neural networks
    dataSetX = dataSetX.reshape((batchSize, length, 1))
    dataSetY = dataSetY.reshape((batchSize, predictionHorizon, 1))
    Xtrain, Ytrain = dataSetX[:(int)(0.6 * batchSize), :], dataSetY[:(int)(0.6 * batchSize), :]
    Xvalid, Yvalid = dataSetX[(int)(0.6 * batchSize):(int)(0.8 * batchSize), :], dataSetY[(int)(0.6 * batchSize):(int)(
        0.8 * batchSize), :]
    Xtest, Ytest = dataSetX[(int)(0.8 * batchSize):, :], dataSetY[(int)(0.8 * batchSize):, :]
    print("===========================================")
    print(Xtest)


    print(np.shape(Xtest))
    # print(np.shape(Xtest[0,:,0]))
    # print(Xtest[0,:,0])
    # print(Ytest[0,:,0])
    # ar =  Xtest[0,:,0]
    # print(ar)
    # ar = Xtest[:, 2:3, :]
    # print(ar)

    model = createModel()
    #
    trainModel()


    #data = json.dumps({"instances": ar.tolist()})


    data = json.dumps({"instances": Xtest.tolist()})

    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/mdl:predict', data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    print(predictions)
    print(np.shape(predictions))
    print(np.shape(predictions)[1])
    #print(predictions[1])
    # print("Ytest[2]")
    # print(predictions[2])





    #print(np.shape(Ytest))



    # saveModel()






