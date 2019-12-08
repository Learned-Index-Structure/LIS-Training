import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from numpy import loadtxt
import tensorflow as tf
import datetime
from sklearn.linear_model import LinearRegression
import threading
import os


NUM_PROC = 5  # Parallel threads during the training of the next layer
buckets = 100  # Number of models in layer 2
fileName = "utils/newData10.csv"  # Dataset
dataset_identifier = "dummy_data"

def import_data(path):
    data = np.loadtxt(path, dtype=float, delimiter=' ')
    key = data[:, 1]
    value = data[:, 0]

    key = np.reshape(key, (-1, 1))
    value = np.reshape(value, (-1, 1))

    return key, value


class TrainTop:
    """
    Trains the top layer. The class variables includes a parametrized Model created using SuperNet
    Call the train function of class to start training and the test function to test the function. The test function also
    writes the data to the buckets file.
    :param epochs: The number of epochs
    :param batch_size: Model training batch size
    :param loss: Defines the criterion function i.e. Loss function. Default is MSELoss()
    :param lr: The learning rate for the model
    :param verbose: Whether to print the logs
    :param identifier: Used to identify the type of dataset
    :param n1: Number of layers in the first layer
    :param n2: Number of layers in the second layer
    :param bias: Bias used or not for Model: bool
    :param validation_split: Amount of validation used to avoid overfitting
    :param optimizer: Defines the optimizer used
    """

    def __init__(self, identifier, epochs, batch_size, filename, lr=0.01, loss="mse", n1=32, n2=0,
                 bias=True, optimizer='RMSprop', validation_split=0.1, verbose=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.verbose = verbose
        self.optimizer = optimizer
        self.identifier = identifier
        self.n1 = n1
        self.n2 = n2
        self.bias = bias
        self.lr = lr
        self.validation_split = validation_split
        self.model = None
        self.keys, self.values = None, None
        self.filename = filename

    def train(self):

        self.model = Sequential()
        self.model.add(Dense(self.n1, activation=tf.nn.relu, use_bias=self.bias, input_shape=(1,)))
        if self.n2 != 0:
            self.model.add(Dense(self.n2, activation=tf.nn.relu, use_bias=self.bias))
        self.model.add(Dense(1))
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['mse', 'mse'])

        self.keys, self.values = import_data(self.filename)
        self.model.fit(self.keys, self.values, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                       validation_split=self.validation_split)

        if not os.path.exists("models_tf/{}".format(self.identifier)):
            os.makedirs("models_tf/{}".format(self.identifier))
        self.model.save("models_tf/{}/super_layer.h5".format(self.identifier))

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)  # TF 2.0
        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

        # Following code is used for further optimization of weights. Here we notice a massive drop in performance
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        sample = tf.cast(self.keys, tf.float32)
        sample = tf.data.Dataset.from_tensor_slices((sample)).batch(1)
        def representative_data_gen():
            for input_value in sample.take(1300000):
                yield [input_value]
        converter.representative_dataset = representative_data_gen

        tflite_model = converter.convert()

        open("models_tf/{}/super_layer.tflite".format(self.identifier), "wb").write(tflite_model)

    def getWeights(self, tflite=False, read_model=False):
        """
        :param read_model: Whether to read the model saved by the train function
        :param tflite: Read from quantized tflite file
        """

        if self.keys is None:
            self.keys, self.values = import_data(self.filename)
        predictions = None

        if read_model:
            if tflite:
                self.model = tf.lite.Interpreter(model_path="models_tf/{}/super_layer.tflite".format(self.identifier))
                self.model.allocate_tensors()
                # Get input and output tensors.
                details = self.model.get_tensor(0)
                print(details)
                details = self.model.get_tensor(1)
                print(details)
                details = self.model.get_tensor(2)
                print(details)
                details = self.model.get_tensor(3)
                print(details)
                # details = self.model.get_tensor(4)
                # print(details)
                # details = self.model.get_tensor(4)
                # print(details)
                # details = self.model.get_tensor(5)
                # print(details)
                # details = self.model.get_tensor(6)
                # print(details)
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()

            else:
                if self.model is None:
                    self.model = tf.keras.models.load_model("models_tf/{}/super_layer.h5".format(self.identifier))
                for layer in self.model.layers:
                    print(layer.get_weights())
        else:
            for layer in self.model.layers:
                print(layer.get_weights())

    def test(self, tflite=False, read_model=False, write_buckets=True, total_buckets=100):
        """
        :param read_model: Whether to read the model saved by the train function
        :param total_buckets: Divide the data between the buckets to train the model for next layer
        :param write_buckets: Write the buckets out to disk. Writes the training data for next layer to the buckets directory.
        :param tflite: Read from quantized tflite file
        """
        if self.keys is None:
            self.keys, self.values = import_data(self.filename)
        predictions = None

        if read_model:
            if tflite:
                self.model = tf.lite.Interpreter(model_path="models_tf/{}/super_layer.tflite".format(self.identifier))
                self.model.allocate_tensors()
                # Get input and output tensors.
                input_details = self.model.get_input_details()
                output_details = self.model.get_output_details()
                self.keys = np.reshape(self.keys, (-1,1,1)).astype(np.float32)
                predictions = []
                for i in range(self.keys.shape[0]):
                    self.model.set_tensor(input_details[0]['index'], self.keys[i])
                    self.model.invoke()
                    predictions.append(self.model.get_tensor(output_details[0]['index'])[0])
                predictions = np.asarray(predictions)
            else:
                self.model = tf.keras.models.load_model("models_tf/{}/super_layer.h5".format(self.identifier))
                predictions = self.model.predict(self.keys)
        else:
            predictions = self.model.predict(self.keys)

        if self.verbose:
            print("\n\nEvaluation:\n\n")


        big_bucket = dict()
        self.keys = np.reshape(self.keys, (-1,1))
        predictions = np.concatenate((self.keys, self.values, predictions), axis=1)

        total_length = predictions.shape[0]

        for i in range(total_buckets):
            big_bucket[i] = []

        for i, (k, v, o) in enumerate(predictions):
            k = k.item()
            v = v.item()
            o = o.item()

            if self.verbose and i % 8000 == 0:
                print("Record: ", i+1, "Key: ", k, "Value: ", v, "Model Output: ", o, "Difference: ", o-v)

            mn = (total_buckets * o) / total_length
            model_num = np.clip(np.floor(mn), 0, total_buckets - 1)
            big_bucket[int(model_num)].append([v, k])

        if write_buckets:
            print("\n\nSaving data files for layer 2:\n\n")
            if not os.path.exists("buckets_tf/{}".format(self.identifier)):
                os.makedirs("buckets_tf/{}".format(self.identifier))
            for b in big_bucket:
                np.savetxt(fname="buckets_tf/{}/bucket_{}.txt".format(self.identifier, b), X=np.array(big_bucket[b]), fmt="%u")

def save_scikit(identifier, buckets, verbose=True):
    """
    Trains the Linear regression model and writes the corresponding output to the file
    :param bucket: Takes the total number of buckets
    :param verbose: Whether to print anything to console
    :return: Trains all the models and saves the model params in model_summary file
    """

    results = []
    for i in range(buckets):
        results.append(scikit_linreg(identifier=identifier, bucket=i, threshold=64, verbose=verbose))

    results = np.asarray(results)
    if not os.path.exists("models_tf/{}".format(identifier)):
        os.makedirs("models_tf/{}".format(identifier))
    np.savetxt("models_tf/{}/model_summary_scikit.txt".format(identifier), results, fmt='%.6f', delimiter='\t',
               newline='\n')


def scikit_linreg(identifier, bucket=0, threshold=64, verbose=True):
    """
    Trains the Linear regression model and writes the corresponding output to the file
    :param bucket: Takes the model number which is trained using linear regression
    :param verbose: Write all the output out to disk for analysis.
    :param threshold: Threshold to determine if the model is going to be replaced
    :return: Returns model param (y intercept and slope) and error measurements
    """
    regressor = LinearRegression(n_jobs=-1)
    data = np.transpose(np.loadtxt("buckets_tf/{}/bucket_{}.txt".format(identifier, bucket), delimiter=' '))
    regressor.fit(data[1].reshape(-1, 1), data[0].reshape(-1, 1))
    op = regressor.predict(data[1].reshape(-1, 1))
    difference = np.subtract(np.transpose(data[0]).reshape(-1, 1), op)
    min_error = np.min(difference)
    max_error = np.max(difference)
    avg_error = np.average(np.abs(difference))
    rms_error = np.average(np.square(difference))

    model_performance = True
    if np.abs(min_error) > threshold and max_error > threshold:
        model_performance = False

    if verbose:
        if not os.path.exists("models_tf/{}".format(identifier)):
            os.makedirs("models_tf/{}".format(identifier))
        op = np.concatenate((op, difference), axis=1)
        data = np.concatenate((np.transpose(data), op), axis=1)
        np.savetxt("models_tf/{}/m_{}.txt".format(identifier, bucket), data, fmt='%.6f', delimiter='\t', newline='\n')
    return [regressor.coef_, regressor.intercept_, min_error, max_error, avg_error, rms_error, model_performance]


if __name__ == '__main__':
    trainer = TrainTop(identifier=dataset_identifier, epochs=20, batch_size=2048, filename=fileName, lr=0.01,
                       loss="mse", n1=32, n2=32,
                       bias=True, optimizer='Adagrad', validation_split=0.0, verbose=True)

    trainer.train()
    trainer.test(tflite=True, read_model=False, write_buckets=True, total_buckets=100)
    trainer.test(tflite=True, read_model=True, write_buckets=True, total_buckets=100)
    # trainer.test(tflite=True, read_model=True, write_buckets=True, total_buckets=100)

    trainer.getWeights(tflite=True, read_model=True)

    start = datetime.datetime.now()

    # save_scikit(identifier=dataset_identifier, buckets=100, verbose=False)

    #
    # # Code to run all the SciKit Learn models parallel
    #
    # processes = []
    # i = 0
    # while i < buckets:
    #     for rank in range(NUM_PROC):
    #         p = threading.Thread(target=scikit_linreg, args=(i+rank, True))
    #         # We first train the model across `num_processes` processes
    #         p.start()
    #         processes.append(p)
    #
    #     for p in processes:
    #         p.join()
    #     i += NUM_PROC

    end = datetime.datetime.now()
    print("\nTime taken: " + str(end - start))
