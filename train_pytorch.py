from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
import os

NUM_PROC = 5  # Parallel threads during the training of the next layer
CUDA = True  # Use GPU if available
buckets = 10000  # Number of models in layer 2
fileName = "/home/yash/Desktop/CSE-662/Code/LIS/data/sorted_keys_small.csv"  # Dataset
dataset_identifier = "dummy_data_1"

torch.set_default_dtype(torch.float64)


class SuperData(Dataset):
    """
    Dataloader class defined for all pyTorch based models
    :param filename: The relative directory path to the filename
    :return: return data item. Used by pyTorch internally to pass the data to the model
    """

    def __init__(self, filename):
        self.data = np.loadtxt(filename, delimiter=' ')
        np.random.shuffle((self.data).astype(np.float64))

    def __getitem__(self, item):
        key = self.data[item, 1]-1425168000107.1
        value = self.data[item, 0]
        return torch.tensor([key]), torch.tensor([value])

    def __len__(self):
        return self.data.shape[0]


class SuperNet(nn.Module):
    """
        SuperNet Class defined to initialize the Top layer
        :param n1: The number of nodes in the first layer
        :param n2: The number of nodes in the second layer, 0 if the second layer does not exist
        :param bias: Whether to add the bias layer or not
        :param bn: Boolean for batch normalization
        :return: Class initializes the model and creates the front propagation loop
        """

    def __init__(self, n1=16, n2=16, bias=True, bn=True):
        super(SuperNet, self).__init__()

        self.bn1 = None

        if n2 == 0:
            self.layers = nn.Linear(in_features=1, out_features=n1, bias=bias)
        else:
            self.layers = nn.Sequential(nn.Linear(in_features=1, out_features=n1, bias=bias),
                                        nn.ReLU(),
                                        nn.Linear(n1, n2, bias=bias),
                                        )

        self.out_layer = nn.Linear(n1 if n2 == 0 else n2, 1, bias=bias)

        if bn:
            self.bn1 = nn.BatchNorm1d(n1 if n2 == 0 else n2)

    def forward(self, x):

        out = F.relu(self.layers(x))
        if self.bn1:
            out = self.bn1(out)

        return self.out_layer(out)


class Net(nn.Module):
    """
    Net Class defined to initialize the second layer Linear regression models
    :return: Class initializes the model and creates the front propagation loop
    """

    def __init__(self, bias=True):
        super(Net, self).__init__()
        self.node = torch.nn.Linear(1, 1, bias=bias)

    def forward(self, x):
        return self.node(x)


class TrainTop:
    """
    Trains the top layer. The class variables includes a parametrized Model created using SuperNet
    Call the train function of class to start training and the test function to test the function. The test function also
    writes the data to the buckets file.
    :param epochs: The number of epochs
    :param batch_size: Model training batch size
    :param crit: Defines the criterion function i.e. Loss function. Default is MSELoss()
    :param lr: The learning rate for the model
    :param verbose: Whether to print the logs
    :param identifier: Used to identify the type of dataset
    :param n1: Number of layers in the first layer
    :param n2: Number of layers in the second layer
    :param bias: Bias used or not for Model: bool
    :param bn: Batch Normalization used or not for the model
    :param device: Whether to use GPU or not
    """

    def __init__(self, identifier, epochs, batch_size, device, filename, lr=0.01, crit=nn.MSELoss(), n1=32, n2=0,
                 bias=True, bn=False, optimizer="Adagrad", verbose=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.crit = crit
        self.verbose = verbose
        self.device = device
        self.identifier = identifier

        self.super = SuperNet(n1=n1, n2=n2, bias=bias, bn=bn).to(device)
        self.super.train()

        if optimizer == "Adagrad":
            self.optimizer = torch.optim.Adagrad(params=self.super.parameters(), lr=lr)
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(params=self.super.parameters(), lr=lr, momentum=0.9)
        elif optimizer == "Adamax":
            self.optimizer = torch.optim.Adamax(params=self.super.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif optimizer == "RMSProp":
            self.optimizer = torch.optim.RMSprop(params=self.super.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(params=self.super.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)

        self.dataset = SuperData(filename=filename)
        self.dataload = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):

        for e in range(self.epochs):
            if self.verbose:
                print("\n\nEpoch:", e+1)

            epoch_loss = 0

            for i, (key, val) in enumerate(self.dataload):
                key = key.to(self.device)
                val = val.to(self.device)

                output = self.super(key)

                loss = self.crit(output, val)

                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()

                if self.verbose:
                    if (i + 1) % 10 == 0:
                        print("Batch: ", i + 1, "Loss: ", loss.item())
                        epoch_loss += loss.item()

            print("\nEpoch: ", e + 1, "Loss: ", epoch_loss, "\n")
            if not os.path.exists("models/{}".format(self.identifier)):
                os.makedirs("models/{}".format(self.identifier))
            torch.save(obj=self.super.state_dict(), f="models/{}/super_layer.pt".format(self.identifier))

    def test(self, read_model=False, write_buckets=True, total_buckets=100):
        """
        :param read_model: Whether to read the model saved by the train function
        :param total_buckets: Divide the data between the buckets to train the model for next layer
        :param identifier: Helps identify the dataset
        :param write_buckets: Write the buckets out to disk. Writes the training data for next layer to the buckets directory.
        """
        if read_model:
            self.super.load_state_dict(torch.load(f="models/{}/super_layer.pt".format(self.identifier)))

        if self.verbose:
            print("\n\nEvaluation:\n\n")

        self.super.eval()

        big_bucket = dict()

        total_length = self.dataset.__len__()

        for i in range(total_buckets):
            big_bucket[i] = []

        f = open("models/{}/model_output.txt".format(self.identifier), "w+")

        for i, (key, val) in enumerate(self.dataload):

            key = key.to(self.device)
            val = val.to(self.device)

            output = self.super(key)

            for k, v, o in zip(key, val, output):
                k = k.item()
                v = v.item()
                o = o.item()

                if self.verbose and i % 80000 == 0:
                    print("Batch: ", i+1, "Key: ", k, "Value: ", v, "Model Output: ", o, "Difference: ", o-v)

                mn = (total_buckets * o) / total_length
                model_num = np.clip(np.floor(mn), 0, total_buckets - 1)
                big_bucket[int(model_num)].append([v, k])

                f.write(str(v) + " " + str(o) + "\n")

        if write_buckets:
            print("\n\nSaving data files for layer 2:\n\n")
            if not os.path.exists("buckets/{}".format(self.identifier)):
                os.makedirs("buckets/{}".format(self.identifier))
            for b in big_bucket:
                np.savetxt(fname="buckets/{}/bucket_{}.txt".format(self.identifier, b), X=np.array(big_bucket[b]), fmt="%f")


def pytorch_linreg(model, identifier, device, bucket=0, epochs=10, batch_size=64, lr=0.001, verbose=False):
    """
    The function has been replaced with the SciKit Learn implementation as the pytorch adds overhead and
    is overkill for training a linear regression model
    :param epochs: The number of epochs
    :param batch_size: Model training batch size
    :param model: Pass the linear regression pytorch model defines by the Net class
    :param lr: The learning rate for the model
    :param verbose: Whether to print the logs
    :param device: Whether to use GPU or not
    :param bucket: The model which needs to be trained
    :param identifier: Used to identify the dataset
    :return: Trains and writes the model performance to the file
    """
    ds = SuperData(filename="buckets/{}/bucket_{}.txt".format(identifier, bucket))
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)

    # criterion = torch.nn.MSELoss(size_average=False)
    # optimizer = torch.optim.SGD(our_model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.001)
    crit = torch.nn.MSELoss()

    if not os.path.exists("models/{}".format(identifier)):
        os.makedirs("models/{}".format(identifier))

    for e in range(epochs):
        for key, val in dl:
            key = key.to(device)
            val = val.to(device)

            output = model(key)

            loss = crit(output, val)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if e == epochs - 1 and verbose:
                for k, v, o in zip(key, val, output):
                    with open("models/{}/m_{}.txt".format(identifier, bucket), "a") as f:
                        f.write("{0}\t{1}\t{2}\t{3}\n".format(k.item(), v.item(), o.item(), v.item() - o.item()))
                        f.close()

    torch.save(obj=model.state_dict(), f="models/{}/model_{}.pt".format(identifier, bucket))


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
    if not os.path.exists("models/{}".format(identifier)):
        os.makedirs("models/{}".format(identifier))
    np.savetxt("models/{}/model_summary_scikit.txt".format(identifier), results, fmt='%.6f', delimiter='\t',
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
    data = np.transpose(np.loadtxt("buckets/{}/bucket_{}.txt".format(identifier, bucket), delimiter=' '))
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
        if not os.path.exists("models/{}".format(identifier)):
            os.makedirs("models/{}".format(identifier))
        op = np.concatenate((op, difference), axis=1)
        data = np.concatenate((np.transpose(data), op), axis=1)
        np.savetxt("models/{}/m_{}.txt".format(identifier, bucket), data, fmt='%.6f', delimiter='\t', newline='\n')
    return [regressor.coef_, regressor.intercept_, min_error, max_error, avg_error, rms_error, model_performance]


if __name__ == '__main__':
    use_cuda = CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #
    # # Create a trainer object for training the first layer
    trainer = TrainTop(identifier=dataset_identifier, epochs=1, batch_size=32768//2, device=device, filename=fileName,
                       lr=0.04, optimizer="Adam", crit=nn.MSELoss(), n1=32, n2=32, bias=True, bn=False, verbose=True)

    # Train the model using the hyper-params passed above
    trainer.train()

    # Testing loop and preparation for the second layer

    trainer.test(read_model=True, write_buckets=False, total_buckets=buckets)
    #
    # start = datetime.datetime.now()

    # save_scikit(identifier=dataset_identifier, buckets=buckets, verbose=False)

    # # Code to run all the pyTorch Linear Regression models parallel
    #
    # processes = []
    # i = 0
    # while i < buckets:
    #     for rank in range(NUM_PROC):
    #         p = mp.Process(target=pytorch_linreg, args=(Net(bias=True).to(device), device, i+rank, 25, 128, 0.01, True))
    #         # We first train the model across `num_processes` processes
    #         p.start()
    #         processes.append(p)
    #
    #     for p in processes:
    #         p.join()
    #     i += NUM_PROC
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

    # end = datetime.datetime.now()
    # print("\nTime taken: " + str(end - start))