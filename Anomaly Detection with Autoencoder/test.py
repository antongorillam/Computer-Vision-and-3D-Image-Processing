# Write an simple autoencoder for anomaly detection on MNIST, using stochastic gradient descent and python, written as a class. Do not use the torch nn library. Input will be a dataloader from torch.

import numpy as np
import torch

from dataloader import DataLoader 

IMG_DIM = 28
MNIST_PATH = r"/home/anton/KTH/year5/5LSH0/mnist"
EPOCHS = 20
BATCH_SIZE = 10


class SimpleAutoencoder:
    # Set the hyperparameters
    def __init__(self, learning_rate, momentum, input_dim, hidden_dim):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights
        self.weights = torch.randn((input_dim, hidden_dim))
        self.bias = torch.zeros((hidden_dim, 1))

    # Stochastic gradient descent
    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            for _, (image, label) in enumerate(data_loader):
                # Forward pass
                image = image.reshape(image.size(0), IMG_DIM*IMG_DIM)
                encoded_vector = image @ self.weights
                # Compute the loss
                reconstruction_vector = torch.mm(encoded_vector, torch.t(self.weights))
                loss = torch.sum(torch.div(torch.pow(reconstruction_vector - image, 2), 2))
                # Compute gradients
                grad_weights = torch.mm(torch.t(image), (reconstruction_vector - image))
                # Update weights
                self.weights -= self.learning_rate * grad_weights + self.momentum * self.weights
                print("epoch {}, loss {}".format(epoch, loss))

    # Compute reconstruction error for a sample
    def compute_reconstruction_error(self, sample):
        # Forward pass
        input_vector = sample[0]
        encoded_vector = torch.mm(input_vector, self.weights)

        # Compute the loss
        reconstruction_vector = torch.mm(encoded_vector, torch.t(self.weights))
        loss = torch.sum(torch.div(torch.pow(reconstruction_vector - input_vector, 2), 2))
        return loss

def main():
    dl = DataLoader()
    dl.prepareMNIST(mnist_path=MNIST_PATH, num_normal_data=100, num_anomaly_data=10)
    normal_loader, anomaly_loader = dl.getDataLoaderMNIST(BATCH_SIZE)
    ae = SimpleAutoencoder(1e-2, 1e-3, input_dim=IMG_DIM*IMG_DIM, hidden_dim=256)
    ae.train(data_loader=normal_loader, epochs=20)

if __name__ == '__main__':
    main()