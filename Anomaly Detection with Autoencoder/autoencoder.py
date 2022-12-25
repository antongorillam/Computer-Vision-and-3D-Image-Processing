import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from mnist import MNIST


from dataloader import DataLoader 


IMG_DIM = 28
MNIST_PATH = r"/home/anton/KTH/year5/5LSH0/mnist"

class Autoencoder:
    def __init__(self, input_shape, z_shape):
        self.w1 = np.random.randn(input_shape, z_shape)
        self.w2 = np.random.randn(z_shape, input_shape)
        self.b1 = np.zeros(z_shape)
        self.b2 = np.zeros(input_shape)
        
    def activation(self, x, activation_function="sigmoid"):
        if activation_function=="sigmoid":
            return 1/(1+np.exp(-x))
        elif activation_function=="relu":
            return np.maximum(0, x)

    def loss(self, y, y_prim, loss_function="mse"):
        if loss_function=="mse":
            return y - y_prim
        # TODO: Add another loss function
        else:
            return

    def forward(self, input):
        self.z1 = np.dot(input, self.w1) + self.b1
        self.a1 = self.activation(self.z1, activation_function="sigmoid")
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.activation(self.z2, activation_function="sigmoid")
        return self.a2
    
    def backward(self, input, output, lr):
        self.output_error = self.loss(input, output)
        self.output_delta = self.output_error * self.a2
        self.z1_error = self.output_delta.dot(self.w2.T)
        self.z1_delta = self.z1_error * self.a1 * (1 - self.a1)
        # print(f"output_error: {self.output_error.sum()}")
        # print(f"z1_delta: {self.z1_delta.sum()}")
        self.w1 -= input.T.dot(self.z1_delta) * lr
        self.w2 -= self.a1.T.dot(self.output_delta) * lr
        self.b1 -= self.z1_delta.sum(axis=0) * lr
        self.b2 -= self.output_delta.sum(axis=0) * lr

    def predict(self, input):
        tot_anomaly = 0
        for (i,_) in enumerate(input):
            y_pred = self.forward(input[i])
            anomaly_score = torch.norm((input - y_pred).float())            
            tot_anomaly += anomaly_score
        return tot_anomaly / len(input)

    def reconstructImages(self, images):
        pass

    def train(self, train_image, epochs):
        for epoch in range(epochs):
            np.random.shuffle(train_image)
            for image in train_image:
                # image = image.flatten().numpy()
                image = image.numpy()
                image = image.reshape(1, image.size)
                construced_image = self.forward(image)
                self.backward(image, construced_image, lr=1e-2)
            print(f"epoch: {epoch+1} outputerror: {self.output_error.sum():.2f}")

            

# def main():
#     dl = DataLoader()
#     dl.prepareMNIST(mnist_path=MNIST_PATH, num_normal_data=100, num_anomaly_data=100)

#     normal_images = dl.normal_data[0]
#     anomaly_images = dl.anomaly_data[0]
#     img_idx = np.random.randint(len(normal_images))
#     # plt.imshow(normal_images[img_idx].reshape(IMG_DIM, IMG_DIM))
#     # plt.show()

#     img_idx = np.random.randint(len(anomaly_images))
    
#     # plt.imshow(anomaly_images[img_idx].reshape(IMG_DIM, IMG_DIM))
#     # plt.show()
    
#     normal_loader, anomaly_loader = dl.getDataLoaderMNIST(50)

#     for i, (image, label) in enumerate(normal_loader):
#         x = image
#         break
    
#     for i, (image, label) in enumerate(normal_loader):
#         y = image
#         break

    

    # ae = Autoencoder(input_shape=IMG_DIM*IMG_DIM, z_shape=256)
    # ae.train(train_image=normal_images, epochs=12)

    # normal_score = ae.predict(normal_images.reshape(normal_images.size(0), IMG_DIM*IMG_DIM))
    # anomaly_score = ae.predict(anomaly_images.reshape(anomaly_images.size(0), IMG_DIM*IMG_DIM))
    # print(f"normal_score: {normal_score:.2f}")
    # print(f"anomaly_score: {anomaly_score:.2f}")
    
    # plt.imshow(pred.reshape(IMG_DIM, IMG_DIM))
    # plt.show()
    # print()

    # ae.train(train_image=train_image, epochs=30)
    
    # pred = ae.predict(image)
    
    # plt.imshow(pred.reshape(IMG_DIM, IMG_DIM))
    # plt.show()
    # print()
    
    
    
    


if __name__ == '__main__':
    main()