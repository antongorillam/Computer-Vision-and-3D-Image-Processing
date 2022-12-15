from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import torch    

IMG_DIM = 28
MNIST_PATH = r"/home/anton/KTH/year5/5LSH0/mnist"

class DataLoader():
    """
    object for loading MNIST dataset
    """
    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.test_data = None


    def prepareMNIST(self, mnist_path):
        mndata = MNIST(mnist_path)
        train_images, train_labels = mndata.load_training()
        test_images, test_labels = mndata.load_testing()

        train_images, train_labels = np.array(train_images), np.array(train_labels)
        test_images, test_labels = np.array(test_images), np.array(test_labels)

        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.10)

        n_train, n_val = train_images.shape[0], val_images.shape[0]
        train_images = train_images.reshape(n_train, 1, IMG_DIM, IMG_DIM)
        val_images = val_images.reshape(n_val, 1, IMG_DIM, IMG_DIM)
        test_images = test_images.reshape(test_images.shape[0], 1, IMG_DIM, IMG_DIM)

        train_images, val_images, test_images = torch.from_numpy(train_images).type(torch.FloatTensor), torch.from_numpy(val_images).type(torch.FloatTensor), torch.from_numpy(test_images).type(torch.FloatTensor)
        train_labels, val_labels, test_labels = train_labels.astype(int), val_labels.astype(int), test_labels.astype(int)
        train_labels, val_labels, test_labels = torch.from_numpy(train_labels), torch.from_numpy(val_labels), torch.from_numpy(test_labels) 
        
        self.train_data = [train_images, train_images]
        self.val_data = [val_images, val_images]
        self.test_data = [test_images, test_images]

    def getDataLoaderMNIST(self, batch_size):
        train_images, train_labels = self.train_data
        val_images, val_labels = self.val_data
        test_images, test_labels = self.test_data

        train = torch.utils.data.TensorDataset(train_images,train_labels)
        val = torch.utils.data.TensorDataset(val_images, val_labels)
        test = torch.utils.data.TensorDataset(test_images, test_labels)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
        
        return train_loader, val_loader, test_loader

if __name__ == '__main__':
    dl = DataLoader()
    dl.prepareMNIST(mnist_path=MNIST_PATH)
    train_image = dl.train_data
    img_idx = np.random.randint(len(train_image))
    # plt.imshow(train_image[img_idx].reshape(IMG_DIM, IMG_DIM))

    train_loader, val_loader, test_loader = dl.getDataLoaderMNIST(batch_size=100)
    for i, (images, labels) in enumerate(train_loader):
        print(images)
        break