import numpy as np
import matplotlib.pyplot as plt
import time

class ANN:

    def __init__(self, m, eta, batch_size, random_state=100) -> None:
        """
        Class attributes
        ----------------
        random_stat : int
            the random seed for the stochastic moments of the network.
        W : np.array(dtype=object) 
            The weight matrices,  W = [W0, W1] where W0~(mxd), W1~(kxm) 
        b : np.array(dtype=object)
            The bias vectors,  b = [b0, b1] where b0~(mx1), b0~(kx1), 
        k : int
            number of labels
        d : int
            number of dimensions for each data (Number of pixels in RBG, 3x32x32 for our case)     
        m : int
            number of hidden-layer (hyperparameter) 
        n : int
            number of training data 
        eta : int
            the learning rate 
        batch_size : int 
            what size of batch size to train during mini-batch SD  
        history: dict
            bookkeeping of desirible fact over each iterations
        """

        self.random_state = random_state
        self.m = m 
        self.W = None
        self.b = None
        self.k = None 
        self.d = None
        self.X = None
        self.X_train = None
        self.Y_train = None
        self.y_train = None
        self.X_test = None
        self.Y_test = None
        self.y_test = None
        self.eta = eta
        self.batch_size = batch_size

        self.history = {
            "cost_train": [],
            "cost_test" : [],
            "loss_train": [],
            "loss_test" : [],
            "acc_train" : [],
            "acc_test"  : [],
            "tot_iters" : 0,
        }

    def softmax(self, x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def relu(self, x):
        """ Standard definition of the relu function """
        return np.maximum(0, x)

    def sigmoid(self, x):
        """ Sigmoid function but independece is assumed """
        return np.exp(x) / (np.exp(x) + 1)

    def getOneHot(self, y):
        """
        Get one-hot representation of labels
        Params:
        -------
            y: array
                Array containing all the labels y~(n,)
        Returns:
            Y: np.array(dtype=float)
                Matrix of shape Y~(k,n)
        """	
        n = len(y)
        Y = np.zeros((self.k, n))
        for idx, elem in enumerate(y):
            Y[elem, idx] = 1
        return Y

    def getWeightBias(self):
        """ Initzilize weight and bias from a gaussian distribution """
        np.random.seed(self.random_state)
        W1 = np.random.normal(0, 1/np.sqrt(self.d), size=(self.m, self.d))
        b1 = np.zeros((self.m, 1))
        W2 = np.random.normal(0, 1/np.sqrt(self.m), size=(self.k, self.m))
        b2 = np.zeros((self.k, 1))
        
        self.W = np.array([W1, W2], dtype=object)
        self.b = np.array([b1, b2], dtype=object)
    

    def computeAccuracy(self, X, y):
        """ 
        Calculate the accuarcy of the model
        -----------------------------------
        Params:
        -------
        X: np.array(dtype=float)
            Input matrix of dimension dxn
        y: array(int)
            Label of corresponding X, list of lenght n
        Returns:
            acc: float
                number representing the percentage of corretly classified labels
        """ 
        p = self.evaluateClassifier(X)
        y_pred = np.argmax(p, axis=0)
        acc = np.mean(y_pred==y)
        
        return acc

    def evaluateClassifier(self, X_train):
        """ Performs forward-feed returns the probability of each label in X, as well as instanciate Xs"""
        layers = self.W.size
        X = np.empty(shape=(layers,), dtype=object)
        X[0] = X_train

        for l in range(layers-1):
            
            s = np.dot(self.W[l], X[l]) + self.b[l] 
            X[l+1] = self.relu(s) 
        
        s = np.dot(self.W[layers-1], X[layers-1]) + self.b[layers-1]
        P = self.softmax(s)
        self.X = X
        return P

    def computeCost(self, X, Y, lamda):
        n = X.shape[1]

        p = self.evaluateClassifier(X)
        
        loss = 1/ n * np.sum(-Y * np.log(p))
        reg = lamda * sum([sum(w) for w in [sum(ws**2)for ws in self.W]])
        cost = loss + reg

        return cost, loss


    def computeGradients(self, Y, P, lamda):
        """ Performs backwards-pass an calculate the gradient for weight and biases """
        layers = self.W.size 
        n = Y.shape[1] # For readability purposes
        grad_W = np.empty(shape=self.W.shape, dtype=object)
        grad_b = np.empty(shape=self.b.shape, dtype=object)
        G = P-Y   
        
        for l in range(layers-1, 0, -1):

            grad_W[l] = 1/n * (G @ self.X[l].T) + 2 * lamda * self.W[l]    
            grad_b[l] = 1/n * G.sum(axis=1) 
            grad_b[l] = grad_b[l].reshape(-1, 1)
            G = self.W[l].T @ G
            G = G * (self.X[l] > 0).astype(int)

        grad_W[0] = 1/n * G @ self.X[0].T + 2 * lamda * self.W[0]
        grad_b[0] = 1/n * G.sum(axis=1)
        grad_b[0] = grad_b[0].reshape(-1, 1)

        return grad_W, grad_b

    def fit(self, X_train, y_train, X_test, y_test, n_epochs, lamda):
        """ Fit the model to the input- and ouput data """
        self.d = X_train.shape[0]
        self.k = max(y_train) + 1
        self.X_train = X_train
        self.Y_train = self.getOneHot(y_train)
        self.y_train = y_train
        self.X_test = X_test
        self.Y_test = self.getOneHot(y_test)
        self.y_test = y_test
        self.getWeightBias() # Initiate weights and biases
        tic = time.perf_counter()

        for epoch in range(n_epochs):
            
            if epoch % 5 == 0:
                # Print something every n-th epoch
                time_elapsed = time.perf_counter() - tic
                print(f'Epoch number: {epoch}/{n_epochs}, Time elapsed: {int(time_elapsed/60)}min {int(time_elapsed%60)}sec')

            indices = np.arange(0, X_train.shape[1])
            np.random.shuffle(indices)
            X = self.X_train[:, indices]
            Y = self.Y_train[:, indices]        
            y = np.array(y_train)[indices]
            self.miniBatchGD(X, Y, y, lamda)
        
        time_elapsed = time.perf_counter() - tic
        self.final_test_acc = self.computeAccuracy(self.X_test, self.y_test)
        print(f'Epoch number: {n_epochs}/{n_epochs}, Time elapsed: {int(time_elapsed/60)}min {int(time_elapsed%60)}sec')

    def miniBatchGD(self, X, Y, y, lamda=0):
        """ Performs Minibatch Gradiend Decent """
        n = X.shape[1]
        n_batch = int(n/self.batch_size)

        for j in range(self.batch_size):
            j_start = int(j*n_batch)
            j_end = int((j+1)*n_batch)
            # inds = np.arange(j_start, j_end).astype(int)
            X_batch = X[:, j_start:j_end]
            Y_batch = Y[:, j_start:j_end]
            y_batch = y[j_start:j_end]

            p = self.evaluateClassifier(X_batch)
            grad_W, grad_b = self.computeGradients(Y_batch, p, lamda=lamda)

            self.W = self.W - self.eta * grad_W
            self.b = self.b - self.eta * grad_b
            
            ''' Performs bookkeeping '''
            self.history['tot_iters'] += 1
            

        cost_train, loss_train = self.computeCost(X_batch, Y_batch, lamda=lamda) 
        cost_test, loss_test = self.computeCost(self.X_test, self.Y_test, lamda=lamda) 
        acc_train = self.computeAccuracy(X_batch, y_batch)
        acc_test = self.computeAccuracy(self.X_test, self.y_test)
        self.history['cost_train'].append(cost_train)
        self.history['cost_test'].append(cost_test)
        self.history['loss_train'].append(loss_train)
        self.history['loss_test'].append(loss_test)
        self.history['acc_train'].append(acc_train)
        self.history['acc_test'].append(acc_test)

def plotLostVEpochs(train, test, iters_list, metric, x_axis, title_string, dir, y_lim=[0,2]):
	[y_min, y_max] = y_lim
	plt.figure()
	if test != []:
        
		plt.plot(iters_list, train, label=f'Training {metric}', color='g')
		plt.plot(iters_list, test, label=f'Validation {metric}', color='r')
	else:
		plt.plot(train, label=f'{metric}', color='r')

	plt.title(title_string)
	plt.xlabel(x_axis)	
	plt.ylabel(metric)
	plt.ylim(bottom=0)
	plt.legend()
	plt.grid()
	filename = title_string
	filename = filename.replace(":", "")
	filename = filename.replace(",", "")
	filename = filename.replace("\n", "")
	filename = filename.replace(" ", "_")
	filename = filename.replace(".", "")
	filename = filename = filename + ".png"
	plt.savefig(dir + filename)

def main():
    from mnist import MNIST
    import seaborn as sns

    mndata = MNIST('mnist')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_images, train_labels = np.array(train_images), np.array(train_labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)

    train_images = train_images.T
    test_images = test_images.T

    np.random.seed(100)

    """
    Best Architeture
    N_EPOCHS = 100
    LAMDA = 0.05
    ETA = 1e-5 
    """

    N_EPOCHS = 100
    LAMDA = 0
    ETA = 1e-5
    k = train_labels.max() + 1
    d = train_images.shape[1]

    print("Starting training model")
    ann = ANN(
        m=100, 
        eta=ETA,  
        batch_size=100,
        )
    ann.k = k
    ann.d = d
    ann.fit(
        X_train=train_images,
        y_train=train_labels,
        X_test=test_images,
        y_test=test_labels,
        n_epochs=N_EPOCHS,
        lamda=LAMDA
        )
    final_acc = ann.final_test_acc
    print(f'{final_acc} for lambda {LAMDA}')

    loss_train= ann.history['loss_train']
    loss_test= ann.history['loss_test']
    acc_train= ann.history['acc_train']
    acc_test= ann.history['acc_test']

    plt.figure()
    sns.set_style("darkgrid")
    sns.lineplot(y=acc_train, x=[i for i, _ in enumerate(acc_train)])
    sns.lineplot(y=acc_test, x=[i for i, _ in enumerate(acc_test)])
    plt.legend(["Train", "Validation"])
    plt.title(f"Model Accuaracy")
    plt.savefig(f'/home/anton/KTH/year5/5LSH0/Fundamentals of Neural Networks/result/Accuaracy')

    plt.figure()
    sns.set_style("darkgrid")
    sns.lineplot(y=loss_train, x=[i for i, _ in enumerate(loss_train)])
    sns.lineplot(y=loss_test, x=[i for i, _ in enumerate(loss_test)])
    plt.legend(["Train", "Validation"])
    plt.title(f"Model Loss")
    plt.savefig(f'/home/anton/KTH/year5/5LSH0/Fundamentals of Neural Networks/result/Loss')


if __name__ == '__main__':
    main()