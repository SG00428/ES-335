import torchvision
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

class MF():

    def __init__(self, R, K, alpha, beta, iterations, logfile):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.current_iteration = 0
        self.last_loss = 0
        self.new_loss = 0
        self.filename = logfile
    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        with open(self.filename, 'a') as file:
                    file.write("K = %d, alpha = %.4f, beta = %.4f, iterations = %d\n" % (self.K, self.alpha, self.beta, self.iterations))
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()

            self.last_loss = self.new_loss
            self.new_loss = mse

            training_process.append((i, mse))
            self.current_iteration = i
            # if (i+1) % 10 == 0:
            #     print("Iteration: %d ; error = %.4f" % (i+1, mse))
            with open(self.filename, 'a') as file:
                    file.write("Iteration: %d ; error = %.4f\n" % (i + 1, mse))
        return training_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.get_alpha() * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.get_alpha() * (e - self.beta * self.b_i[j])

            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]

            # Update user and item latent feature matrices
            self.P[i, :] += self.get_alpha() * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.get_alpha() * (e * P_i - self.beta * self.Q[j,:])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

    def get_alpha(self):
        count = self.current_iteration
        if count < 100:
            return self.alpha
        elif count < 250:
            self.alpha = self.alpha/2
            return self.alpha
        else:
            if self.last_loss < self.new_loss:
                self.alpha = self.alpha/2
                return self.alpha
            else:
                self.alpha = self.alpha * 1.05
                return self.alpha



image = Image.open('dog.jpeg')
image.thumbnail((500, 500))
image.save('dog2.jpeg')

img = torchvision.io.read_image("dog2.jpeg")
image=image.convert('RGB')
# img_2d_1 = img[0,:,:]
# img_2d_2 = img[1,:,:]
# img_2d_3 = img[2,:,:]
img_2d_1 = image.getchannel('R')
img_2d_2 = image.getchannel('G')
img_2d_3 = image.getchannel('B')
print(type(img_2d_1))
img_1 = np.array(img_2d_1)
print(img_1.shape)
Image.fromarray(img_1).save('dog_original_1.jpeg')
img_1_normalized = img_1/255

img_2 = np.array(img_2d_2)
Image.fromarray(img_2).save('dog_original_2.jpeg')
img_2_normalized = img_2/255

img_3 = np.array(img_2d_3)
Image.fromarray(img_3).save('dog_original_3.jpeg')
img_3_normalized = img_3/255

mf1 = MF(img_1_normalized, K=50, alpha=0.02, beta=0, iterations=500, logfile= 'training_log1.txt')
training_process = mf1.train()
Image.fromarray((mf1.full_matrix()*255).astype(np.uint8)).save('dog_reconstructed1.jpeg')


mf2 = MF(img_1_normalized, K=50, alpha=0.02, beta=0, iterations=500, logfile= 'training_log2.txt')
training_process = mf2.train()
Image.fromarray((mf2.full_matrix()*255).astype(np.uint8)).save('dog_reconstructed2.jpeg')


mf3 = MF(img_1_normalized, K=50, alpha=0.02, beta=0, iterations=500, logfile= 'training_log3.txt')
training_process = mf3.train()
Image.fromarray((mf3.full_matrix()*255).astype(np.uint8)).save('dog_reconstructed3.jpeg')

bgr = cv2.merge([mf1.full_matrix()*255,mf2.full_matrix()*255,mf3.full_matrix()*255])
cv2.imwrite('dog_combined3.jpg', bgr)