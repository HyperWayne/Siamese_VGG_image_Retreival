import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import random
from tensorflow.examples.tutorials.mnist import input_data
from numpy.random import choice, permutation
from itertools import combinations

flags = tf.app.flags
FLAGS = flags.FLAGS


class BatchGenerator():
	def __init__(self, imagesA, imagesB, labels):
		np.random.seed(0)
		random.seed(0)
		self.labels = labels
		#print (images.shape)
		self.imagesA = imagesA.reshape((781, 256, 256, 3))
		self.imagesB = imagesB.reshape((781, 256, 256, 3))
		

		#self.tot = len(labels)
		#self.i = 5    
		self.num_idx = dict()
        
       # 對label這個list一個一個index跑，值為num
		for idx, num in enumerate(self.labels):
            #如果這個數字已經有在dicionary，串起來，如果是5號，串在5號後面，並標明index是多少
            #如果不在dictionary，那5號就是第一個拉，一樣把index記錄下來
			if num in self.num_idx:
				self.num_idx[num].append(idx)
			else:
				self.num_idx[num] = [idx]		

		self.to_imgA = lambda x: self.imagesA[x]
		self.to_imgB = lambda x: self.imagesB[x]

	def next_batch(self, batch_size):
		left = []
		right = []
		sim = []	
        

		l_Unsim = choice(self.num_idx[False], 360, replace=False).tolist()
		left.append(self.to_imgA(l_Unsim.pop()))
		right.append(self.to_imgB(l_Unsim.pop()))
		sim.append([0])

		l_sim = choice(self.num_idx[True], 421, replace=False).tolist()
		left.append(self.to_imgA(l_sim.pop()))
		right.append(self.to_imgB(l_sim.pop()))
		sim.append([1])

		return np.array(left), np.array(right), np.array(sim)
"""
		for i in range(10):
			n = 45
              # choice:list, size, no repeated 
			l = choice(self.num_idx[i], n*2, replace=False).tolist()
			left.append(self.to_img(l.pop()))
			right.append(self.to_img(l.pop()))
			sim.append([1])
			
		for i,j in combinations(range(10), 2):
			left.append(self.to_img(choice(self.num_idx[i])))
			right.append(self.to_img(choice(self.num_idx[j])))
			sim.append([0])
""" 

def get_mnist():
	mnist = input_data.read_data_sets("MNIST_data/")
	return mnist
    
