# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 21:48:04 2017

@author: WayneLee
"""

import cv2
import numpy as np

class ImageSet:
    
  def __init__(self, filepath, start_position):
      
        self.filepath = filepath
        self.start_position = start_position
        
        # Split data into several attributes
      
        with open(self.filepath) as f:   
            content = f.readlines()
        
        # Verify the format
        tmp_label= ['Data/'+i.split()[0] for i in content]
        tmp_simA = ['Data/'+i.split()[1] for i in content]
        tmp_simB = ['Data/'+i.split()[2] for i in content]
        
        # Number of current data (SimilarA : N, same as B and all unsimilar)
        N = 781
        self.label= []
        self.simA = np.zeros((N, 256*256, 3)) 
        self.simB = np.zeros((N, 256*256, 3)) 
        
        for idx,element in enumerate(tmp_simA):
            imgA = cv2.imread(element)
            self.simA[idx] = np.reshape(imgA,[256*256, 3])
            
        for idx,element in enumerate(tmp_simB):
            imgB = cv2.imread(element)
            self.simB[idx]  = np.reshape(imgB,[256*256, 3])
            self.label.append('Un' not in tmp_label[idx])