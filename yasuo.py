# -*- coding: utf-8 -*-
"""
Wayne Lee 

NTHU 

"""

import cv2
import numpy as np
from dataImage import ImageSet

#abc = Train();
#abc.simA

# Train.simA

class Yasuo:
    
  def __init__(self, filepath):
      
        self.filepath = filepath
        
        # start_position reserved for the cut of Validation and test data
        # 0 : train, 1 : test, 2 : Validation
        
        start_position = np.zeros(3) 
        
        self.Train      = ImageSet(self.filepath,start_position[0])
        self.Test       = ImageSet(self.filepath,start_position[1])
        self.Validation = ImageSet(self.filepath,start_position[2])
    

"""
    #filepath_unsim = 'Data/unpair.txt'
    
    # step 2 : filename_queue
    simA_queue = tf.train.string_input_producer(simA)
    simB_queue = tf.train.string_input_producer(simB)
    
    # step 3: read, decode and resize images
    reader = tf.WholeFileReader()
    
    filenameA, contentA = reader.read(simA_queue)
    filenameB, contentB = reader.read(simB_queue)
    
    imageA = tf.image.decode_png(contentA, channels=3)  
    imageA = tf.cast(imageA, tf.float32)
    resized_imageA = tf.image.resize_images(imageA, [256, 256])
    
    imageB = tf.image.decode_png(contentB, channels=3)
    imageB = tf.cast(imageB, tf.float32)
    resized_imageB = tf.image.resize_images(imageB, [256, 256])
    
    # step 4: Batching
    image_batchA = tf.train.batch([resized_imageA], batch_size=8)
    image_batchB = tf.train.batch([resized_imageB], batch_size=8)
"""

