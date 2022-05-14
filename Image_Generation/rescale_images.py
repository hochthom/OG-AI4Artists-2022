# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:19:05 2021

@author: Thomas
"""

import os
import cv2
import tqdm


IMG_DIR = 'D:/ART/AI-4-Artists/Datasets/AI4Artists/trainA+B'
#IMG_DIR = 'D:/ART/AI-4-Artists/Datasets/StyleGAN/metFaces/images'
IMG_SIZE = 256

# get the names of all the images in the folder 
imgs = os.listdir(IMG_DIR)

# process each of them
cnt = 0
for fn in tqdm.tqdm(imgs):
    # first reading the image
    I = cv2.imread(os.path.join(IMG_DIR, fn))
    if I is None:
        continue
    
    # now we want to know the smaller side of it 
    sz = min(I.shape[:2])
#    if sz < IMG_SIZE:
#        # we skip the image
#        continue
    
    # we process the offset to extract only the square inner part of the image
    oy = 0#(I.shape[0] - sz) // 2 
    ox = (I.shape[1] - sz) // 2
    I = cv2.resize(I[oy:oy+sz, ox:ox+sz, :], (IMG_SIZE, IMG_SIZE))
    
    # or we just rescale the image to the new side ignoreing the aspect ratio
    #I = cv2.resize(I, (IMG_SIZE, IMG_SIZE))
    
    # save the image to the images folder
    cv2.imwrite('AI4Artists/images/img_%05i.png' % cnt, I)
    cnt += 1
    #break

