import numpy as np
import cv2
import glob
import os
path = r'E:\bigdata\mnist\mnist1\DataImages-Test'+'/'
outpath = r'E:\bigdata\mnist\mnist1\test'+'/'
list = []
if (os.path.exists(path)):
    files = os.listdir(path)
    for file in files:
        m = os.path.join(path,file)
        if (os.path.isdir(m)):
            h = os.path.split(m)
            list.append(h[1])
            
for x in range (len(list)):
    path_new = r'E:\bigdata\mnist\mnist1\DataImages-Test/' + list[x]+ "/"
    imagelist = os.listdir(path_new)
    for i in range (len(imagelist)):
        path_image = path + list[x] + "/" + imagelist[i]
        print( path_image)
        img = cv2.imread( path_image)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img)
        img2[:,:,0] = gray
        img2[:,:,1] = gray
        img2[:,:,2] = gray
        cv2.imwrite(outpath + str(x) + '/' + imagelist[i], img2)
            
