import os    
import cv2
    
path = r"E:\5-14minst\mnist_dataset\5_20\test\2/"
outpath = r'E:\5-14minst\mnist_dataset\5_20\28x28\test\2/'
list = []
if (os.path.exists(path)):
    files = os.listdir(path)
    for file in files:
        m = os.path.join(path,file)
        if (os.path.isdir(m)):
            h = os.path.split(m)
            list.append(h[1])
            
for x in range (len(list)):
    path_new = r'E:\5-14minst\mnist_dataset\5_20\test\2/'#+ list[x]+ "/"
    print(path_new)
    imagelist = os.listdir(path_new)
    for i in range (len(imagelist)):
        img = cv2.imread(path_new +imagelist[i])
        size = (28,28)  
        img = cv2.resize(img, size)

#cv2.imshow('image',img)
        cv2.imwrite(outpath +"/"+ list[x] +"/" + imagelist[i],img)