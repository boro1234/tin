from PIL import Image
import os,glob
import numpy as np
from sklearn import model_selection

classes = ["bottle","chimney","mushroom","sausage","snake"]
num_classes = len(classes)
image_size = 50

#gazou no yomikomi

X = []
Y = []
for index,tin in enumerate(classes):
    photos_dir = "./"+ tin
    files = glob.glob(photos_dir+"/*.jpg")
    for i, fil in enumerate(files):
        if i >= 200 :break
        image = Image.open(fil)
        image = image.convert("RGB")
        image = image.resize((image_size,image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y)
xy = (X_train,X_test,Y_train,Y_test)
np.save("./tin.npy",xy)
