import tensorflow as tf
keras = tf.keras
from keras.models import load_model
import numpy as np
import pandas as pd

happyNames = ["happy", "sad"]
realisticNames = ["abstract", "realistic"]
modernNames = ["classic", "modern"]
minimalisticNames = ["complex", "minimalistic"]

happyModel = load_model("icHappyModel")
realisticModel = load_model("icRealisticModel")
modernModel = load_model("icModernModel")
minimalisticModel = load_model("icMinimalisticModel")

n= 516
img_array = np.random.random((n,180,180,3))

for i in range(n):
    img = tf.keras.utils.load_img("projekt/base/{}.jpg".format(i+1, target_size=(180, 180)))
    img = tf.image.resize(img, (180,180))
    img_array[i] = img

happyPredictions = happyModel.predict(img_array)
realisticPredictions = realisticModel.predict(img_array)
modernPredictions = modernModel.predict(img_array)
minimalisticPredictions = minimalisticModel.predict(img_array)

predictions = [happyPredictions, realisticPredictions, modernPredictions, minimalisticPredictions]
results = np.random.random((n,4))


for i in range(n):

    results[i][0] = str(np.argmax(tf.nn.softmax(predictions[0][i])))
    results[i][1] = str(np.argmax(tf.nn.softmax(predictions[1][i])))
    results[i][2] = str(np.argmax(tf.nn.softmax(predictions[2][i])))
    results[i][3] =  str(np.argmax(tf.nn.softmax(predictions[3][i])))
    
    
print(results)
df = pd.DataFrame(results, columns=['sad', 'realistic', 'modern', 'minimalistic'])
df.to_excel("baseCategorized.xlsx", index=False)