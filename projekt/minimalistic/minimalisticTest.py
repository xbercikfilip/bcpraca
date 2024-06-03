import tensorflow as tf
keras = tf.keras
from keras.models import load_model
import numpy as np

class_names = ["complex", "minimalistic"]
model = load_model("minimalisticModel.keras")
n=24
img_array = np.random.random((n,180,180,3))

for i in range(n):
    img = tf.keras.utils.load_img("projekt/minimalistic/testPhotos/{}.jpg".format(i+1, target_size=(180, 180)))
    img = tf.image.resize(img, (180,180))
    img_array[i] = img

predictions = model.predict(img_array)

for i in range(n):
    score = tf.nn.softmax(predictions[i])
    print(
    "Image number {} most likely belongs to {} with a {:.2f} percent confidence."
    .format(i+1, class_names[np.argmax(score)], 100 * np.max(score))
    )
