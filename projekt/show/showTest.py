import tensorflow as tf
keras = tf.keras
from keras.models import load_model
import numpy as np
import pandas as pd

def get_top_n_keys(sorted_items, n=100):
    top_n_keys = [key for key, _ in sorted_items[:n]]
    return top_n_keys

def sort_dict_by_value(input_dict):
    sorted_items = sorted(input_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_items

def getImages():
    model = load_model("showModel")
    excel_file = 'data.xlsx' 
    dataset = pd.read_excel(excel_file)
    n = len(dataset.index.tolist())
    predictions = model.predict(dataset)

    results = {}
    for i in range(n):
        score = 100*predictions[i][0]
        results[i] = score
        t = "don't show"
        if(score > 50):
            t = "show"

        print(
            "Image number {} belongs to {} with {:.2f} percent.".format(i+1, t, score)
        )   
    results = sort_dict_by_value(results)
    return get_top_n_keys(results)

print(getImages())



    