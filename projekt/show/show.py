import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import layers, Sequential
from sklearn.model_selection import train_test_split

def show(user, userData, imagesData):
  METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.RootMeanSquaredError(name='rmse'),
      keras.metrics.MeanAbsoluteError(name='mae')
    ] 
    
  df1 = userData[user]
  df1.index = df1.index - 1 
  df2 = imagesData 
  raw_dataset = pd.concat([df2, df1], axis=1)
  dataset = raw_dataset.copy()
  n = len(dataset.index.tolist())
  print(n) 
  dataset = dataset.dropna()
  neg, pos = np.bincount(dataset[user])
  weight_for_0 = (1 / neg) * (n / 2.0) 
  weight_for_1 = (1 / pos) * (n / 2.0)

  class_weight = {0: weight_for_0, 1: weight_for_1}

  print('Weight for class 0: {:.2f}'.format(weight_for_0))
  print('Weight for class 1: {:.2f}'.format(weight_for_1))

  x_train = dataset[['sad', 'realistic', 'minimalistic', 'modern']].values
  y_train = dataset[user].values

  model = Sequential([
      layers.Dense(64, activation='relu', input_shape=(4,)),
      layers.Dense(32, activation='relu'),
      layers.Dense(16, activation='relu'),
      layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(), metrics=METRICS)
              
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

  history = model.fit(
    x_train, 
    y_train,
    validation_data=(x_val, y_val), 
    epochs=50,
    batch_size=26,
    class_weight=class_weight,
  )
  return model, history

def get_top_n_keys(sorted_items, n=100):
    top_n_keys = [key for key, _ in sorted_items[:n]]
    return top_n_keys

def sort_dict_by_value(input_dict):
    sorted_items = sorted(input_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_items

def getRecommendations(likedImages, mode):
    user_table = pd.read_excel("baseUsers.xlsx")
    data = {}
    for column in user_table.columns:
      data[column] = user_table[column].tolist()
    data["user21"] = likedImages
    transformed_table = pd.DataFrame(0, index=range(1,517), columns=user_table.columns)
    for user_column in data:
        transformed_table.loc[data[user_column], user_column] = 1
    targetUser = "user21"

    liked_array = list(transformed_table.index[transformed_table["user21"] == 1] - 1)  

    if(mode == "collab"):
        targetUser = find_similar_user(targetUser, transformed_table)[0]["user"]
        similar_users = find_similar_user(targetUser, transformed_table)[:2]
        print(similar_users)
        joined_array = []
        for user in similar_users:
            for item in data[user["user"]]:
                if item not in joined_array:
                    joined_array.append(item - 1)
        np.random.shuffle(joined_array)
        print(joined_array)
        return joined_array, liked_array
    
    elif(mode == "eval"):
        targetUser = "user21"
    elif(mode == "hybrid"):
        second = find_similar_user(targetUser, transformed_table)[0]["user"]
        second = list(transformed_table.index[transformed_table[second] == 1])  
        third = find_similar_user(targetUser, transformed_table)[1]["user"]
        third = list(transformed_table.index[transformed_table[third] == 1])  
        joined_array = likedImages + second + third
        data['user21'] = joined_array
        transformed_table.loc[data["user21"], "user21"] = 1
    categorized_images = pd.read_excel("baseCategorized.xlsx")
    model, history  = show(targetUser, transformed_table, categorized_images)
    n = len(categorized_images.index.tolist())
    predictions = model.predict(categorized_images)
    results = {}
    for i in range(n):
        score = 100*predictions[i][0]
        results[i] = score  
    results = sort_dict_by_value(results)
    """
    count = 0
    for key, value in results:
        print(f"{key}: {value}", end="\t")
        count += 1
        if count % 5 == 0:
            print()
   
    sorted_values = [value for _, value in results]
    y_true = transformed_table[targetUser].values
    y_pred = (predictions > 0.75).astype(int)
    
     """
    return get_top_n_keys(results), liked_array


def find_similar_user(user, data):
    similarities = [
        {'user': other_user, 'similarity': cosine_similarity(data[user], data[other_user])}
        for other_user in data if other_user != user
    ]
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0 
    return dot_product / (magnitude_a * magnitude_b)