import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix, precision_recall_curve, auc
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
  model.save("showModel")
  return model, history
def get_top_n_keys(sorted_items, n=100):
    top_n_keys = [key for key, _ in sorted_items[:n]]
    return top_n_keys

def sort_dict_by_value(input_dict):
    sorted_items = sorted(input_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_items

def getRecommendations(likedImages, mode, user):
    user_table = pd.read_excel("baseUsers.xlsx")

    split_index = 15
    train_data = user_table.iloc[:split_index]
    eval_data = user_table.iloc[split_index:]
    user_table = train_data
    eval_user = user

    data = {}
    for column in user_table.columns:
      data[column] = user_table[column].tolist()
    data["user21"] = likedImages
    transformed_table = pd.DataFrame(0, index=range(1,517), columns=user_table.columns)
    for user_column in data:
        transformed_table.loc[data[user_column], user_column] = 1
    targetUser = user
    liked_array = list(transformed_table.index[transformed_table["user21"] == 1] - 1)  

    if(mode == "collab"):
        targetUser = find_similar_user(targetUser, transformed_table)[0]["user"]
        similar_users = find_similar_user(targetUser, transformed_table)[:2]
        #print(similar_users)
        joined_array = []
        for user in similar_users:
            for item in data[user["user"]]:
                if item not in joined_array:
                    joined_array.append(item - 1)
        np.random.shuffle(joined_array)
        #print(joined_array)
        return joined_array, liked_array
    
    elif(mode == "eval"):
        targetUser = eval_user
    elif(mode == "hybrid"):
        second = find_similar_user(targetUser, transformed_table)[0]["user"]
        second = list(transformed_table.index[transformed_table[second] == 1])  
        third = find_similar_user(targetUser, transformed_table)[1]["user"]
        third = list(transformed_table.index[transformed_table[third] == 1]) 
        if targetUser in user_table.columns:
            userData = user_table[targetUser].values.tolist()
        print(userData)
        print(second)
        print(third)
        joined_array = userData + second + third
        data[user] = joined_array
        transformed_table.loc[data[user], user] = 1

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
    """
    sorted_values = [value for _, value in results]
    y_true = transformed_table[targetUser].values
    y_pred = (predictions > 0.75).astype(int)
    results = get_top_n_keys(results)
    if eval_user in eval_data.columns:
        column_values = eval_data[eval_user].values.tolist()
        count_in_results = sum(1 for value in column_values if value in results)
    #plot_evaluation_metrics(history, y_true, y_pred, sorted_values)
    return results, liked_array, column_values, count_in_results


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


def plot_evaluation_metrics(history, y_true, y_pred, sorted_ratings, threshold=0.5):
    # Plot Training and Validation Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()
    # Calculate MAE and RMSE
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Plot MAE and RMSE
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['rmse'], label='RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Root Mean Squared Error (RMSE)')
    plt.legend()
    plt.show()
  
    """
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 6))
    cm = confusion_matrix(y_true, y_pred > threshold)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    #plt.show()

   

  # Calculate the cumulative distribution function
    sorted_ratings = np.sort(sorted_ratings)
    cdf = np.arange(1, len(sorted_ratings) + 1) / len(sorted_ratings)
    # Plot the CDF of ratings
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_ratings, cdf, marker='o', linestyle='-')
    plt.xlabel('Rating')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function (CDF) of Ratings')
    plt.grid(True)
    plt.show()
    """
users = []

for i in range(20):
    break
    user = "user" + str(i + 1)
    results, liked_array, column_values, count_in_results = getRecommendations([361, 363, 368, 369, 370, 371, 373], "hybrid", user)
    
    user_data = {
        "user": user,
        "count_in_results": count_in_results
    }
    
    users.append(user_data)
print(users)
getRecommendations([361, 363, 368, 369, 370, 371, 373], "hybrid")
#print(getRecommendations([101,102,103,104,516,105,106,107,111,112,113,114,121,122,123,124,516,125,136,137,141,142,143,161,152,153,154,356,155,156,177,171,172,173,181,182,183,184,396,195,196,197,191,212,213], "content"))
#print(getRecommendations([101,102,103,104,516,105,106,107,111,112,113,114,121,122,123,124,
#                          516,125,136,137,141,142,143,161,152,153,154,356,155,156,177,171,
#                          172,173,181,182,183,184,396,195,196,197,191,212,213], "hybrid"))