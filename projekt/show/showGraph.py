import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
  print(df1)
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

def getRecommendations(likedImages, mode):
    user_table = pd.read_excel("baseUsers.xlsx")
    data = {}
    for column in user_table.columns:
      data[column] = user_table[column].tolist()
    data["user21"] = likedImages
    transformed_table = pd.DataFrame(0, index=range(1,517), columns=user_table.columns)
    for user_column in data:
        transformed_table.loc[data[user_column], user_column] = 1
    rmse_train_list = []
    mae_train_list = []
    rmse_test_list = []
    mae_test_list = []
    for i in range(1, 21):
        mode = "hybrid"
        targetUser = "user" + str(i)
        print("Processing", targetUser)


        if mode == "hybrid":
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
            data[targetUser] = joined_array
            transformed_table.loc[data[targetUser], targetUser] = 1
        categorized_images = pd.read_excel("baseCategorized.xlsx")
        model, history = show(targetUser, transformed_table, categorized_images)
        predictions = model.predict(categorized_images)
        n = len(categorized_images.index.tolist())
        y_true = transformed_table[targetUser].values
        y_pred = (predictions > 0.75).astype(int)

        train_rmse = history.history['rmse'][-1]
        train_mae = history.history['mae'][-1]
        test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        test_mae = mean_absolute_error(y_true, y_pred)
        rmse_train_list.append(train_rmse)
        mae_train_list.append(train_mae)
        rmse_test_list.append(test_rmse)
        mae_test_list.append(test_mae)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, 21), rmse_train_list, label='Training RMSE', marker='o', color='blue')
    plt.plot(range(1, 21), rmse_test_list, label='Testing RMSE', marker='o', color='red')
    plt.xlabel('User')
    plt.ylabel('Error')
    plt.title('Training and Testing RMSE for Users 1 to 20')
    plt.xticks(range(1, 21))
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(range(1, 21), mae_train_list, label='Training MAE', marker='o', color='green')
    plt.plot(range(1, 21), mae_test_list, label='Testing MAE', marker='o', color='orange')
    plt.xlabel('User')
    plt.ylabel('Error')
    plt.title('Training and Testing MAE for Users 1 to 20')
    plt.xticks(range(1, 21))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    196,203,208,261,288,310,340,342,343,365,367,420,424,455,478,479,480"""
print(getRecommendations([3,5,7,10,23,51,73,92,93,124,145,170], "hybrid"))

#print(getRecommendations([101,102,103,104,516,105,106,107,111,112,113,114,121,122,123,124,516,125,136,137,141,142,143,161,152,153,154,356,155,156,177,171,172,173,181,182,183,184,396,195,196,197,191,212,213], "content"))
#print(getRecommendations([101,102,103,104,516,105,106,107,111,112,113,114,121,122,123,124,
#                          516,125,136,137,141,142,143,161,152,153,154,356,155,156,177,171,
#                          172,173,181,182,183,184,396,195,196,197,191,212,213], "hybrid"))