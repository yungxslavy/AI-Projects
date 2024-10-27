# Written By: Slavy 
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer

# data = pd.read_csv('./toy.csv')
training_data = pd.read_csv('./train.csv')
dev_data = pd.read_csv('./dev.csv')
cat_processor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
num_processor = MinMaxScaler(feature_range=(0, 2))

# Training Data Splits 
train_x = training_data.drop(columns=['id', 'target'])
train_y = training_data['target']

# Dev Data Splits
dev_x = dev_data.drop(columns=['id', 'target'])
dev_y = dev_data['target']

# Encode only cat data 
preprocessor = ColumnTransformer([('num', num_processor, ['age', 'hours']),
                                 ('cat', cat_processor, ['sector','edu','marriage',
                                                   'occupation', 'race', 'sex',
                                                   'country'])])

# One-hot encode our train_x data
train_encoded = preprocessor.fit_transform(train_x)
dev_encoded = preprocessor.transform(dev_x)

# Initialize lists to track error rates and positive prediction rates
train_error_rates = []
dev_error_rates = []
positive_rates_train = []
positive_rates_dev = []

for k in range(1, 101, 2):
    # Initialize with our k 
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(train_encoded, train_y)

    train_predictions = knn.predict(train_encoded)
    dev_predictions = knn.predict(dev_encoded)
    
    # Get scores 
    train_accuracy = accuracy_score(train_y, train_predictions)
    dev_accuracy = accuracy_score(dev_y, dev_predictions)
    train_error_rate = 1 - train_accuracy
    dev_error_rate = 1 - dev_accuracy


    # Calculate the positive prediction rates
    positive_rate_train = np.mean(train_predictions == '>50K') * 100 
    positive_rate_dev = np.mean(dev_predictions == '>50K') * 100

    # Store the error rates and positive prediction rates
    train_error_rates.append(train_error_rate * 100) 
    dev_error_rates.append(dev_error_rate * 100)
    positive_rates_train.append(positive_rate_train)
    positive_rates_dev.append(positive_rate_dev)
    
    # Print the results for this value of k
    print(f"k={k} train_err {train_error_rate*100:.1f}% (+: {positive_rate_train:.1f}%) "
          f"dev_err {dev_error_rate*100:.1f}% (+: {positive_rate_dev:.1f}%)")

# Identify the best dev error rate and corresponding k
best_k = 1 + 2 * np.argmin(dev_error_rates)
best_dev_error = np.min(dev_error_rates)

print(f"\nBest dev error: {best_dev_error:.1f}% achieved with k={best_k}")
