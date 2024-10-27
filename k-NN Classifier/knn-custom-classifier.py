# Written By: Slavy 
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from collections import Counter

class CustomKNNClassifier:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric.lower()
        
    def fit(self, X_train, y_train):
        # Store the data 
        self.X_train = X_train
        self.y_train = y_train
    
    def distance(self, x1, x2):
        # Compute the distance between two points based on the selected metric 
        if self.metric == 'euclidean':
            return np.linalg.norm(x1 - x2, axis=1)
        elif self.metric == 'manhattan':
            # ord=1 for Manhattan distance
            return np.linalg.norm(x1 - x2, ord=1, axis=1)
        else:
            raise ValueError("Invalid distance metric. Use 'euclidean' or 'manhattan'.")
    
    def k_nearest(self, x):
        # Find the k nearest neighbors and their distances 
        distances = self.distance(self.X_train, x)
        
        # Get the sorted indices and distances
        k_indices = np.argsort(distances)[:self.k]
        k_distances = distances[k_indices]
        
        return k_indices, k_distances
        
    def predict(self, X_test):
        # Predict the class for each point in X_test 
        return np.array([self._predict(x) for x in X_test])
    
    def _predict(self, x):
        # Predict the outcome of a single point using majority voting
        k_indices, _ = self.k_nearest(x)  
        k_nearest_labels = [self.y_train[i] for i in k_indices]      
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Setup 
training_data = pd.read_csv('./train.csv')
dev_data = pd.read_csv('./dev.csv')
cat_processor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
num_processor = MinMaxScaler(feature_range=(0, 2))

# Training Data Splits 
train_x = training_data.drop(columns=['id', 'target'])
train_y = training_data['target']

dev_x = dev_data.drop(columns=['id', 'target'])
dev_y = dev_data['target']

# Encode only cat data 
preprocessor = ColumnTransformer([('num', num_processor, ['age', 'hours']),
                                 ('cat', cat_processor, ['sector','edu','marriage',
                                                   'occupation', 'race', 'sex',
                                                   'country'])])

# One-hot encode our train_x data
train_encoded = preprocessor.fit_transform(train_x)
dev_encoded = preprocessor.transform(dev_data.drop(columns=['id']))

# Initialize lists to track error rates and positive prediction rates
train_error_rates = []
dev_error_rates = []
positive_rates_train = []
positive_rates_dev = []

for currK in range(1, 101, 2):
    custom_knn = CustomKNNClassifier(k=currK, metric='euclidean')
    custom_knn.fit(train_encoded, train_y)

    train_predictions = custom_knn.predict(train_encoded)
    dev_predictions = custom_knn.predict(dev_encoded)

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
    print(f"k={currK} train_err {train_error_rate*100:.1f}% (+: {positive_rate_train:.1f}%) "
            f"dev_err {dev_error_rate*100:.1f}% (+: {positive_rate_dev:.1f}%)")
    
# Identify the best dev error rate and corresponding k
best_k = 1 + 2 * np.argmin(dev_error_rates)
best_dev_error = np.min(dev_error_rates)

print(f"\nBest dev error: {best_dev_error:.1f}% achieved with k={best_k}")