# AI Projects
This repository is a collection of code from my AI (Machine Learning) learning journey. It's designed to serve as both a showcase of my progress and a resource for my future self or anyone interested in exploring this field.

# Table of Contents
1. [k-NN Classifier](#k-nn-classifier)


# k-NN Classifier
The k-NN Classifier, short for "k-Nearest Neighbors," is a method used to estimate categorical values based on a mix of continuous and categorical data. It's important to note that k-NN can also be applied to estimate continuous values in pairs with regression models. In the "k-NN Classification" folder, my code demonstrates the use of the k-NN Classifier with both the sklearn library and a custom-built k-NN model that I developed. This custom model provides insight into the inner workings of the classification process.  

### How k-Nearest Neighbors (k-NN) Works
k-Nearest Neighbors (k-NN) operates by calculating the distance between a target point and each point in the dataset. Typically, it uses **Euclidean distance**, although other metrics like **Manhattan** can be applied depending on the problem.

To efficiently calculate these distances, k-NN often utilizes matrix operations to handle the data in bulk. For example, in Euclidean distance, the calculation involves:

1. **Subtracting** each feature value of the target point from the corresponding feature value in each data point.
2. **Squaring** the result of each subtraction.
3. **Summing** the squared differences for each point.
4. **Taking the square root** of the sum to get the final Euclidean distance.

By using matrix math, these steps can be efficiently performed across the entire dataset, allowing k-NN to quickly identify the closest neighbors to the target point.

When we get our "Nearest Neighbors" We can assume that a certain value about our target will be most likely the same as the majority of it's neighbors. If all of its closest neighbors are making more than 50k a year we can assume that because our target is so closely related in certain data points, they likely are making more than 50k a year. 

### Highlights
- knn-classifier-binarization
    - This file exemplifies the dumb binarization using a one-hot encoder which simply converts every potential data value to be a 1 or 0 *(binarization)* It is dumb because it does it for every value including the numbers. So if you have 100 different ages it will make a category *age_00 age_01* ... for every value. 
    - While this still works *(You can try it)* and isn't too bad, however, this spread out our classification and hindered our results. Which leads us to the next file: Smart + Scaled Binarization

- knn-classifier-smart-scale
    - We now use ColumnTransformer to use our one-hot encoder for the categorical data, while keeping our numerical data untouched. This leads to another problem where now our numerical values like age and hours worked are way too high. This makes their weight, when assessing distances, heavy *(It compares 1s and 0s to numbers like 50)* which is wack because our entire dataset essentially gets ignored except for those big boy values. 
    - We resolve this by simply: `MinMaxScaler(feature_range=(0, 2))` Normalizing our numerical values from 0 to 2, we do 2 here because we utilize manhattan in this example, you can make it whatever makes sense to you. 
    - Now our classifier is considered **Smart + Scaled** Because we encoded our categorical data and scaled our numerical data to reduce the sensitivity. This is now our best chance at getting the best classifications. 

- knn-custom-classifier
    - This file is my custom classifier class that mimics and shows exactly everything that is happening under the hood when predicting. 
    - My custom classifier uses some tricks to speed up the run time *(its still mad slow)* take note of the usage of:
        - `np.linalg.norm(x1 - x2, ord=1, axis=1)` which is a method to efficiently get the distance of vectors.
        - `k_indices = np.argsort(distances)[:self.k]` A faster way to sort and remove data to make computations faster. We can also use `partition` which doesn't fully sort the array but only enough to get the top values we are looking for *(recommended)* 
    - The rest is all almost identical to the previous examples in getting the dev and training rates. 
    - This is one of many possible ways to implement a k-NN Classifier. You will likely have different methods to match the situation and datasets you find yourself swimming in. 
    - Use this as an example or to learn the step-by-step process of the k-NN Classifier. 
     

(More Soon...)

---
Thank you so much for checking this repo out! I genuinely hope this has been helpful, or inspiring, even in the smallest way.

Slavy : \)