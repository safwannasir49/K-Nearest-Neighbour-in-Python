<div align="center"><img src="ML Images/knn.png" width="100%"></div>

## Overview:
K Nearest Neighbors (KNN) is a simple and powerful algorithm for classification tasks. The KNN classifier predicts the class of a given data point by identifying the K nearest neighbors and using a majority vote to determine the class. It is widely used due to its simplicity and effectiveness in various domains.

### KNN Algorithm Intuition:
KNN algorithm works by finding the distance between the test data point and all the training data points. It then selects the K nearest data points and assigns the class that is most common among them. The distance metric commonly used is the Euclidean distance.

### Applications of KNN:
KNN is useful in several applications such as:
- Image recognition
- Recommendation systems
- Video recognition
- Data imputation

<hr/>

## Dataset:
The dataset used in this implementation is provided by a company, with feature column names hidden and a target class to predict. The first column is used as the index.

## Implementation:

**Libraries:**  `NumPy` `pandas` `matplotlib` `sklearn` `seaborn`

### Data Exploration:

```python
import pandas as pd

df = pd.read_csv("Classified Data", index_col=0)
df.head()
```

### Standardization:
Since KNN is sensitive to the scale of the data, we standardize the features to ensure that each feature contributes equally to the distance calculations.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
df_feat.head()
```

### Train Test Split:
Split the data into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.30)
```

### KNN Implementation:
We start with k=1.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
```

### Evaluation:
Evaluate the KNN model using confusion matrix and classification report.

```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```

### Choosing K Value:
Use the elbow method to find the optimal k value.

```python
error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
```

### Retrain with Optimal K:
Retrain the model with the optimal k value (k=23) and evaluate.

```python
knn = KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```
<br><br><br><br>
<hr/>
<br><br><br><br>
<h2 align="center">Results</h2>

### Error Rate vs K-Value
<img src="">

The model with k=23 demonstrated a higher accuracy and better classification performance compared to k=1.

### Confusion Matrix with k=23:
```plaintext
[[132  11]
 [  5 152]]
```

### Classification Report with k=23:
```plaintext
             precision    recall  f1-score   support

          0       0.96      0.92      0.94       143
          1       0.93      0.97      0.95       157

avg / total       0.95      0.95      0.95       300
```

### Conclusion
The KNN Classifier, with its simplicity and effectiveness, proved to be a powerful tool for classification tasks. After selecting the optimal k value, the model showed a significant improvement in performance metrics, making it a reliable choice for predicting class labels in this dataset.

## Contributing
Contributions are welcome! Please create an issue or submit a pull request.

