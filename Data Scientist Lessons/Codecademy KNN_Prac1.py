# -*- coding: utf-8 -*-

#Codecademy K-nearest neighbor Practice 1

import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#1. Import dataset
breast_cancer_data = load_breast_cancer()

#2-3. visualise dataset
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

#4-7. splitting dataset into test and train
training_data, validation_data, training_labels, validation_labels  = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

#8. Creating classifier object
classifier = KNeighborsClassifier(n_neighbors=3)

#9. Fitting training data into classifier object
classifier.fit(training_data, training_labels)

#10. Accuracy of the validation set
print(classifier.score(validation_data, validation_labels))

#11. Finding the best fit K
accuracies = []

for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors=k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))

#12. plotting Ks
k_list = range(1,101)
plt.plot(k_list, accuracies)
plt.xlabel("K")
plt.ylabel("Accuracies")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()

