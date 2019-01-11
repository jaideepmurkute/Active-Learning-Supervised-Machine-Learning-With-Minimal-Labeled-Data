# Active-Learning---Supervised-Machine-Learning-With-Minimal-Data
Today, many of the machine learning applications which perform with very high accuracy on practical use cases, rely on supervised learning strategies. Deep learning algorithms are able to learn even more complex phenomena but almost always they require even larger sets of labeled data to learn from. However, the task of data labeling is often costly, time-consuming, requires expert knowledge and is not always possible to accurately label large number of data samples. Active learning frameworks aims to alleviate this problem to lower the label complexity of the learning process and still have an equal or in somecases more accurate model for decision making. This repository implements active learning framework from scratch.

To test the validity of the approch, we use one-vs-all logistic regression model for multiclass classification tasks. Specifically, we make use MNIST-Digits and newer and more challenging MNIST-Fashion datasets and compare performance of active and passive learning models. Active learning models tested with three approches for data instance sampling - *Query by Committee, Least Confidence sampling and Entropy Based sampling*.

Repository also contains a research paper PDF that provides in depth description of the approach, literature studies and visualizations and summary of performance evaluation.

**Dependencies:**

sklearn -    for logistic regression model

numpy -      for vector operations support

keras -      for easy dataset access)

matplotlib - for visualizations

**Sample Results:**


![alt text](https://github.com/jaideepmurkute/Active-Learning---Supervised-Machine-Learning-With-Minimal-Data/blob/master/MNIST_Digits_Logistic_Regression.jpg)

![alt text](https://github.com/jaideepmurkute/Active-Learning---Supervised-Machine-Learning-With-Minimal-Data/blob/master/MNIST_Fashion_Logistic_Regression.jpg)
