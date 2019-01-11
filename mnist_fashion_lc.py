import sys
import os

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
#For sample ranking function from https://github.com/davefernig/alp
from active_learning.active_learning import ActiveLearner
from keras.datasets import fashion_mnist

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def main():
    classifier_random=[LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=1000) for i in range(10)]
    classifier_active = [LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=1000) for i in range(10)]
    
    k = 0
    active_results = {'least_confident':[]}
    passive_results = []

    (X_train_set, y_train_set), (X_test_set, y_test_set) = fashion_mnist.load_data()

    x_train, x_test, y_train, y_test = train_test_split(X_train_set,y_train_set)

    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])

    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])
    
    X_labeled, X_unlabeled, y_labeled, y_oracle = train_test_split(x_train,y_train,test_size = 0.99)
    
    for model in classifier_random:
        model.classes_ = np.arange(10)
        model.fit(X_labeled, y_labeled)
        
    for model in classifier_active:
        model.classes_ = np.arange(10)
        model.fit(X_labeled, y_labeled)
    
    X_labeled_rand = deepcopy(X_labeled)
    y_labeled_rand = deepcopy(y_labeled)
    X_labeled_active = deepcopy(X_labeled)
    y_labeled_active = deepcopy(y_labeled)
    
    batch_size = 32
        
    new_sample_size = [32]*20

    seen_examples_count = 32
    for new_sample_size in new_sample_size:
        seen_examples_count = seen_examples_count + new_sample_size
        num_samples.append(new_sample_size)
        
        random_queries = np.random.choice(X_unlabeled.shape[0], new_sample_size, replace=False)
        
        X_labeled_rand = np.concatenate((X_labeled_rand, X_unlabeled[random_queries, :]))
        y_labeled_rand = np.concatenate((y_labeled_rand, y_oracle[random_queries]))
        
        predictions = []
        for model in classifier_random:
            model.fit(X_labeled_rand, y_labeled_rand)
            predictions.append(model.predict(X_test))

        prediction_stack = np.stack(predictions)
        commitee_decision = np.apply_along_axis(\
            lambda x: Counter(x).most_common()[0][0],\
            0, prediction_stack)
        matches = np.sum(commitee_decision == y_test)
        average_accuracy = matches / np.shape(X_test)[0]
        passive_results.append(average_accuracy)

        
        al_obj = ActiveLearner(strategy='least_confident')
        for model in classifier_active:
            model.classes_ = np.arange(10)
        indexes = al_obj.rank(classifier_active, X_unlabeled, new_sample_size)
        
        X_labeled_active = np.concatenate((X_labeled_active, X_unlabeled[indexes, :]))
        y_labeled_active = np.concatenate((y_labeled_active, y_oracle[indexes]))

        predictions = []
            
        for model in classifier_active:
            model.fit(X_labeled_active, y_labeled_active)
            curr_pred = model.predict(X_test)
            predictions.append(curr_pred)
                
        prediction_stack = np.stack(predictions)
        commitee_decision = np.apply_along_axis(\
            lambda x: Counter(x).most_common()[0][0],\
            0, prediction_stack)
        matches = np.sum(commitee_decision == y_test)
        average_accuracy = matches / np.shape(X_test)[0]
        active_results['least_confident'].append(average_accuracy)
                
        k = k + 1

    np.savetxt('./misc/random_model_accuracy.txt', passive_results)
    np.savetxt('./misc/active_model_accuracy.txt', active_results['least_confident'])
    

if __name__ == '__main__':
    main()
    
