from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import sys
import os
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
    results_record = {'entropy': []}
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
        
    examples_list = [32]*30
        
    seen_examples_count = 32
    for new_examples_count in examples_list:
        seen_examples_count = seen_examples_count + new_examples_count
        num_samples.append(new_examples_count)
    
        random_datapoint = np.random.choice(X_unlabeled.shape[0], new_examples_count, replace=False)
        
        X_labeled_rand = np.concatenate((X_labeled_rand, X_unlabeled[random_datapoint, :]))
        y_labeled_rand = np.concatenate((y_labeled_rand, y_oracle[random_datapoint]))
        
        predictions = []
        for model in classifier_random:
            model.fit(X_labeled_rand, y_labeled_rand)
            predictions.append(model.predict(X_test))
        
        prediction_stack = np.stack(predictions)
        commitee_decision = np.apply_along_axis(\
            lambda x: Counter(x).most_common()[0][0],\
            0, prediction_stack)
        matches = np.sum(commitee_decision == y_test)
        average_accuracy =  matches / np.shape(X_test)[0]
        
        passive_results.append(average_accuracy)
        
        al_obj = ActiveLearner(strategy='entropy')
        for model in classifier_active:
            model.classes_ = np.arange(10)
        indexes = al_obj.rank(classifier_active, X_unlabeled, new_examples_count)
        
        X_labeled_active = np.concatenate((X_labeled_active, X_unlabeled[indexes, :]))
        y_labeled_active = np.concatenate((y_labeled_active, y_oracle[indexes]))
        
        predictions = []
            
        for model in classifier_active:
            model.fit(X_labeled_active, y_labeled_active)
            curr_pred = model.predict(X_test)
            predictions.append(curr_pred)
                
        commitee_decision = np.apply_along_axis(\
            lambda x: Counter(x).most_common()[0][0],\
            0, np.stack(predictions))
        matches = np.sum(commitee_decision == y_test)
    
        average_accuracy =  matches / np.shape(X_test)[0]
        results_record['entropy'].append(average_accuracy)
                
        k = k + 1
        
    np.savetxt('./misc/random_model_accuracy.txt', passive_results)
    np.savetxt('./misc/active_model_accuracy.txt', results_record['entropy'])
    
if __name__ == '__main__':
    main()
    
