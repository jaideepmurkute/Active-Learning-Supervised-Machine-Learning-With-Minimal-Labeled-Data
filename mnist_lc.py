from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
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

sampling_strategy = {'least_confident': []}

def main():
    classifier_random=[LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=1000) for i in range(10)]
    classifier_active = [LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=1000) for i in range(10)]
    
    k = 0
    results_record = deepcopy(sampling_strategy)
    passive_results = []
    
    mnist = fetch_mldata('MNIST original')
    x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target)
    
    labeled_data, X_unlabeled, available_label, oracle_label_results = train_test_split(x_train,y_train,test_size = 0.99)
    
    for model in classifier_random:
        model.classes_ = np.arange(10)
        model.fit(labeled_data, available_label)
        
    for model in classifier_active:
        model.classes_ = np.arange(10)
        model.fit(labeled_data, available_label)
    
    labeled_data_rand = deepcopy(labeled_data)
    available_label_rand = deepcopy(available_label)
    labeled_data_active = deepcopy(labeled_data)
    available_label_active = deepcopy(available_label)
    
    batch_size = 32
        
    examples_list = [32]*30
            
    seen_examples_count = 32
    for num_queries in examples_list:
        seen_examples_count = seen_examples_count + num_queries
        num_samples.append(num_queries)
        
        random_datapoints = np.random.choice(X_unlabeled.shape[0], num_queries, replace=False)
        
        labeled_data_rand = np.concatenate((labeled_data_rand, X_unlabeled[random_datapoints, :]))
        available_label_rand = np.concatenate((available_label_rand, oracle_label_results[random_datapoints]))
        
        predictions = []
        for model in classifier_random:
            model.fit(labeled_data_rand, available_label_rand)
            predictions.append(model.predict(X_test))
        
        converted_to_stack = np.stack(predictions)
        commitee_decision = np.apply_along_axis(\
            lambda x: Counter( x ).most_common()[0][0],\
            0, converted_to_stack)
        matches = np.sum(commitee_decision == y_test)
        average_accuracy = matches / np.shape(X_test)[0]
        passive_results.append(average_accuracy)

        
        al_obj = ActiveLearner(strategy='least_confident')
        for model in classifier_active:
            model.classes_ = np.arange(10)
        indexes = al_obj.rank(classifier_active, X_unlabeled, num_queries)
        
        labeled_data_active = np.concatenate((labeled_data_active, X_unlabeled[indexes, :]))
        available_label_active = np.concatenate((available_label_active, oracle_label_results[indexes]))
        
        predictions = []
            
        for model in classifier_active:
            model.fit(labeled_data_active, available_label_active)
            curr_pred = model.predict(X_test)
            predictions.append(curr_pred)
                
        converted_to_stack = np.stack(predictions)
        commitee_decision = np.apply_along_axis(\
            lambda x: Counter(x).most_common()[0][0],\
            0, converted_to_stack)
        matches = np.sum(commitee_decision == y_test)
        average_accuracy =  matches / np.shape(X_test)[0]
        results_record['least_confident'].append(average_accuracy)
        k = k + 1
        
    np.savetxt('./misc/random_model_accuracy.txt', passive_results)
    np.savetxt('./misc/active_model_accuracy.txt', results_record['least_confident'])
    

if __name__ == '__main__':
    main()
        
