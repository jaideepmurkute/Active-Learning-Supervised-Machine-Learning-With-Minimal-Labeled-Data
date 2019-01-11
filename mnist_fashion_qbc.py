from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import sys
import os
#For active learning sample ranking function from https://github.com/davefernig/alp
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
    decision = {'vote_entropy': []}
    passive_results = []

    (X_train_set, y_train_set), (X_test_set, y_test_set) = fashion_mnist.load_data()
    x_train, x_test, y_train, y_test = train_test_split(X_train_set,y_train_set)
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

    labeled_data, unlabeled_data, available_label, oracle_label = train_test_split(x_train,y_train,test_size = 0.99)
    
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
        
    new_samples_listing = [32]*10    

    for new_samples in new_samples_listing:
        random_queries = np.random.choice(unlabeled_data.shape[0], new_samples, replace=False)
        
        labeled_data_rand = np.concatenate((labeled_data_rand, unlabeled_data[random_queries, :]))
        available_label_rand = np.concatenate((available_label_rand, oracle_label[random_queries]))
        
        predictions = []
        for model in classifier_random:
            model.fit(labeled_data_rand, available_label_rand)
            predictions.append(model.predict(X_test))

        prediction_stack = np.stack(predictions)
        commitee_decision = np.apply_along_axis(\
            lambda x: Counter(x).most_common()[0][0], 0,\
            prediction_stack)
        matches = np.sum(commitee_decision == y_test)
        average_accuracy =  matches/ np.shape(X_test)[0]
        passive_results.append(average_accuracy)

        
        al_obj = ActiveLearner(strategy='vote_entropy')
        for model in classifier_active:
            model.classes_ = np.arange(10)
        indexes = al_obj.rank(classifier_active, unlabeled_data, new_samples)
        
        labeled_data_active = np.concatenate((labeled_data_active, unlabeled_data[indexes, :]))
        available_label_active = np.concatenate((available_label_active, oracle_label[indexes]))

        predictions = []
            
        for model in classifier_active:
            model.fit(labeled_data_active, available_label_active)
            curr_pred = model.predict(X_test)
            predictions.append(curr_pred)
                
        prediction_stack = np.stack(predictions)
        commitee_decision = np.apply_along_axis(lambda x: Counter(x).most_common()[0][0], 0, prediction_stack)
        matches = np.sum(commitee_decision==y_test)
        average_accuracy =  matches/ np.shape(X_test)[0]
        decision['vote_entropy'].append(average_accuracy)
        
        k = k + 1
    np.savetxt('./misc/random_model_accuracy.txt', passive_results)
    np.savetxt('./misc/active_model_accuracy.txt', decision)
    

if __name__ == '__main__':
    main()
    
