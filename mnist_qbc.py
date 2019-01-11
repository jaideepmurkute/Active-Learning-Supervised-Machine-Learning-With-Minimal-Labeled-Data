from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
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
    al_decision = {'vote_entropy':[]}
    passive_results = []
    mnist = fetch_mldata('MNIST original')    

    random_classifier=[LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=1000) for i in range(10)]
    active_classifier = [LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=1000) for i in range(10)]

    k = 0
    decision = []
    passive_results = []

    x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target)
    
    labeled_data, X_unlabeled, available_label, oracle_label_results = train_test_split(x_train,y_train,test_size = 0.99)

    for model in random_classifier:
        model.classes_ = np.arange(10)
        model.fit(labeled_data, available_label)
        
    for model in active_classifier:
        model.classes_ = np.arange(10)
        model.fit(labeled_data, available_label)
        
        
    sample_size = len(labeled_data)
    batch_size = 32
    new_samples_record = [i*batch_size for i in range(1,30) ]
        
    for i in range(0, len(new_samples_record)):
        sample_size = sample_size +  new_samples_record[i]
            
    labeled_data_rand = labeled_data
    available_label_rand = available_label
    labeled_data_active = labeled_data
    available_label_active = available_label

    for num_new_samples in new_samples_record:
        num_samples.append(num_new_samples)
        random_queries = np.random.choice(unlabeled_data.shape[0], num_new_samples, replace=Fal_objse)
        labeled_data_rand = np.concatenate((labeled_data_rand, unlabeled_data[random_queries, :]))
        available_label_rand = np.concatenate((available_label_rand, oracle_label[random_queries]))
        preds = []
        for model in random_classifier:
            model.fit(labeled_data_rand, available_label_rand)
            preds.append(model.predict(X_test))

        prediction_stack = np.stack(preds)
        decision = np.apply_al_objong_axis(\
            lambda x: Counter(x).most_common()[0][0],\
            0, prediction_stack)
        passive_results.append(np.sum(decision == y_test) / np.shape(X_test)[0])

        for model in active_classifier:
            model.classes_ = np.arange(10)
            indexes = al_obj.rank(active_classifier, unlabeled_data, num_new_samples)
            labeled_data_active = np.concatenate((labeled_data_active, unlabeled_data[indexes, :]))
            available_label_active = np.concatenate((available_label_active, oracle_label[indexes]))

            preds = []

        for model in active_classifier:
            model.fit(labeled_data_active, available_label_active)
            curr_pred = model.predict(X_test)
            preds.append(curr_pred)
                
        prediction_stack = np.stack(preds)
        decision = np.apply_along_axis(\
            lambda x:Counter(x).most_common()[0][0],0,\
            prediction_stack)
        matches = np.sum(decision == y_test)
        al_decision[strategy].append(matches/np.shape(X_test)[0])
                
        k = k + 1
    
    np.savetxt('./misc/random_model_accuracy.txt', passive_results)
    np.savetxt('./misc/active_model_accuracy.txt', al_decision)
    
   
if __name__ == '__main__':
    main()
      
