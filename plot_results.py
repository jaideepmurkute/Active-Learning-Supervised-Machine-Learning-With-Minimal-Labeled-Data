import numpy as np
import matplotlib.pyplot as plt

al_data = []
rand_data = []

with open("./misc/active_model_accuracy.txt",'r') as f_al:
    for row in f_al:
        al_data.append((float(row.rstrip('\n'))))
with open("./misc/random_model_accuracy.txt",'r') as f_rand:
    for row in f_rand:
        rand_data.append((float(row.rstrip('\n'))))
        
examples = [i*2 for i in range(1,32)]
examples_intervals = []
num_examples = 32

for i in range(1,31):
    examples_intervals.append(examples[i] + num_examples)
    num_examples = examples[i] + num_examples
    
num_examples = sum([i*32 for i in range(1,30)])
x_data = np.arange(0,num_examples)

plt.figure()
plt.plot(examples_intervals,al_data,'g',label='Entropy Sampling (Active Learner)')
#plt.plot(examples_intervals,rand_data,'r',label='Voted Entropy Sampling (Active Learner)')
plt.plot(examples_intervals,rand_data,'r',label="Random Sampling (Passive Learner)")

plt.ylabel("Model Accuracy")
plt.xlabel("Number of new data sample label batches requested from oracle")
plt.title("MNIST Digits dataset with Logistic Regression(one-vs-all)")
plt.legend()
plt.savefig("MNIST_Logistic_Regression.jpg")
plt.show()
