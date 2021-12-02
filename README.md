# Diagnosing Alzheimer’s disease from on-line handwriting: a benchmark dataset and baseline systems performance evaluation
## _The HAND dataset_


this git contains the simulation scripts to test the HAND dataset.

there is a python script for each experiment.

- expewriment_01.py
- expewriment_02.py
- expewriment_03.py


## Preliminary Installations
The scripts for the experiments are written in python. to run them you need to install python at the following version:
- python 3.8

Once python is installed, you need to install the necessary libraries. you can do this using the command:
```sh
pip install -r requirements.txt
```

## Experiment 1
The script checks the following classification algorithms:
 - Random forest (RF)
 - Logistic Regression (LR)
 - k nearest neighbor (KNN)
 - Linear Discriminant Analysis (LDA)
 - Gaussian Naive Bayes (GNB)
 - SVM
 - Decision Tree (DT)
 - Multi Layet Perceptron (MLP)
 - Learning Vector Quantization (LVQ)
 
In this experiment, the entire dataset is used to train classifiers, without considering any division by task.

The scripr reads a .csv file that contains the features. It is a general file, and can consist 
of the data of a single or of several tasks. 
The input file is placed in the folder 'data'

The script output is a PDF file that summarizes the classification results for each algorithm for 
20 different runs. Output file is saved in the 'out' folder.

The script begins with a series of flags, one for each classification algorithm. To select the algorithm you want to test, you need to set its flag to _True_.
For example, if you wanted to test only the Random Forest, you should set the flags as follows:
```sh
ranFor = True
logReg = False
knn = False
LDA = False
GNB = False
SVM = False
DECISION_TREE = False
MLP = False
LVQ = False
```

Once the relative flags have been set, the script can be launched by executing the following command:
```cmd
python experiment_01.py
```

## Experiment 2
The script checks the following classification algorithms:
 - Random forest (RF)
 - Logistic Regression (LR)
 - k nearest neighbor (KNN)
 - Linear Discriminant Analysis (LDA)
 - Gaussian Naive Bayes (GNB)
 - SVM
 - Decision Tree (DT)
 - Multi Layet Perceptron (MLP)
 - Learning Vector Quantization (LVQ)

In this experiment the dataset is divided, considering each time a single task on which each classifier is trained.

The input is a .csv file that contains the features. It is a general file, it can consist 
of the data of a single activity or of several activities. Thi input file is placed in the 
folder 'data'

The script output is stored in the "out" folder. As many folders are generated as runs are 
performed, each folder contains a pdf file with the classification results for each task.

The script begins with a series of flags, one for each classification algorithm. To select the algorithm you want to test, you need to set its flag to _True_.
For example, if you wanted to test only the Random Forest, you should set the flags as follows:
```sh
ranFor = True
logReg = False
knn = False
LDA = False
GNB = False
SVM = False
DECISION_TREE = False
MLP = False
LVQ = False
```

Once the relative flags have been set, the script can be launched by executing the following command:
```cmd
python experiment_02.py
```