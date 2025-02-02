# Diagnosing Alzheimer’s disease from on-line handwriting: a novel dataset and performance benchmarking
## _The DARWIN dataset (Diagnosis AlzheimeR WIth haNdwriting)_


this git contains the simulation scripts to test the HAND dataset.

there is a python script for each experiment.

- expewriment_01.py
- expewriment_02.py
- expewriment_03.py
- expewriment_04.py
- expewriment_05.py


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


## Experiment 3
The third experiment performs in the same way as experiment 2. The only difference is that this time only one run is performed on a new division of the dataset into test set and training set.

To run the esperiment, use the following command:
```cmd
python experiment_03.py
```


## Experiment 4
The fourth experiment combines the outputs of the classifiers trained on the individual tasks to obtain a single classification label.
The combination is done with a majority vote strategy

To run the esperiment, use the following command:
```cmd
python experiment_04.py
```

## Experiment 5
The experiment is similar to Experiment 1.
The difference is in the dataset used for each classification method. In this case, the 
dataset comprises features extracted from a sub-set of tasks, and the sub-set depends on
the classification method in use. To change the subset for each classification method, you 
need to modify the value of the dictionary "task_to_combine". The key of the dictionary is
the name of the classification  method, while the value is a list of integer representing 
the task to include into the  sub-set:

    task_to_combine["method_name"] = ["List of task to select..."]

To run the esperiment, use the following command:
```cmd
python experiment_05.py
```
