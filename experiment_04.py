import pandas
import os
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn_lvq import GlvqModel  # https://mrnuggelz.github.io/sklearn-lvq/index.html
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from column_extractor import ColumnExtractor


"""
------------------------------------------------------------------------------------------------------------------------
Experiment 4
The script combines the outputs of 25 classifiers creating the set of combinations iteratively.
we start with a classifier, add a second and calculate the performance, then add a third and so on.

Each of the 25 classifiers is trained on a different task
For each classification models, the classifiers are stored in the classifiers list. 
They are sorted by the task to it refers (i.e. the first classifier in the list works on task 1, the second in the list works on task 2, ...

the ranking_accuracy list contains the order of accuracy of the classifiers:
for example: ranking_accuracy = [4, 5, 15, ...]
first consider the task in position 4 of the classifiers list, to this then add the one in position 5, then that
in position 15 ... and so on. The values ​​are the indices of the list, so they start from 0!

The script input is the CSV file which contains all 25 tasks!
------------------------------------------------------------------------------------------------------------------------
"""

print("-- Experiment 04 --")

random_state = 100

NUM_FEATURE = 18

ranFor = True
logReg = True
knn = True
LDA = True
GNB = True
SVM = True
SVM2 = True
DECISION_TREE = True
MLP = True
LVQ = True
BFT = True

names = []
all_classifiers = {}
order_tasks = {}
######################
# processing data
dataset_path = "data/task_ALL.csv"
output_path = "out/experiment4"

if not os.path.isdir(output_path):
    os.mkdir(output_path)

with open(dataset_path) as file:
    line = file.readline().strip()
    features_names = list(line.split(","))
    data = pandas.read_csv(file, names=features_names, index_col=0)

dataset_values = data.values
X = dataset_values[:, 0:len(features_names) - 2]  # -2
y = dataset_values[:, len(features_names) - 2]  # -2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

classifiers = []
if ranFor:
    # RF
    name = "Random_Forest"
    classifiers = []
    classifiers.append(RandomForestClassifier(max_depth=5, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=8, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=7, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=9, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=8, n_estimators=300, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=3, n_estimators=250, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=6, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=5, n_estimators=250, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=9, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=7, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=9, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=3, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=3, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=3, n_estimators=250, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=6, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=6, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=6, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=5, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=3, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=7, n_estimators=200, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=5, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=5, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=3, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=9, n_estimators=250, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=5, random_state=100))

    names.append(name)
    all_classifiers[name] = classifiers
    order_tasks[name] = [22, 16, 20, 21, 18, 24, 4, 6, 8, 13, 14, 19, 15, 23, 5, 3, 9, 7, 10, 1, 17, 0, 12, 2, 11]
if logReg:
    # LR
    name = "Logistic_Regression"
    classifiers = []
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.5, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=10, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.1, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.05, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.5, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.05, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.01, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.001, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=10, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.01, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=10, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=10, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=10, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.1, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.5, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.5, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.1, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.05, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.005, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.05, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.1, random_state=100))]))

    names.append(name)
    all_classifiers[name] = classifiers
    order_tasks[name] = [6, 22, 20, 18, 7, 16, 23, 24, 14, 15, 2, 17, 21, 5, 8, 9, 4, 19, 12, 1, 0, 11, 10, 13, 3]
if knn:
    # KNN
    name = "KNN"
    classifiers = []
    classifiers.append(KNeighborsClassifier(n_neighbors=11))
    classifiers.append(KNeighborsClassifier(n_neighbors=9))
    classifiers.append(KNeighborsClassifier(n_neighbors=20))
    classifiers.append(KNeighborsClassifier(n_neighbors=20))
    classifiers.append(KNeighborsClassifier(n_neighbors=8))
    classifiers.append(KNeighborsClassifier(n_neighbors=8))
    classifiers.append(KNeighborsClassifier(n_neighbors=9))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(KNeighborsClassifier())
    classifiers.append(KNeighborsClassifier())
    classifiers.append(KNeighborsClassifier(n_neighbors=10))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(KNeighborsClassifier())
    classifiers.append(KNeighborsClassifier(n_neighbors=8))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(KNeighborsClassifier())
    classifiers.append(KNeighborsClassifier())
    classifiers.append(KNeighborsClassifier())
    classifiers.append(KNeighborsClassifier())
    classifiers.append(KNeighborsClassifier(n_neighbors=9))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(KNeighborsClassifier(n_neighbors=11))
    classifiers.append(KNeighborsClassifier(n_neighbors=11))
    classifiers.append(KNeighborsClassifier(n_neighbors=10))
    classifiers.append(KNeighborsClassifier(n_neighbors=8))

    names.append(name)
    all_classifiers[name] = classifiers
    order_tasks[name] = [18, 16, 22, 20, 21, 24, 15, 8, 3, 4, 6, 13, 5, 14, 2, 1, 23, 17, 11, 7, 19, 12, 9, 10, 0]
if LDA:
    # LDA
    name = "LDA"
    classifiers = []
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(LinearDiscriminantAnalysis())

    names.append(name)
    all_classifiers[name] = classifiers
    order_tasks[name] = [6, 16, 18, 20, 7, 1, 24, 22, 17, 8, 21, 2, 23, 14, 19, 12, 11, 5, 15, 9, 4, 13, 0, 10, 3]
if GNB:
    # GaussianNB
    name = "Gaussian_NB"
    classifiers = []
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())
    classifiers.append(GaussianNB())

    names.append(name)
    all_classifiers[name] = classifiers
    order_tasks[name] = [6, 22, 15, 16, 4, 2, 14, 5, 20, 23, 12, 24, 9, 21, 19, 18, 17, 10, 13, 11, 8, 7, 3, 1, 0]
if SVM:
    # SVM
    name = "SVM"
    classifiers = []
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.01, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.5, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.05, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=1, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.1, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.001, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.5, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=10, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.05, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=1, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=1, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=10, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.05, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=1, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.001, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=1, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.01, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=10, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.5, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.05, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=1, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.5, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.005, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=0.01, loss='hinge', max_iter=50000, random_state=100))]))
    classifiers.append(Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=10, loss='hinge', max_iter=50000, random_state=100))]))

    names.append(name)
    all_classifiers[name] = classifiers
    order_tasks[name] = [6, 22, 20, 18, 7, 16, 15, 17, 14, 24, 2, 21, 8, 23, 5, 19, 9, 12, 4, 1, 13, 0, 11, 3, 10]
if DECISION_TREE:
    # Decision Tree
    name = "Decision_Tree"
    classifiers = []
    classifiers.append(DecisionTreeClassifier(criterion='entropy', max_depth=6, max_leaf_nodes=15, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(max_leaf_nodes=9, min_samples_leaf=5, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(min_weight_fraction_leaf=0.3, random_state=100))
    classifiers.append(DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=4, min_samples_leaf=15, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(min_samples_leaf=25, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=6, min_samples_leaf=2, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10, min_samples_leaf=2, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(min_weight_fraction_leaf=0.4, random_state=100))
    classifiers.append(DecisionTreeClassifier(min_weight_fraction_leaf=0.3, random_state=100))
    classifiers.append(DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf=2, min_samples_split=5, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(max_leaf_nodes=5, min_samples_leaf=15, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(min_weight_fraction_leaf=0.4, random_state=100))
    classifiers.append(DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=8, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(criterion='entropy', min_weight_fraction_leaf=0.4, random_state=100))
    classifiers.append(DecisionTreeClassifier(max_depth=4, max_leaf_nodes=6, min_samples_leaf=4, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=5, min_samples_leaf=5, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(max_leaf_nodes=8, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(min_weight_fraction_leaf=0.2, random_state=100))
    classifiers.append(DecisionTreeClassifier(max_leaf_nodes=9, min_samples_leaf=4, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(criterion='entropy', max_depth=3, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(max_leaf_nodes=15, min_samples_split=5, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(min_weight_fraction_leaf=0.2, random_state=100))
    classifiers.append(DecisionTreeClassifier(max_depth=3, max_leaf_nodes=5, min_samples_leaf=5, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(DecisionTreeClassifier(max_leaf_nodes=4, min_weight_fraction_leaf=0, random_state=100))

    names.append(name)
    all_classifiers[name] = classifiers
    order_tasks[name] = [22, 15, 16, 21, 14, 4, 18, 6, 5, 8, 10, 1, 20, 24, 19, 23, 13, 2, 7, 17, 3, 12, 11, 0, 9]
if MLP:
    # MLP
    name = "MLP"
    classifiers = []
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='tanh', hidden_layer_sizes=500, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=500, learning_rate_init=0.025, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='logistic', hidden_layer_sizes=500, learning_rate_init=0.025, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='logistic', hidden_layer_sizes=400, learning_rate_init=0.05, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=1000, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='tanh', hidden_layer_sizes=700, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='logistic', hidden_layer_sizes=300, learning_rate_init=0.05, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='tanh', hidden_layer_sizes=700, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='logistic', hidden_layer_sizes=1000, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='tanh', hidden_layer_sizes=700, learning_rate_init=0.05, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ( 'clf', MLPClassifier(hidden_layer_sizes=200, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='logistic', hidden_layer_sizes=1000, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='tanh', hidden_layer_sizes=500, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ( 'clf', MLPClassifier(hidden_layer_sizes=1000, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='logistic', hidden_layer_sizes=300, learning_rate_init=0.01, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='logistic', hidden_layer_sizes=1000, learning_rate_init=0.005, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='tanh', hidden_layer_sizes=1000, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='tanh', hidden_layer_sizes=200, learning_rate_init=0.05, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='logistic', hidden_layer_sizes=700, learning_rate_init=0.01, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=500, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(hidden_layer_sizes=300, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='logistic', hidden_layer_sizes=100, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='logistic', hidden_layer_sizes=200, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='tanh', hidden_layer_sizes=1000, learning_rate_init=0.1, max_iter=10000, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='logistic', hidden_layer_sizes=700, max_iter=10000, random_state=100))]))

    names.append(name)
    all_classifiers[name] = classifiers
    order_tasks[name] = [20, 16, 15, 6, 17, 21, 13, 22, 14, 8, 19, 7, 4, 23, 2, 24, 18, 3, 5, 12, 1, 9, 10, 0, 11]
if LVQ:
    # LVQ
    name = "LVQ"
    classifiers = []
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(gtol=1e-06, prototypes_per_class=2, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=40, gtol=1e-06, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=3, gtol=1e-06, prototypes_per_class=5, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=45, gtol=1e-06, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=50, gtol=1e-06, prototypes_per_class=30, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(gtol=1e-06, prototypes_per_class=30, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=15, gtol=1e-06, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=30, gtol=1e-06, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=4, gtol=1e-06, prototypes_per_class=2, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=7, gtol=1e-06, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=4, gtol=1e-06, prototypes_per_class=35, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=50, gtol=1e-06, prototypes_per_class=20, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=20, gtol=1e-06, prototypes_per_class=2, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=8, gtol=1e-06, prototypes_per_class=4, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=10, gtol=1e-06, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=50, gtol=1e-06, prototypes_per_class=3, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=45, gtol=1e-06, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=30, gtol=1e-06, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=50, gtol=1e-06, prototypes_per_class=35, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=5, gtol=1e-06, prototypes_per_class=5, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(gtol=1e-06, prototypes_per_class=40, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(gtol=1e-06, prototypes_per_class=3, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=40, gtol=1e-06, prototypes_per_class=5, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=10, gtol=1e-06, prototypes_per_class=4, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', GlvqModel(beta=50, gtol=1e-06, random_state=100))]))

    names.append(name)
    all_classifiers[name] = classifiers
    order_tasks[name] = [22, 15, 20, 16, 6, 23, 14, 18, 17, 24, 2, 7, 21, 13, 19, 3, 4, 8, 5, 12, 1, 10, 9, 0, 11]
if BFT:
    # Best for task
    name = "BFT"
    classifiers = []
    classifiers.append(RandomForestClassifier(max_depth=5, random_state=100))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=10, random_state=100))]))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.1, random_state=100))]))
    classifiers.append(RandomForestClassifier(max_depth=9, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=8, n_estimators=300, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=3, n_estimators=250, random_state=100))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=1, random_state=100))]))
    classifiers.append(LinearDiscriminantAnalysis())
    classifiers.append(RandomForestClassifier(max_depth=9, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=7, n_estimators=150, random_state=100))
    classifiers.append(GaussianNB())
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=10, random_state=100))]))
    classifiers.append(RandomForestClassifier(max_depth=3, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=3, n_estimators=250, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=6, n_estimators=150, random_state=100))
    classifiers.append(DecisionTreeClassifier(max_depth=4, max_leaf_nodes=6, min_samples_leaf=4, min_weight_fraction_leaf=0, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=6, random_state=100))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', MLPClassifier(activation='tanh', hidden_layer_sizes=200, learning_rate_init=0.05, max_iter=10000, random_state=100))]))
    classifiers.append(RandomForestClassifier(max_depth=3, n_estimators=150, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=7, n_estimators=200, random_state=100))
    classifiers.append(RandomForestClassifier(max_depth=5, random_state=100))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=5, n_estimators=150, random_state=100))
    classifiers.append(DecisionTreeClassifier(min_weight_fraction_leaf=0.2, random_state=100))
    classifiers.append(Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(C=0.05, random_state=100))]))
    classifiers.append(RandomForestClassifier(bootstrap=False, max_depth=5, random_state=100))

    names.append(name)
    all_classifiers[name] = classifiers
    order_tasks[name] = [22, 16, 20, 21, 6, 18, 24, 15, 4, 17, 8, 13, 14, 23, 5, 3, 10, 1, 7, 9, 19, 2, 0, 12, 11]

######################

for name in names:
    print(name)

    # train the classifier for each task
    all_pipe = []
    for ind in range(25):
        pipe = Pipeline([
            ('col_extract', ColumnExtractor(cols=range((NUM_FEATURE * ind), NUM_FEATURE * (ind + 1)))),
            ('clf', all_classifiers[name][ind]),
        ])
        pipe.fit(X_train, y_train)
        #print("score pipe", ind + 1, ": ", pipe.score(X_test, y_test))
        all_pipe.append(pipe)

    ###########################################################

    estimators_i = []
    weights_i = []

    out_current_file_path = os.path.join(output_path, name+ ".txt")
    out_current_file = open(out_current_file_path, "w")

    ind = 0
    for clf_ind in tqdm(order_tasks[name]):
        estimators_i.append(("clf" + str(ind + 1), all_pipe[clf_ind]))
        weights_i.append(1)

        eclf = VotingClassifier(estimators=estimators_i, voting='hard', weights=weights_i)
        eclf.fit(X_train, y_train)
        score = eclf.score(X_test, y_test)

        # Print Results
        predictions = eclf.predict(X_test)
        confusion_m = confusion_matrix(y_test, predictions)

        out_current_file.write("\nIndex: " + str(ind + 1) + "----------------------")
        out_current_file.write("\nTask: " + str(clf_ind + 1))
        out_current_file.write("\nAccuracy:\n" + str(score))
        out_current_file.write("\n\nReport:\n" + str(classification_report(y_test, predictions)))
        out_current_file.write("\n\nConfuzion Matrix:\n")
        out_current_file.write("    \tpredA\t\tpredP\n"
            "actA\t" + str(confusion_m[0][0]) + "\t\t" + str(confusion_m[0][1]) + "\n"
            "actP\t" + str(confusion_m[1][0]) + "\t\t" + str(confusion_m[1][1]) + "\n")
        """
        print("Index: ", ind + 1, "----------------------")
        print("Task: ", clf_ind + 1)
        print("\nAccuracy: ", score)
        print(classification_report(y_test, predictions))
        print("    \tpredA\t\tpredP\n"
              "actA\t", confusion_m[0][0], "\t\t", confusion_m[0][1], "\n"
              "actP\t", confusion_m[1][0], "\t\t", confusion_m[1][1], "\n")
        """
        ind += 1
    out_current_file.close()
