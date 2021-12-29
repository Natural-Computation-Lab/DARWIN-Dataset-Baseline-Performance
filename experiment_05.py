import os
import datetime
import pandas
import collections
import copy
from tasks_extractor import TasksExtractor
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn_lvq import GlvqModel
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus.tables import Table
from reportlab.platypus import PageBreak

"""
------------------------------------------------------------------------------------------------------------------------
The script checks the following classification algorithms:
Random forest, logistic regression, Knn, LDA, Gaussian NB, SVM, decision tree, MLP and LVQ

The input is a .csv file that contains the features. It is a general file, it can consist 
of the data of a single activity or of several activities. Thi input file is placed in the 
folder 'data'

The output is a PDF file that summarizes the classification results for each algorithm for 
20 different runs. Output file is saved in the 'out' folder.

The experiment is similar to Experiment 1.
The difference is in the dataset used for each classification method. In this case, the 
dataset comprises features extracted from a sub-set of tasks, and the sub-set depends on
the classification method in use. To change the subset for each classification method, you 
need to modify the value of the dictionary "task_to_combine". The key of the dictionary is
the name of the classification  method, while the value is a list of integer representing 
the task to include into the  sub-set:

    task_to_combine["method_name"] = ["List of task to select..."]

------------------------------------------------------------------------------------------------------------------------
"""

print("-- Experiment 05 --")

NUM_FEATURE = 18

# -- The following flags enable classifiers
ranFor = False
logReg = False
knn = False
LDA = False
GNB = False
SVM = True
DECISION_TREE = False
MLP = False
LVQ = False

# dataset parameters
test_ratio = 0.2  # ratio training - test
random_states = [42, 43, 10, 28, 36, 98, 75, 9, 53, 62, 8, 32, 84, 22, 54, 82, 15, 90, 30, 12]


# classification parameters
cross_val = 5
f_select_threshold = 0.05


def add_modelresults_to_report(name, scoring, grid_search, score_on_cv_training, y_test, predictions):
    if grid_search is None:
        best_params = ""
        best_score = 0
    else:
        best_params = str(grid_search.best_params_)
        best_score = grid_search.best_score_

    # story.append(PageBreak())
    par = '<font size = 24>%s</font>' % name
    story.append(Spacer(1, 20))
    story.append(Paragraph(par, styles["Normal"]))

    par = '<font size = 12>' \
          'Scoring: %s<br/><br/>' \
          'Grid Search parameters tuning:<br/>' \
          'Best parameters: %s<br/>' \
          'Best score: %f<br/>' \
          'Score on all Folds: %s<br/>' \
          '<br/>' \
          'On test set<br/>' \
          'Score: %f<br/>' \
          'Confusion matrix:' \
          '</font>' % (scoring, best_params, best_score, score_on_cv_training, accuracy_score(y_test, predictions))

    story.append(Spacer(1, 15))
    story.append(Paragraph(par, styles["Normal"]))

    confusion_m = confusion_matrix(y_test, predictions)
    conf_m = [("", "A", "P"), ("A", confusion_m[0][0], confusion_m[0][1]), ("P", confusion_m[1][0], confusion_m[1][1])]
    table = Table(conf_m, colWidths=20, rowHeights=20)
    story.append(Spacer(1, 5))
    story.append(table)

    par = '<font size = 12>' \
          'Report:' \
          '</font> '
    story.append(Spacer(1, 5))
    story.append(Paragraph(par, styles["Normal"]))

    report = classification_report(y_test, predictions)
    supp_a = confusion_m[0][0] + confusion_m[1][0]
    p_a = confusion_m[0][0] / supp_a
    r_a = confusion_m[0][0] / (confusion_m[0][0] + confusion_m[0][1])
    f1_a = 2 * (p_a * r_a) / (p_a + r_a)

    supp_p = confusion_m[0][1] + confusion_m[1][1]
    p_p = confusion_m[1][1] / supp_p
    r_p = confusion_m[1][1] / (confusion_m[1][1] + confusion_m[1][0])
    f1_p = 2 * (p_p * r_p) / (p_p + r_p)

    supp_all = supp_a + supp_p
    accuracy = (confusion_m[0][0] + confusion_m[1][1]) / supp_all

    macro_avg_p = (p_a + p_p) / 2
    macro_avg_r = (r_a + r_p) / 2
    macro_avg_f1 = (f1_a + f1_p) / 2

    weighted_avg_p = (p_a * supp_a + p_p * supp_p) / supp_all
    weighted_avg_r = (r_a * supp_a + r_p * supp_p) / supp_all
    weighted_avg_f1 = (f1_a * supp_a + f1_p * supp_p) / supp_all

    report_m = [("", "precision", "recall", "f1-score", "support"),
                ("A", round(p_a, 2), round(r_a, 2), round(f1_a, 2), supp_a),
                ("P", round(p_p, 2), round(r_p, 2), round(f1_p, 2), supp_p),
                ("", "", "", "", ""),
                ("accuracy", "", "", round(accuracy, 2), supp_all),
                ("macro avg", round(macro_avg_p, 2), round(macro_avg_r, 2), round(macro_avg_f1, 2), supp_all),
                ("weighted avg", round(weighted_avg_p, 2), round(weighted_avg_r, 2), round(weighted_avg_f1, 2),
                 supp_all)]
    table = Table(report_m, colWidths=100, rowHeights=20)
    story.append(Spacer(1, 5))
    story.append(table)


mod_alg = "output"
names = []

task_to_combine = {}

if ranFor:
    name = "RandomForset"
    names.append(name)
    task_to_combine[name] = [22, 16, 20, 21, 18]
elif logReg:
    name = "LogRegression"
    names.append(name)
    task_to_combine[name] = [6,22,20,18,7]
elif knn:
    name = "KNN"
    names.append(name)
    task_to_combine[name] = [18,16,22,20,21,24,15,8,3,4]

elif LDA:
    name = "LDA"
    names.append(name)
    task_to_combine[name] = [6,16,18,20,7,1]
elif GNB:
    name = "GNB"
    names.append(name)
    task_to_combine[name] = [6,22,15,16,4,2,14,5,20]
elif SVM:
    name = "SVM"
    names.append(name)
    task_to_combine[name] = [6,22,20,18,7]
elif DECISION_TREE:
    name = "Decision tree"
    names.append(name)
    task_to_combine[name] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
elif MLP:
    name = "MLP"
    names.append(name)
    task_to_combine[name] = [20,16,15,6,17,21,13,22,14,8,19,7,4,23,2]
elif LVQ:
    name = "LVQ"
    names.append(name)
    task_to_combine[name] = [22,15,20,16,6,23,14,18,17]



dataset_path = "data/task_ALL.csv"
print("Path dataset: ", dataset_path)

output_path_base = "out/experiment5"
if not os.path.isdir(output_path_base):
    os.mkdir(output_path_base)

with open(dataset_path) as file:
    line = file.readline().strip()
    features_names = list(line.split(","))
    print(features_names)
    # par = '<font size = 12>Features: %s</font>' % str(features_names[1:-2])
    # story.append(Paragraph(par, styles["Normal"]))
    dataset = pandas.read_csv(file, names=features_names, index_col=0)

for mod_alg in names:
    output_path = os.path.join(output_path_base, mod_alg + ".pdf")
    output_path_txt = os.path.dirname(output_path)
    output_path_txt = os.path.join(output_path_txt, mod_alg + ".txt")
    print("Path outfile pdf: ", output_path)
    print("Path outfile txt: ", output_path_txt)

    doc = SimpleDocTemplate(output_path,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    story = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

    story.append(Paragraph(str(datetime.datetime.now()), styles["Normal"]))
    story.append(Spacer(10, 50))

    #

    par = '<font size = 12>Dataset shape:<br/>N elements and attributes: %s</font>' % str(dataset.shape)
    story.append(Spacer(1, 12))
    story.append(Paragraph(par, styles["Normal"]))

    par = '<font size = 12>Class Distribution.<br/>%s</font>' % str(dataset.groupby('class').size())
    story.append(Spacer(1, 12))
    story.append(Paragraph(par, styles["Normal"]))

    file_txt = open(output_path_txt, 'a')

    for random_state in random_states:
        # -----------------------------------------------------------------------------------------------------------
        # --- DIVIDING DATASET  -------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------
        story.append(PageBreak())
        par = '<font size = 24>Dataset division</font>'
        story.append(Spacer(1, 20))
        story.append(Paragraph(par, styles["Normal"]))
        # random_state = random.randint(1, 100)
        dataset_values = dataset.values
        X = dataset_values[:, 0:len(features_names) - 2]  # -2
        Y = dataset_values[:, len(features_names) - 2]  # -2

        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_ratio,
                                                                            random_state=random_state, stratify=Y)
        counter_train = collections.Counter(y_train)
        counter_test = collections.Counter(y_test)
        par = '<font size = 12>' \
              'test_ratio: %f<br/>' \
              'random_state: %d<br/><br/>' \
              'Element in train: %d<br/>' \
              '    %s<br/>' \
              'Element in test: %d<br/>' \
              '    %s<br/>' \
              '</font>' % (test_ratio, random_state, len(x_train), str(counter_train), len(x_test), str(counter_test))
        story.append(Spacer(1, 15))
        story.append(Paragraph(par, styles["Normal"]))

        # ------------------------------------------------------------------------------------------------------------
        # --- CLASSIFICATION    -------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------

        # - RANDOM FOREST -------------------------------------------------------------------------------------------
        if ranFor:
            print("Random Forest...")
            scoring = 'accuracy'
            name = 'Random Forest'
            model = RandomForestClassifier(random_state=random_state)
            model = Pipeline([
                ('tasks_extractor', TasksExtractor(task_to_combine[mod_alg], NUM_FEATURE)),
                ('clf', model)
            ])

            print("    Grid search parameters tuning...")
            #
            param_grid = [{'clf__n_estimators': [100, 150, 200, 250, 300], 'clf__max_depth': [3, 4, 5, 6, 7, 8, 9]},
                          {'clf__bootstrap': [False], 'clf__n_estimators': [100, 150, 200, 250, 300],
                           'clf__max_depth': [3, 4, 5, 6, 7, 8, 9]}]

            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)

            score_on_cv_training = grid_search.cv_results_['mean_test_score']

            best_model = grid_search.best_estimator_
            print("        ", best_model)

            # on test set
            predictions = best_model.predict(x_test)
            print("Report")
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))

            add_modelresults_to_report(name, scoring, grid_search, score_on_cv_training, y_test, predictions)
            file_txt.write("Accuracy\n" + str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.write("Confusion Mtrix:\n" + str(confusion_matrix(y_test, predictions)) + "\n\n")
            story_2 = copy.deepcopy(story)
            doc.build(story_2)

        # - knn ----------------------------------------------------------------------------------
        if knn:
            print("KNN...")
            scoring = 'accuracy'
            name = 'KNN'
            model = KNeighborsClassifier()
            model = Pipeline([
                ('tasks_extractor', TasksExtractor(task_to_combine[mod_alg], NUM_FEATURE)),
                ('clf', model)
            ])

            print("    Grid search parameters tuning...")
            #
            #param_grid = [{'clf__n_neighbors': [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25]}]
            param_grid = [{'clf__n_neighbors': [5, 8, 9, 10, 11, 20]}]
            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)

            score_on_cv_training = grid_search.cv_results_['mean_test_score']

            best_model = grid_search.best_estimator_
            print("        ", best_model)

            # on test set
            predictions = best_model.predict(x_test)
            print("Report")
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))

            add_modelresults_to_report(name, scoring, grid_search, score_on_cv_training, y_test, predictions)
            file_txt.write("Accuracy\n" + str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.write("Confusion Mtrix:\n" + str(confusion_matrix(y_test, predictions)) + "\n\n")
            story_2 = copy.deepcopy(story)
            doc.build(story_2)

        # - LDA ---------------------------------------------------------------------------------
        if LDA:
            print("LDA...")
            scoring = 'accuracy'
            name = 'LDA'
            model = LinearDiscriminantAnalysis()
            best_model = Pipeline([
                ('tasks_extractor', TasksExtractor(task_to_combine[mod_alg], NUM_FEATURE)),
                ('clf', model)
            ])

            kfold = model_selection.KFold(n_splits=cross_val, random_state=random_state, shuffle=True)
            for train_index, test_index in kfold.split(x_train):
                X_train_cv, X_test_cv_cv, y_train_cv, y_test_cv = x_train[train_index], x_train[test_index], y_train[
                    train_index], y_train[test_index]
                best_model.fit(X_train_cv, y_train_cv)
            best_model.fit(X_train_cv, y_train_cv)

            # on test set
            predictions = best_model.predict(x_test)
            print("Report")
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))

            add_modelresults_to_report(name, scoring, None, None, y_test, predictions)
            file_txt.write("Accuracy\n" + str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.write("Confusion Mtrix:\n" + str(confusion_matrix(y_test, predictions)) + "\n\n")
            story_2 = copy.deepcopy(story)
            doc.build(story_2)

        # - Gaussian NB ---------------------------------------------------------------------------------
        if GNB:
            print("Gaussian NB...")
            scoring = 'accuracy'
            name = 'Gaussian NB'
            model = GaussianNB()
            best_model = Pipeline([
                ('tasks_extractor', TasksExtractor(task_to_combine[mod_alg], NUM_FEATURE)),
                ('clf', model)
            ])

            kfold = model_selection.KFold(n_splits=cross_val, random_state=random_state, shuffle=True)
            cv_results = model_selection.cross_val_score(best_model, x_train, y_train, cv=kfold, scoring=scoring)
            for train_index, test_index in kfold.split(x_train):
                X_train_cv, X_test_cv_cv, y_train_cv, y_test_cv = x_train[train_index], x_train[test_index], y_train[
                    train_index], y_train[test_index]
                best_model.fit(X_train_cv, y_train_cv)
            best_model.fit(x_train, y_train)

            # on test set
            predictions = best_model.predict(x_test)
            print("Report")
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))

            add_modelresults_to_report(name, scoring, None, None, y_test, predictions)
            file_txt.write("Accuracy\n" + str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.write("Confusion Mtrix:\n" + str(confusion_matrix(y_test, predictions)) + "\n\n")
            story_2 = copy.deepcopy(story)
            doc.build(story_2)

        # - LOGISTIC REGRESSION ----------------------------------------------------------------------------------
        if logReg:
            print("Logistic Regression...")
            scoring = 'accuracy'
            name = 'Logistic Regression'
            model = Pipeline([("tasks_extractor", TasksExtractor(task_to_combine[mod_alg], NUM_FEATURE)),
                              ("scaler", StandardScaler()),
                              ("clf", LogisticRegression(random_state=random_state))])

            print("    Grid search parameters tuning...")
            #
            param_grid = [{'clf__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]}]
            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)

            score_on_cv_training = grid_search.cv_results_['mean_test_score']

            best_model = grid_search.best_estimator_
            print("        ", best_model)

            # on test set
            predictions = best_model.predict(x_test)
            print("Report")
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))

            add_modelresults_to_report(name, scoring, grid_search, score_on_cv_training, y_test, predictions)
            file_txt.write("Accuracy\n" + str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.write("Confusion Mtrix:\n" + str(confusion_matrix(y_test, predictions)) + "\n\n")
            story_2 = copy.deepcopy(story)
            doc.build(story_2)

        # - SVM -------------------------------------------------------------------------------------------
        if SVM:
            print("SVM...")
            scoring = 'accuracy'
            name = 'SVM'

            model = Pipeline([("tasks_extractor", TasksExtractor(task_to_combine[mod_alg], NUM_FEATURE)),
                              ("scaler", StandardScaler()),
                              ("linear_svm", LinearSVC(loss="hinge", max_iter=50000, random_state=random_state))])

            print("    Grid search parameters tuning...")
            param_grid = [{'linear_svm__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]}]
            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)

            score_on_cv_training = grid_search.cv_results_['mean_test_score']

            best_model = grid_search.best_estimator_
            print("        ", best_model)

            # on test set
            predictions = best_model.predict(x_test)
            print("Report")
            print(classification_report(y_test, predictions))

            add_modelresults_to_report(name, scoring, grid_search, score_on_cv_training, y_test, predictions)
            file_txt.write("Accuracy\n" + str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.write("Confusion Mtrix:\n" + str(confusion_matrix(y_test, predictions)) + "\n\n")
            story_2 = copy.deepcopy(story)
            doc.build(story_2)

        # - DECISION TREE -------------------------------------------------------------------------------------------
        if DECISION_TREE:
            print("Decision Tree")
            scoring = 'accuracy'
            name = 'Decision Tree'

            model = DecisionTreeClassifier(random_state=random_state)

            model = Pipeline([
                ('tasks_extractor', TasksExtractor(task_to_combine[mod_alg], NUM_FEATURE)),
                ('clf', model)
            ])

            print("    Grid search parameters tuning...")

            #
            param_grid = [{'clf__criterion': ["gini", "entropy"],
                           'clf__max_depth': [None, 2, 3, 4, 5, 6, 7, 8],
                           'clf__min_samples_split': [2, 3, 4, 5],
                           'clf__min_samples_leaf': [1, 2, 3, 4, 5, 10, 15, 20, 25],
                           'clf__min_weight_fraction_leaf': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                           'clf__max_leaf_nodes': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
                           },
                          ]

            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)

            score_on_cv_training = grid_search.cv_results_['mean_test_score']

            best_model = grid_search.best_estimator_
            print("        ", best_model)

            # on test set
            predictions = best_model.predict(x_test)
            print("Report")
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))

            add_modelresults_to_report(name, scoring, grid_search, score_on_cv_training, y_test, predictions)
            file_txt.write("Accuracy\n" + str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.write("Confusion Mtrix:\n" + str(confusion_matrix(y_test, predictions)) + "\n\n")
            story_2 = copy.deepcopy(story)
            doc.build(story_2)

        # - MLP  ---------------------------------------------------------------------------------------
        if MLP:
            print("MLP")
            scoring = 'accuracy'
            name = 'Multi-layer Perceptron'

            model = Pipeline([
                ('tasks_extractor', TasksExtractor(task_to_combine[mod_alg], NUM_FEATURE)),
                ("scaler", StandardScaler()),
                ('clf', MLPClassifier(random_state=random_state)),
            ])

            print("    Grid search parameters tuning...")
            #

            param_grid = [{'clf__solver': ["adam"],
                           'clf__hidden_layer_sizes': [100, 200, 300, 400, 500, 700, 1000],
                           'clf__activation': ["relu", "logistic", "tanh"],
                           'clf__alpha': [0.0001],
                           'clf__batch_size': ["auto"],
                           'clf__learning_rate_init': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
                           'clf__max_iter': [10000],
                           'clf__shuffle': [True],
                           },
                          ]

            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)

            score_on_cv_training = grid_search.cv_results_['mean_test_score']

            best_model = grid_search.best_estimator_
            print("        ", best_model)

            # on test set
            predictions = best_model.predict(x_test)
            print("Report")
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))

            add_modelresults_to_report(name, scoring, grid_search, score_on_cv_training, y_test, predictions)
            file_txt.write("Accuracy\n" + str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.write("Confusion Mtrix:\n" + str(confusion_matrix(y_test, predictions)) + "\n\n")
            story_2 = copy.deepcopy(story)
            doc.build(story_2)

        # - Learning Vector Quantization -----------------------------------------------------------------------
        if LVQ:
            print("LVQ")
            scoring = 'accuracy'
            name = 'Learning Vector Quantization'

            model = Pipeline([
                ('tasks_extractor', TasksExtractor(task_to_combine[mod_alg], NUM_FEATURE)),
                ("scaler", StandardScaler()),
                ('clf', GlvqModel(random_state=random_state)),
            ])

            print("    Grid search parameters tuning...")

            param_grid = [{'clf__prototypes_per_class': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                           'clf__max_iter': [2500],
                           'clf__gtol': [0.000001],
                           'clf__beta': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                           # 'clf__C': [[2, 3]],
                           },
                          ]

            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)

            score_on_cv_training = grid_search.cv_results_['mean_test_score']

            best_model = grid_search.best_estimator_
            print("        ", best_model)

            # on test set
            predictions = best_model.predict(x_test)
            print("Report")
            print(confusion_matrix(y_test, predictions))
            print(classification_report(y_test, predictions))

            add_modelresults_to_report(name, scoring, grid_search, score_on_cv_training, y_test, predictions)
            file_txt.write("Accuracy\n" + str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.write("Confusion Mtrix:\n" + str(confusion_matrix(y_test, predictions)) + "\n\n")
            story_2 = copy.deepcopy(story)
            doc.build(story_2)

    # -------------------------------------------------------------------------------------------------
    file_txt.close()