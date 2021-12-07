import datetime
import pandas
import os
import tkinter as tk
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn_lvq import GlvqModel  # https://mrnuggelz.github.io/sklearn-lvq/index.html

from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus.tables import Table
from reportlab.platypus import PageBreak
from column_extractor import ColumnExtractor

"""
------------------------------------------------------------------------------------------------------------------------
Experiment 2
The script tests the following classification algorithms:
Random forest, logistic regression, Knn, LDA, Gaussian NB, SVM, decision tree, MLP and LVQ


The input is a .csv file that contains the features. It is a general file, it can consist 
of the data of a single activity or of several activities. Thi input file is placed in the 
folder 'data'

The script output is stored in the "out" folder. As many folders are generated as runs are 
performed, each folder contains a pdf file with the classification results for each task.
------------------------------------------------------------------------------------------------------------------------
"""
print("-- Experiment 02 --")

NUM_FEATURE = 18

"I Seguenti flag abilitano i classificatori"
ranFor = False
logReg = False
knn = False
LDA = True
GNB = True
SVM = False
SVM2 = False
DECISION_TREE = False
MLP = False
LVQ = False

# dataset parameters
test_ratio = 0.2  # ratio training - test
random_state = 42
random_states = [42, 43, 10, 28, 36, 98, 75, 9, 53, 62, 8, 32, 84, 22, 54, 82, 15, 90, 30, 12]
# random_state = random.randint(1, 100)

# classification parameters
cross_val = 5
f_select_threshold = 0.05


def add_modelresults_to_report(name, scoring, grid_search, best_score, y_test, predictions, prediction_TS):
    if grid_search is None:
        best_params = ""
    else:
        best_params = str(grid_search.best_params_)

    story.append(PageBreak())
    par = '<font size = 24>%s</font>' % name
    story.append(Spacer(1, 20))
    story.append(Paragraph(par, styles["Normal"]))

    par = '<font size = 12>' \
          'Scoring: %s<br/><br/>' \
          'Grid Search parameters tuning:<br/>' \
          'best parameters: %s<br/>' \
          'best score: %f<br/>' \
          '<br/>' \
          'On test set<br/>' \
          'Score: %f<br/>' \
          'Confusion matrix:' \
          '</font>' % (scoring, best_params, best_score, accuracy_score(y_test, predictions))

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

    par = '<font size = 12>' \
          '<h2>PREDICTION ON TEST</h2>' \
          '<br/>Real values: <br/> %s <br/>' \
          '<br/>Predictions: <br/> %s' \
          '</font>' % (y_test, predictions)

    story.append(Spacer(1, 15))
    story.append(Paragraph(par, styles["Normal"]))

    par = '<font size = 12>' \
          '<h2>PREDICTION ON TRAINING</h2>' \
          '<br/>Predictions: <br/> %s' \
          '</font>' % (prediction_TS)

    story.append(Spacer(1, 15))
    story.append(Paragraph(par, styles["Normal"]))


#root = tk.Tk()
#root.withdraw()

# selezionare file dataset
dataset_path = "data/task_ALL.csv"
# dataset_path = filedialog.askopenfilename(title="Seleziona il file CSV del Dataset.",
#                                          filetypes=(('CSV files', '*.csv'), ("all files", "*.*")))
print("Path dataset: ", dataset_path)

# select out folder
output_path = "out"
# output_path = filedialog.askdirectory(title="Seleziona dove salvare il file PDF")
print("Path outfile: ", output_path)

with open(dataset_path) as file:
    line = file.readline().strip()
    features_names = list(line.split(","))
    # print(features_names)
    # par = '<font size = 12>Features: %s</font>' % str(features_names[1:-2])
    # story.append(Paragraph(par, styles["Normal"]))
    dataset = pandas.read_csv(file, names=features_names, index_col=0)

for random_state in random_states:
    seed_folder = os.path.join(output_path, str(random_state))
    if not os.path.isdir(seed_folder):
        os.mkdir(seed_folder)
    # -----------------------------------------------------------------------------------------------------------
    # --- DIVIDING DATASET  -------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------

    dataset_values = dataset.values
    X = dataset_values[:, 0:len(features_names) - 2]  # -2
    Y = dataset_values[:, len(features_names) - 2]  # -2

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=test_ratio,
                                                                        random_state=random_state, stratify=Y)

    print("------> Seed ", random_state)
    for task in range(25):
        print("  Task ", task+1)
        output_path_file = os.path.join(seed_folder, str(task + 1) + ".pdf")
        # print(output_path_file)

        doc = SimpleDocTemplate(output_path_file,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        story = []
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))

        story.append(Paragraph(str(datetime.datetime.now()), styles["Normal"]))
        story.append(Spacer(10, 50))

        # ------------------------------------------------------------------------------------------------------------
        # --- CLASSIFICATION    -------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------

        # - RANDOM FOREST -------------------------------------------------------------------------------------------
        if ranFor:
            scoring = 'accuracy'
            name = 'Random Forest'
            print(name)

            # Accuracy txt file
            output_path_txt = os.path.join(output_path, name + "_" + str(task + 1) + ".txt")
            file_txt = open(output_path_txt, 'a')

            model = Pipeline([
                ('col_extract', ColumnExtractor(cols=range((NUM_FEATURE * task), NUM_FEATURE * (task + 1)))),
                ('clf', RandomForestClassifier(random_state=random_state)),
            ])

            # print("    Grid search parameters tuning...")
            #
            param_grid = [{'clf__n_estimators': [100, 150, 200, 250, 300], 'clf__max_depth': [3, 4, 5, 6, 7, 8, 9]},
                          {'clf__bootstrap': [False], 'clf__n_estimators': [100, 150, 200, 250, 300],
                           'clf__max_depth': [3, 4, 5, 6, 7, 8, 9]}]

            # param_grid = [{'clf__bootstrap': [True], 'clf__n_estimators': [150], 'clf__max_depth': [3]}]

            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_
            # print("        ", best_model)

            # on test set
            # print(best_model.score(x_test, y_test))
            predictions = best_model.predict(x_test)
            predictions_TS = best_model.predict(x_train)
            # print("Report")
            # print(confusion_matrix(y_test, predictions))
            # print(classification_report(y_test, predictions))
            # print(predictions)
            # print(best_model.predict(x_train))

            add_modelresults_to_report(name, scoring, grid_search, grid_search.best_score_, y_test, predictions,
                                       predictions_TS)
            file_txt.write(str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.close()

        # - knn ----------------------------------------------------------------------------------
        if knn:
            scoring = 'accuracy'
            name = 'KNN'
            print(name)

            # Accuracy txt file
            output_path_txt = os.path.join(output_path, name + "_" + str(task + 1) + ".txt")
            file_txt = open(output_path_txt, 'a')

            model = Pipeline([
                ('col_extract', ColumnExtractor(cols=range((NUM_FEATURE * task), NUM_FEATURE * (task + 1)))),
                ('clf', KNeighborsClassifier()),
            ])

            # print("    Grid search parameters tuning...")
            #
            param_grid = [{'clf__n_neighbors': [5, 8, 9, 10, 11, 20]}]
            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_
            # print("        ", best_model)

            # on test set
            # print(best_model.score(x_test, y_test))
            predictions = best_model.predict(x_test)
            predictions_TS = best_model.predict(x_train)
            # print("Report")
            # print(confusion_matrix(y_test, predictions))
            # print(classification_report(y_test, predictions))

            # print(predictions)
            # print(best_model.predict(x_train))

            add_modelresults_to_report(name, scoring, grid_search, grid_search.best_score_, y_test, predictions,
                                       predictions_TS)
            file_txt.write(str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.close()

        # - LDA ---------------------------------------------------------------------------------
        if LDA:
            scoring = 'accuracy'
            name = 'LDA'
            print(name)

            # Accuracy txt file
            output_path_txt = os.path.join(output_path, name+"_"+str(task+1)+".txt")
            file_txt = open(output_path_txt, 'a')

            best_model = Pipeline([
                ('col_extract', ColumnExtractor(cols=range((NUM_FEATURE * task), NUM_FEATURE * (task + 1)))),
                ('clf', LinearDiscriminantAnalysis()),
            ])

            kfold = model_selection.KFold(n_splits=cross_val, random_state=random_state, shuffle=True)
            cv_results = model_selection.cross_val_score(best_model, x_train, y_train, cv=kfold, scoring=scoring)
            for train_index, test_index in kfold.split(x_train):
                X_train_cv, X_test_cv_cv, y_train_cv, y_test_cv = x_train[train_index], x_train[test_index], y_train[
                    train_index], y_train[test_index]
                best_model.fit(X_train_cv, y_train_cv)

            best_model.fit(x_train, y_train)
            # on test set
            # print(best_model.score(x_test, y_test))
            predictions = best_model.predict(x_test)
            predictions_TS = best_model.predict(x_train)
            # print("Report")
            # print(confusion_matrix(y_test, predictions))
            # print(classification_report(y_test, predictions))
            # print(predictions)
            # print(best_model.predict(x_train))

            add_modelresults_to_report(name, scoring, None, cv_results.mean(), y_test, predictions, predictions_TS)
            file_txt.write(str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.close()

        # - Gaussian NB ---------------------------------------------------------------------------------
        if GNB:
            scoring = 'accuracy'
            name = 'Gausian NB'
            print(name)

            # Accuracy txt file
            output_path_txt = os.path.join(output_path, name + "_" + str(task + 1) + ".txt")
            file_txt = open(output_path_txt, 'a')

            best_model = Pipeline([
                ('col_extract', ColumnExtractor(cols=range((NUM_FEATURE * task), NUM_FEATURE * (task + 1)))),
                ('clf', GaussianNB()),
            ])

            kfold = model_selection.KFold(n_splits=cross_val, random_state=random_state, shuffle=True)
            cv_results = model_selection.cross_val_score(best_model, x_train, y_train, cv=kfold, scoring=scoring)
            for train_index, test_index in kfold.split(x_train):
                X_train_cv, X_test_cv_cv, y_train_cv, y_test_cv = x_train[train_index], x_train[test_index], y_train[
                    train_index], y_train[test_index]
                best_model.fit(X_train_cv, y_train_cv)
            best_model.fit(x_train, y_train)

            # on test set
            # print(best_model.score(x_test, y_test))
            predictions = best_model.predict(x_test)
            predictions_TS = best_model.predict(x_train)
            # print("Report")
            # print(confusion_matrix(y_test, predictions))
            # print(classification_report(y_test, predictions))
            # print(predictions)
            # print(best_model.predict(x_train))

            add_modelresults_to_report(name, scoring, None, cv_results.mean(), y_test, predictions, predictions_TS)
            file_txt.write(str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.close()

        # - LOGISTIC REGRESSION ----------------------------------------------------------------------------------
        if logReg:
            scoring = 'accuracy'
            name = 'Logistic Regression'
            print(name)

            # Accuracy txt file
            output_path_txt = os.path.join(output_path, name + "_" + str(task + 1) + ".txt")
            file_txt = open(output_path_txt, 'a')

            model = Pipeline([('col_extract', ColumnExtractor(cols=range((NUM_FEATURE * task), NUM_FEATURE * (task + 1)))),
                              ("scaler", StandardScaler()),
                              ("clf", LogisticRegression(random_state=random_state))])

            # print("    Grid search parameters tuning...")
            #
            param_grid = [{'clf__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]}]
            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_
            # print("        ", best_model)

            # on test set
            # print(best_model.score(x_test, y_test))
            predictions = best_model.predict(x_test)
            predictions_TS = best_model.predict(x_train)
            # print("Report")
            # print(confusion_matrix(y_test, predictions))
            # print(classification_report(y_test, predictions))
            # print(predictions)
            # print(best_model.predict(x_train))

            add_modelresults_to_report(name, scoring, grid_search, grid_search.best_score_, y_test, predictions,
                                       predictions_TS)
            file_txt.write(str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.close()

        # - SVM -rbf --------------------------------------------------------------------------------------
        if SVM2:
            scoring = 'accuracy'
            name = 'SVM-rbf'
            print(name)

            # Accuracy txt file
            output_path_txt = os.path.join(output_path, name + "_" + str(task + 1) + ".txt")
            file_txt = open(output_path_txt, 'a')

            model = Pipeline([('col_extract', ColumnExtractor(cols=range((NUM_FEATURE * task), NUM_FEATURE * (task + 1)))),
                              ("scaler", StandardScaler()),
                              ("clf", SVC(kernel="rbf", gamma='scale', random_state=random_state))])

            # print("    Grid search parameters tuning...")
            param_grid = [{'clf__C': [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]}]
            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_
            # print("        ", best_model)

            # on test set
            # print(best_model.score(x_test, y_test))
            predictions = best_model.predict(x_test)
            predictions_TS = best_model.predict(x_train)
            # print("Report")
            # print(classification_report(y_test, predictions))
            # rint(predictions)
            # print(best_model.predict(x_train))

            add_modelresults_to_report(name, scoring, grid_search, grid_search.best_score_, y_test, predictions,
                                       predictions_TS)
            file_txt.write(str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.close()
        # - SVM -------------------------------------------------------------------------------------------
        if SVM:
            scoring = 'accuracy'
            name = 'SVM'
            print(name)

            # Accuracy txt file
            output_path_txt = os.path.join(output_path, name + "_" + str(task + 1) + ".txt")
            file_txt = open(output_path_txt, 'a')

            model = Pipeline([('col_extract', ColumnExtractor(cols=range((NUM_FEATURE * task), NUM_FEATURE * (task + 1)))),
                              ("scaler", StandardScaler()),
                              ("clf", LinearSVC(loss="hinge", max_iter=50000, random_state=random_state))])

            # print("    Grid search parameters tuning...")
            param_grid = [{'clf__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10]}]
            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_
            # print("        ", best_model)

            # on test set
            # print(best_model.score(x_test, y_test))
            predictions = best_model.predict(x_test)
            predictions_TS = best_model.predict(x_train)
            # print("Report")
            # print(classification_report(y_test, predictions))
            # rint(predictions)
            # print(best_model.predict(x_train))

            add_modelresults_to_report(name, scoring, grid_search, grid_search.best_score_, y_test, predictions,
                                       predictions_TS)
            file_txt.write(str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.close()

        # - DECISION TREE -------------------------------------------------------------------------------------------
        if DECISION_TREE:
            scoring = 'accuracy'
            name = 'Decision Tree'
            print(name)

            # Accuracy txt file
            output_path_txt = os.path.join(output_path, name + "_" + str(task + 1) + ".txt")
            file_txt = open(output_path_txt, 'a')

            model = Pipeline([
                ('col_extract', ColumnExtractor(cols=range((NUM_FEATURE * task), NUM_FEATURE * (task + 1)))),
                ('clf', DecisionTreeClassifier(random_state=random_state)),
            ])

            # print("    Grid search parameters tuning...")
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
            best_model = grid_search.best_estimator_
            # print("        ", best_model)

            # on test set
            # print(best_model.score(x_test, y_test))
            predictions = best_model.predict(x_test)
            predictions_TS = best_model.predict(x_train)
            # print("Report")
            # print(confusion_matrix(y_test, predictions))
            # print(classification_report(y_test, predictions))
            # print(predictions)
            # print(predictions_TS)  # training

            add_modelresults_to_report(name, scoring, grid_search, grid_search.best_score_, y_test, predictions,
                                       predictions_TS)
            file_txt.write(str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.close()

        # - MLP  ---------------------------------------------------------------------------------------
        if MLP:
            scoring = 'accuracy'
            name = 'Multi-layer Perceptron'
            print(name)

            # Accuracy txt file
            output_path_txt = os.path.join(output_path, name + "_" + str(task + 1) + ".txt")
            file_txt = open(output_path_txt, 'a')

            model = Pipeline([
                ('col_extract', ColumnExtractor(cols=range((NUM_FEATURE * task), NUM_FEATURE * (task + 1)))),
                ("scaler", StandardScaler()),
                ('clf', MLPClassifier(random_state=random_state)),
            ])

            # print("    Grid search parameters tuning...")
            #
            """
                    param_grid = [{'clf__solver': ["adam"],
                                   'clf__hidden_layer_sizes': [100, 200, 300, 400, 500, 700, 1000],
                                   'clf__activation': ["relu", "logistic", "tanh"],
                                   'clf__alpha': [0.0001],
                                   'clf__batch_size': ["auto"],
                                   'clf__learning_rate_init': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
                                   'clf__max_iter': [1000],
                                   'clf__shuffle': [True],
                                   },
                                  {'clf__solver': ["sgd"],
                                   'clf__power_t': [0.5],  # solver sgd
                                   'clf__momentum': [0.9],  # solver sgd
                                   'clf__hidden_layer_sizes': [100, 200, 300, 400, 500, 700, 1000],
                                   'clf__activation': ["relu", "logistic", "tanh"],
                                   'clf__alpha': [0.0001],
                                   'clf__batch_size': ["auto"],
                                   'clf__learning_rate_init': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
                                   'clf__max_iter': [10000],
                                   'clf__shuffle': [True],
                                   },
                                  ]
                    """
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
            best_model = grid_search.best_estimator_
            # print("        ", best_model)

            # on test set
            # print(best_model.score(x_test, y_test))
            predictions = best_model.predict(x_test)
            predictions_TS = best_model.predict(x_train)
            # print("Report")
            # print(confusion_matrix(y_test, predictions))
            # print(classification_report(y_test, predictions))
            # print(predictions)
            # print(predictions_TS)  # training

            add_modelresults_to_report(name, scoring, grid_search, grid_search.best_score_, y_test, predictions,
                                       predictions_TS)
            file_txt.write(str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.close()

        # - Learning Vector Quantization -----------------------------------------------------------------------
        if LVQ:
            scoring = 'accuracy'
            name = 'Learning Vector Quantization'
            print(name)

            # Accuracy txt file
            output_path_txt = os.path.join(output_path, name + "_" + str(task + 1) + ".txt")
            file_txt = open(output_path_txt, 'a')

            model = Pipeline([
                ('col_extract', ColumnExtractor(cols=range((NUM_FEATURE * task), NUM_FEATURE * (task + 1)))),
                ("scaler", StandardScaler()),
                ('clf', GlvqModel(random_state=random_state)),
            ])

            # print("    Grid search parameters tuning...")
            #

            param_grid = [{'clf__prototypes_per_class': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                           'clf__max_iter': [2500],
                           'clf__gtol': [0.000001],
                           'clf__beta': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                           # 'clf__C': [[2, 3]],

                           },
                          ]

            grid_search = GridSearchCV(model, param_grid, cv=cross_val, scoring=scoring, return_train_score=True)
            grid_search.fit(x_train, y_train)
            best_model = grid_search.best_estimator_
            # print("        ", best_model)

            # on test set
            # print(best_model.score(x_test, y_test))
            predictions = best_model.predict(x_test)
            predictions_TS = best_model.predict(x_train)
            # print("Report")
            # print(confusion_matrix(y_test, predictions))
            # print(classification_report(y_test, predictions))
            # print(predictions)
            # print(predictions_TS)  # training

            add_modelresults_to_report(name, scoring, grid_search, grid_search.best_score_, y_test, predictions,
                                       predictions_TS)
            file_txt.write(str(accuracy_score(y_test, predictions)) + "\n")
            file_txt.close()

        # -------------------------------------------------------------------------------------------------
        doc.build(story)
