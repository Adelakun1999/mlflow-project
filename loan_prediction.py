import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os 
import mlflow
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

df = pd.read_csv('train.csv')
df.drop(columns='Loan_ID')
cat_col = df.select_dtypes(include='object').columns.to_list()
num_col = df.select_dtypes(exclude='object').columns.to_list()
cat_col.remove('Loan_Status')

for col in cat_col:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in num_col:
    df[col].fillna(df[col].median(), inplace=True)

#Outliers 
df[num_col] = df[num_col].apply(lambda x : x.clip(*x.quantile([0.05,0.95])))
df['LoanAmount'] = np.log(df['LoanAmount']).copy()
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome'] = np.log(df['TotalIncome']).copy()

df = df.drop(columns=['ApplicantIncome','CoapplicantIncome'])
label = LabelEncoder()
for col in cat_col:
    df[col] = label.fit_transform(df[col])

df['Loan_Status'] = label.fit_transform(df['Loan_Status'])


X = df.drop('Loan_Status', axis = 1)
y = df['Loan_Status']
RANDOM_SEED = 6

X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=0.3, random_state=RANDOM_SEED)

#RandomForest 
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forst = {
    'n_estimators': [200,400,700],
    'max_depth':[10,20,30, 40],
    'criterion': ['gini','entropy'],
    'max_leaf_nodes': [50,70, 100]
}
grid_forest = GridSearchCV(estimator=rf, 
                           param_grid=param_grid_forst,
                           cv=5, n_jobs=-1,scoring='accuracy',
                           verbose=0)

model_forest = grid_forest.fit(X_train, y_train)

#LogisticRegression 

lr = LogisticRegression(random_state=RANDOM_SEED)
param_grid_log = {
    'C': [100, 10, 1.0, 0.1, 0.01],
    'penalty': ['l1','l2'],
    'solver':['liblinear']
}

grid_log = GridSearchCV(
        estimator=lr,
        param_grid=param_grid_log, 
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=0
    )
model_log = grid_log.fit(X_train, y_train)

#Decision Tree

dt = DecisionTreeClassifier(
    random_state=RANDOM_SEED
)

param_grid_tree = {
    "max_depth": [3, 5, 7, 9, 11, 13],
    'criterion' : ["gini", "entropy"],
}

grid_tree = GridSearchCV(
        estimator=dt,
        param_grid=param_grid_tree, 
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=0
    )
model_tree = grid_tree.fit(X_train, y_train)

mlflow.set_experiment('Loan_Prediction')

#model evaluation metrics 
def eval_metric(actual , pred):
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = roc_curve(actual, pred)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc_score)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # Close plot
    plt.close()
    return(accuracy, f1, auc_score)

def mlflow_logging(model, X, y, name):
    with mlflow.start_run() as run :
        run_id = run.info.run_id
        mlflow.set_tag('run_id',run_id)
        pred = model.predict(X)
        (accuracy, f1, auc_score) = eval_metric(y, pred)

        #logging best parameters from grid search
        mlflow.log_params(model.best_params_)
        #log the metrics 
        mlflow.log_metric('Mean Cv score',model.best_score_)
        mlflow.log_metric('Accuracy',accuracy)
        mlflow.log_metric('f1-score',f1)
        mlflow.log_metric('AUC',auc_score)

        mlflow.log_artifact('plots/ROC_curve.png')
        mlflow.sklearn.log_model(model, name)

        mlflow.end_run()


mlflow_logging(model_tree, X_test, y_test, 'DecisionTreeClassifier')
mlflow_logging(model_log, X_test, y_test, 'LogisticRegression')
mlflow_logging(model_forest, X_test, y_test, 'RandomForestClassifier')











