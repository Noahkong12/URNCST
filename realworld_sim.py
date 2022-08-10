""" feature selection + model"""
import shap
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

df = pd.read_csv('binary.csv')

# browse the data
print(df.describe().T)
print(df.isnull().sum())

# rename the output variable and give it to y
df = df.rename(columns={'Diagnosis': 'Label'})
print(df.dtypes)
df.dropna(subset=["Label"], inplace=True)
df['Label'].value_counts()
y = df["Label"].values

# transform categorical values to numeric values
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y)

# drop the irrelavenet variables and variables that contain too many empty data
# such as the patient ID
X = df.drop(labels=["Label", "RID", "VISCODE", "SITE", 'PTID', 'EXAMDATE', 'ORIGPROT', 'COLPROT',
                    'PIB', 'AV45', 'ABETA', 'TAU', 'PTAU', 'DIGITSCOR', 'FDG', 'DX', 'Month', 'M'],
            axis=1)

limitPer = len(X) * 0.90
X = X.dropna(thresh=limitPer, axis=1)

# imputation and normalization
# using mean value for numeric imputation
# using mode for categrocial value imputation
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])
categorical_transformer = SimpleImputer(strategy="most_frequent")
num_d = X.select_dtypes(exclude=['object'])
cat_d = X.select_dtypes(include=['object'])
X[num_d.columns] = numeric_transformer.fit_transform(num_d)
X[cat_d.columns] = categorical_transformer.fit_transform(cat_d)

# normalization for the categorical variable  1
X = pd.get_dummies(X)
feature_names = np.array(X.columns)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,
                                                    random_state=20220702)

# balance class- undersampling
under = RandomUnderSampler(random_state=20220702)
X_train, y_train = under.fit_resample(X_train, y_train)
X_test, y_test = under.fit_resample(X_test, y_test)

X_filtered = X_train
X_test_filtered = X_test

# model 1: xgb
def xgb():
    xgb_model = XGBClassifier()
    xgb_model.fit(X_filtered, y_train)
    y_pred = xgb_model.predict(X_test_filtered)
    y_prob = xgb_model.predict_proba(X_test_filtered)
    return y_pred, y_prob


# model 2: random forest
def random_forest():
    model = RandomForestClassifier(n_estimators=10, criterion='entropy',
                                   random_state=20220702, max_depth=10)
    model.fit(X_filtered, y_train)
    y_pred = model.predict(X_test_filtered)
    y_prob = model.predict_proba(X_test_filtered)
    return y_pred, y_prob


# model 3: logistic regression
def logistic():
    model = LogisticRegression(solver='liblinear', random_state=20220702)
    model.fit(X_filtered, y_train)
    y_pred = model.predict(X_test_filtered)
    y_prob = model.predict_proba(X_test_filtered)
    return y_pred, y_prob


#
# # model 4: SVM model
def SVM():
    model = svm.SVC(decision_function_shape='ovo')
    model.fit(X_filtered, y_train)
    y_pred = model.predict(X_test_filtered)
    return y_pred


# model 5 KNN model
def KNN():
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_filtered, y_train)
    y_pred = model.predict(X_test_filtered)
    y_prob = model.predict_proba(X_test_filtered)
    return y_pred, y_prob


# model 6: ensemble method
def ensemble():
    xgb_model = XGBClassifier()
    rdf = RandomForestClassifier(n_estimators=10, criterion='entropy',
                                 random_state=20220702)
    svm_model = svm.SVC(decision_function_shape='ovo')
    estimators = [('xgb', xgb_model), ('random_forest', rdf), ('svm', svm_model)]
    final_estimator = GradientBoostingClassifier(n_estimators=25, subsample=0.5,
                                                 min_samples_leaf=25, max_features=1,
                                                 random_state=20220702)
    model = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
    model.fit(X_filtered, y_train)
    y_pred = model.predict(X_test_filtered)
    y_prob = model.predict_proba(X_test_filtered)
    return y_pred, y_prob


def shap_xgb(k):
    """
    return the shap value summary plot for k = 0(non-AD), k = 1(AD)
    of XGBoost model.
    """
    model = XGBClassifier()
    model.fit(X_filtered, y_train)
    explainer = shap.TreeExplainer(model)
    X_test4shap = pd.DataFrame(X_test_filtered, columns=feature_names)
    shap_value = explainer.shap_values(X_test4shap, approximate=True)
    return shap.summary_plot(shap_value[k], X_test4shap)


def accuracy_cm_cmreport(y_pred):
    """
    return the accuracy, confusion matrix and confusion matrix report of predictions
    """
    print("Accuracy = ", round(metrics.accuracy_score(y_test, y_pred), 3))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred))


def heatmap(y_pred):
    """
    return the confusion matrix displayed by seaborn
    """
    cm = confusion_matrix(y_test, y_pred)
    return sns.heatmap(cm / np.sum(cm), annot=True,
                       fmt='.2%', cmap='Blues')


def npv_ppv(y_predd):
    """
    return the negative predictive value&positive predictive value
    """
    cm = confusion_matrix(y_test, y_predd)
    tp, fn, fp, tn = cm.ravel()
    npv = (tn) / (tn + fn)
    ppv = (tp) / (tp + fp)
    print("NPV = ", round(npv, 3))
    print("PPV = ", round(ppv, 3))


def auc_roc(y_prob):
    """
    return the AUC_ROC score of the predcitions
    """
    return ('AUC-ROC:', round(roc_auc_score(y_test, y_prob), 3))


def shap_rf(k):
    """
    return the shap value summary plot for k = 0(non-AD), k = 1(AD)
    of random forest model.
    """
    model = RandomForestClassifier(n_estimators=10, criterion='entropy',
                                   random_state=20220702, max_depth=10)
    model.fit(X_filtered, y_train)
    explainer = shap.TreeExplainer(model)
    X_test4shap = pd.DataFrame(X_test_filtered, columns=important)
    shap_value = explainer.shap_values(X_test4shap, approximate=True)
    return shap.summary_plot(shap_value[k], X_test4shap)
