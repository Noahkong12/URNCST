""" feature selection + model"""
import shap
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from boruta import BorutaPy
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
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
df = df.rename(columns={'Diagnosis': 'Label'})
print(df.dtypes)
df.dropna(subset=["Label"], inplace=True)
df['Label'].value_counts()
y = df["Label"].values

# process the output variables
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(y)

# drop the irrelavenet variables and variables with too many missing values
X = df.drop(labels=["Label", "RID", "VISCODE", "SITE", 'PTID', 'EXAMDATE', 'ORIGPROT', 'COLPROT',
                    'PIB', 'AV45', 'ABETA', 'TAU', 'PTAU', 'DIGITSCOR', 'FDG', 'DX'],
            axis=1)

# also drop features filled with over 10% emtpy value
limitPer = len(X) * 0.80
X = X.dropna(thresh=limitPer, axis=1)

# feature_names = np.array(X.columns)

# # data imputation 1
# sc = StandardScaler()
# num_d = X.select_dtypes(exclude=['object'])
# X[num_d.columns] = sc.fit_transform(num_d)
#
# X = pd.get_dummies(X)
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# X = imp.fit_transform(X)


# imputation 2
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])
categorical_transformer = SimpleImputer(strategy="most_frequent")
num_d = X.select_dtypes(exclude=['object'])
cat_d = X.select_dtypes(include=['object'])
X[num_d.columns] = numeric_transformer.fit_transform(num_d)
X[cat_d.columns] = categorical_transformer.fit_transform(cat_d)

# for column in cat_d.columns:
#     X[column] = labelencoder.fit_transform(X[column])

# normalization for the categorical variable  1
X = pd.get_dummies(X)
feature_names = np.array(X.columns)


# normalization for the categorical variable  2
# onehot = OneHotEncoder(handle_unknown="ignore")
# temp_X = pd.Dataframe(data=onehot.transform(X[cat_d]), columns=onehot.get_feature_names_out())
# X.drop(columns=cat_d, axis=1, inplace=True)
# X = pd.concat([X.reset_index(drop=True), temp_X], axis=1)
# I think the feature names statement should be put here if you want to use get dummies
# or the onehot encoder. They both increase the number of variables.


# categorical_transformer = Pipeline(steps=[
#          ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#          ('onehot', OneHotEncoder(handle_unknown='ignore'))])
# num_d = X.select_dtypes(exclude=['object'])
# categorical_features = X.select_dtypes(include=['object'])
# X[num_d.columns] = numeric_transformer.fit_transform(num_d)
# X[categorical_features.columns] = categorical_transformer.fit_transform(categorical_features)

# after feature selection
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,
                                                    random_state=20220702)

# balance class
under = RandomUnderSampler(random_state=20220702)
X_train, y_train = under.fit_resample(X_train, y_train)
X_test, y_test = under.fit_resample(X_test, y_test)

# feature selection
feat_classifier = xgb.XGBClassifier()
feat_selector = BorutaPy(feat_classifier, n_estimators='auto', verbose=2,
                         random_state=20220702)

# find all relevant features
feat_selector.fit(np.array(X_train), y_train)
# it only takes array

important = list(X_train.columns[feat_selector.support_])

# check features
print(feat_selector.support_)
print(feat_selector.ranking_)

X_filtered = feat_selector.transform(np.array(X_train))  # it only takes array
feature_ranks = list(zip(feature_names,
                         feat_selector.ranking_,
                         feat_selector.support_))

for feat in feature_ranks:
    print('Feature: {:<30} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))

X_test_filtered = feat_selector.transform(np.array(X_test))


# X_filtered = X_train
# X_test_filtered = X_test

#
# # model 1: xgb
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
# # model 4: knn model
def SVM():
    model = svm.SVC(decision_function_shape='ovo')
    model.fit(X_filtered, y_train)
    y_pred = model.predict(X_test_filtered)
    return y_pred


def KNN():
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_filtered, y_train)
    y_pred = model.predict(X_test_filtered)
    y_prob = model.predict_proba(X_test_filtered)
    return y_pred, y_prob


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
    model = XGBClassifier()
    model.fit(X_filtered, y_train)
    explainer = shap.TreeExplainer(model)
    X_test4shap = pd.DataFrame(X_test_filtered, columns=important)
    shap_value = explainer.shap_values(X_test4shap, approximate=True)
    if k == 0:
        label = 'not AD'
    else:
        label = 'AD'
    # shap.summary_plot(shap_value[k], X_test4shap, title='SHAP Value Summary Plot for' + label)
    # name = 'rf' + label + '.png'
    # plt.savefig(name)
    return shap.summary_plot(shap_value[k], X_test4shap, title='SHAP Value Summary Plot for' + label)


def exp_xgb(y_pred):
    figure, axis = plt.subplots(2, 2)
    axis[0, 0].heatmap(y_pred)
    axis[0, 0].set_title("Confusion Matrix for XGBoost model")

    axis[0, 1].shap_xgb(0)
    axis[0, 1].set_title("SHAP Summary Plot for CN")

    axis[1, 0].shap_xgb(1)
    axis[1, 0].set_title("SHAP Summary Plot for MCI")

    axis[1, 1].shap_xgb(2)
    axis[1, 1].set_title("SHAP Summary Plot for AD")

    plt.show()


def a_m(y_pred):
    print("Accuracy = ", round(metrics.accuracy_score(y_test, y_pred), 3))
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred))


def heatmap(y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return sns.heatmap(cm / np.sum(cm), annot=True,
                       fmt='.2%', cmap='Blues')

def npv_ppv(y_predd):
    cm = confusion_matrix(y_test, y_predd)
    tp, fn, fp, tn = cm.ravel()
    npv = (tn) / (tn + fn)
    ppv = (tp) / (tp + fp)
    print("NPV = ", round(npv, 3))
    print("PPV = ", (round(ppv, 3)))

def auc_roc(y_prob):
    return ('AUC-ROC:', round(roc_auc_score(y_test, y_prob, multi_class='ovo'), 3))

def shap_rf(k):
    model = RandomForestClassifier(n_estimators=10, criterion='entropy',
                                   random_state=20220702, max_depth=10)
    model.fit(X_filtered, y_train)
    explainer = shap.TreeExplainer(model)
    X_test4shap = pd.DataFrame(X_test_filtered, columns=important)
    shap_value = explainer.shap_values(X_test4shap, approximate=True)
    if k == 0:
        label = 'not AD'
    else:
        label = 'AD'
    # shap.summary_plot(shap_value[k], X_test4shap, title='SHAP Value Summary Plot for' + label)
    # name = 'rf' + label + '.png'
    # plt.savefig(name)
    return shap.summary_plot(shap_value[k], X_test4shap, title='SHAP Value Summary Plot for' + label)


