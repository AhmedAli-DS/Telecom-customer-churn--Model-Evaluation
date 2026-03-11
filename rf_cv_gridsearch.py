import pandas as pd
from sklearn.model_selection import cross_val_score,GridSearchCV,cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_auc_score

#loading the updated csv file
data = pd.read_csv("Telco_Selected_Features.csv")

X = data.drop("Churn",axis=1)
y = data["Churn"]

#Initialize Random Forest and perform 5-fold cross validation using accuracy
rf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(rf,X,y,cv=5,scoring="accuracy")
#print("Accuracy for each fold",cv_scores)
#print("Average CV accuracy",cv_scores.mean())

#define a parameter grid tu tune
param_grid = {
    'n_estimators':[100,200,300],
    'max_depth':[None,5,10,15],
    'min_samples_split':[2,5,10],
    'min_samples_leaf':[1,2,4]
}

#Initailize GridSearchCV with 5-fold CV
grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,scoring="accuracy",n_jobs=-1)
grid_search.fit(X,y)

#print("Best Hyperparameter",grid_search.best_params_)
#print("Best CV accuracy",grid_search.best_score_)

#using cross_val_predict to get predictions for each fold
best_rf = grid_search.best_estimator_
y_pred = cross_val_predict(best_rf,X,y,cv=5)

print("\n=== Performance Metrics for Tuned Random Forest ===")
print("F1 Score:", f1_score(y, y_pred))
print("Recall:", recall_score(y, y_pred))
print("ROC-AUC Score:", roc_auc_score(y, y_pred))