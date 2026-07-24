import sys
import os

import numpy as np
import pandas as pd


from src.exception import CustomeException
from src.logger import logging
from sklearn.metrics import r2_score
import dill


from sklearn.model_selection import GridSearchCV


def save_object(file_path,obj):
    try:
        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,"wb") as f:
            dill.dump(obj,f)
    except Exception as ex:
        raise CustomeException(ex,sys)



def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}

        for name in models.keys():
            model = models[name]
            param = params[name]

            gs = GridSearchCV(estimator=model,cv=3,n_jobs=-1,param_grid=param)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train,y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_data_score = r2_score(y_train,y_train_pred)
            test_data_score = r2_score(y_test,y_test_pred)

            report[name] = test_data_score
        
        return report
    except Exception as ex:
        raise CustomeException(ex,sys)

