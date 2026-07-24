import sys
import os
from dataclasses import dataclass


from src.exception import CustomeException
from src.logger import logging



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score


from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        logging.info("Entered to initiate model trainer method")
        try:
            logging.info("Splitting Training and Testing Dataset")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "Linear Regression":LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Random Forest Regressor":RandomForestRegressor(),
                "Decision Tree Regressor":DecisionTreeRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=False),
                "XGB Regressor":XGBRegressor(),
                "K Neighbors Regressor":KNeighborsRegressor(),
            }

            params = {
                "Linear Regression": {},

                "Decision Tree Regressor": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "splitter": ["best", "random"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Random Forest Regressor": {
                    "n_estimators": [100, 200, 300],
                    "criterion": ["squared_error", "absolute_error"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"]
                },

                "Gradient Boosting Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"]
                },

                "CatBoost Regressor": {
                    "iterations": [100, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "depth": [4, 6, 8, 10],
                    "l2_leaf_reg": [1, 3, 5, 7]
                },

                "XGB Regressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                },

                "K Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "p": [1, 2]
                }
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                raise CustomeException("No Best Socre Found")
            
            logging.info("Best Model is found After Training on Datasets")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            best_model.fit(X_train,y_train)

            y_pred = best_model.predict(X_test)

            return r2_score(y_test,y_pred)
        
        except Exception as ex:
            raise CustomeException(ex,sys)
    
