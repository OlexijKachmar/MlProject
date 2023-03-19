import os
import sys

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
   
    # can add preprocessor_path
    def initiate_model_trainer(self, train_arrray, test_array):
        try:
            
            X_train, X_test, y_train, y_test = (train_arrray[:, :-1],
                                                test_array[:, :-1],
                                                train_arrray[:, -1],
                                                test_array[:, -1])
            
            model_report: dict = evaluate_model(X_train, X_test, y_train, y_test, self.models)#self.hyper_parameters
            logging.info(f"Got modelling results")

            best_model = max(model_report, key = model_report.get)
            best_score = model_report[best_model]

            if best_score < 0.6:
                raise CustomException('No best model found!')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info(f"Saved modelling object")
            prediction = best_model.predict(X_test)
            r2_squares = r2_score(y_test, prediction)
            return r2_squares

        except Exception as e:
            raise CustomException(e, sys)
