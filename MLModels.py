from sklearn.linear_model import LinearRegression
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from surprise import accuracy
from surprise.model_selection import GridSearchCV as SurpriseGridSearch
from surprise import KNNBasic
from surprise.model_selection import train_test_split as surprise_split
from implicit.evaluation import train_test_split as implicit_split
from implicit.evaluation import precision_at_k
import itertools
import joblib
from typing import Tuple
from DataHandler import DataPreProcess, FeaturesEngineer


logging.basicConfig(level=logging.INFO)


"""
create class to build and train the models
all the data get its data from DataPrePricess class
"""

class TrainModel:

### create and train and evaluate context based models ###
#in each function we train the model make prediction and evaluate the prediction 
#they all get the data from context_based_data_preprocessing_pipeline in DataPreProcessing class

    #linear regression model 
    @staticmethod
    def context_based_linear_regression_model(x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.Series):
        model = LinearRegression()
        model.fit(x_train,y_train)
        logging.info("model training complete")
        prediction = model.predict(x_test)
        logging.info("created prediction")

        return model, prediction
    
    #random forest tree regression
    @staticmethod
    def context_based_radom_tree_regression(x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.Series):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        logging.info("training context based random forest regression model")
        prediction = model.predict(x_test)
        logging.info("create a prediction")
        return model, prediction
    
    #extrime gradient boosting model
    @staticmethod
    def XBG_gradient_boosting_model(x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.Series):
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(x_train, y_train)
        logging.info("successfully trained the model")
        prediction = model.predict(x_test)
        logging.info("prediction has been made")
        return model, prediction
    
    #light gradient boosting model
    @staticmethod
    def Light_gradient_boosting_model(x_train:pd.DataFrame, x_test:pd.DataFrame, y_train:pd.Series):
        model = LGBMRegressor(n_estimators=100,learning_rate=0.1, random_state=42)
        model.fit(x_train, y_train)
        logging.info("create and train model")
        prediction = model.predict(x_test)
        logging.info("successfully created a prediction")
        return model, prediction
    
    #models evaluation 
    @staticmethod
    def context_based_models_evaluetion(prediction:np.ndarray, y_test:pd.Series):
        mse = mean_squared_error(y_test, prediction)
        logging.info(f"the mse evaluetion is: {mse}")
        rmse = np.sqrt(mse)
        logging.info(f"the rmse evaluetion is: {rmse}")
        r2 = r2_score(y_test, prediction)
        logging.info(f"the r2 evaluetion is: {r2}")

        evaluation_scores = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse
        }
        return evaluation_scores


### create train and evaluate users based model ###
#get user-item train matrix from split_user_item_matrix in DataPreProcessing

    #KNN model
    @staticmethod
    def knn_user_based_model(train_matrix):
        model = NearestNeighbors(metric="cosine", algorithm="brute")
        model.fit(train_matrix)
        logging.info("successfully created user based model with KNN")
        
        return model
    
    
    #ALS model
    @staticmethod
    def als_user_based_model(train_matrix):
        model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
        model.fit(train_matrix.T)
        logging.info("ALS model creates and trained successfully")
        
        return model
    
    ## important!! SVD algorithm get Ratins dataframe not the item user matrix

    #svd model
    #return the model the train set and the test set
    @staticmethod
    def svd_user_based_model(rating_df:pd.DataFrame):
        reader = Reader(rating_scale=(1,10))    #rating scale
        data = Dataset.load_from_df(rating_df[["user_id", "isbn", "book_rating"]], reader)   #surprise data object
        train_set, testset = train_test_split(data, test_size=0.2, random_state=42)   #split the data into 2 matrixes
        model = SVD(n_factors=50, random_state=42)
        model.fit(train_set)
        logging.info("SVD model create and trained successfully")
        return model, train_set, testset

    #create user based predictions
      
    #KNN prediction
    @staticmethod
    def user_based_knn_prediction(user_id: int, model: NearestNeighbors, user_item_matrix: pd.DataFrame, train_matrix: csr_matrix, k=5) -> list:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = train_matrix[user_index]
        distances, indices = model.kneighbors(user_vector, n_neighbors=k + 1)
        neighbor_indices = indices.flatten()[1:]  
        neighbor_vectors = train_matrix[neighbor_indices]    
        recommendation_scores_array = neighbor_vectors.mean(axis=0)    
        recommendation_scores = pd.Series(recommendation_scores_array.A1, index=user_item_matrix.columns)
        user_rated_books = user_item_matrix.iloc[user_index]    
        recommendation_scores = recommendation_scores[user_rated_books == 0]    
        top_recommends = recommendation_scores.sort_values(ascending=False).head(k)    
        logging.info(f"Generated {k} recommendations for user {user_id} using KNN.")
        return list(top_recommends.items())

    #ALS prediction
    @staticmethod
    def user_based_als_prediction(user_id: int, model: AlternatingLeastSquares, user_item_matrix: pd.DataFrame, train_matrix_T: csr_matrix, num_recommendation: int = 5) -> list:
        user_index = user_item_matrix.index.get_loc(user_id)        
        item_indices, scores = model.recommend(user_index, train_matrix_T, N=num_recommendation)  
        recommendations = []
        for item_index, score in zip(item_indices, scores):
            isbn = user_item_matrix.columns[item_index]
            recommendations.append((isbn, score))
            
        logging.info(f"Generated {num_recommendation} recommendations for user {user_id} using ALS.")
        return recommendations

    #SVD prediction
    @staticmethod
    def user_based_svd_prediction(user_id: int, model: SVD, train_df: pd.DataFrame, num_recommendation=5) -> list:
        all_isbns = set(train_df['isbn'].unique())
        rated_isbns = set(train_df[train_df['user_id'] == user_id]['isbn'].unique())
        unrated_books = all_isbns - rated_isbns
        predictions = []
        for isbn in unrated_books:
            pred_obj = model.predict(uid=user_id, iid=isbn)
            predictions.append((pred_obj.iid, pred_obj.est))
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        logging.info(f"Generated {num_recommendation} recommendations for user {user_id} using SVD.")
        return predictions[:num_recommendation]

    
    #create evaluation functions for the user based models

    # Helper function to predict a single rating for KNN
    @staticmethod
    def _predict_knn_rating(user_idx, item_idx, model, train_matrix, k):
        user_vector = train_matrix[user_idx]
        _, indices = model.kneighbors(user_vector, n_neighbors=k + 1)
        neighbor_indices = indices.flatten()[1:]
        neighbor_ratings = train_matrix[neighbor_indices, item_idx].toarray().flatten()
        valid_ratings = neighbor_ratings[neighbor_ratings > 0]
        return np.mean(valid_ratings) if valid_ratings.size > 0 else train_matrix.data.mean()


    #knn model evaluation
    @staticmethod
    def evaluate_knn_model(model, train_matrix, test_matrix, user_item_matrix, k=10, rating_threshold=8.0):
        all_precisions, all_recalls = [], []
        
        for user_idx in range(test_matrix.shape[0]):
            user_id = user_item_matrix.index[user_idx]
            
            relevant_items = set(user_item_matrix.columns[test_matrix[user_idx].indices[test_matrix[user_idx].data >= rating_threshold]])
            if not relevant_items:
                continue
            
            recs = TrainModel.user_based_knn_prediction(user_id, model, user_item_matrix, train_matrix, k=k)
            recommended_items = {isbn for isbn, score in recs}
            
            hits = len(recommended_items.intersection(relevant_items))
            all_precisions.append(hits / k)
            all_recalls.append(hits / len(relevant_items))

        avg_precision = np.mean(all_precisions) if all_precisions else 0
        avg_recall = np.mean(all_recalls) if all_recalls else 0
        
        test_user_indices, test_item_indices = test_matrix.nonzero()
        actual_ratings = test_matrix.data
        predicted_ratings = [TrainModel._predict_knn_rating(u, i, model, train_matrix, k) for u, i in zip(test_user_indices, test_item_indices)]
            
        mse = mean_squared_error(actual_ratings, predicted_ratings)
        rmse = np.sqrt(mse)

        logging.info(f"KNN Model Evaluation (k={k}):")
        logging.info(f"  - RMSE: {rmse:.4f}")
        logging.info(f"  - Average Precision: {avg_precision:.4f}")
        logging.info(f"  - Average Recall: {avg_recall:.4f}")

        return {"rmse": rmse, "precision": avg_precision, "recall": avg_recall}

    #svd model evaluation
    @staticmethod
    def evaluate_svd_model(model: SVD, testset, k=10, rating_threshold=8.0):
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)

        user_predictions = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_predictions[uid].append((iid, est))

        user_ground_truth = defaultdict(list)
        for uid, iid, true_r in testset:
            if true_r >= rating_threshold:
                user_ground_truth[uid].append(iid)

        all_precisions = dict()
        all_recalls = dict()

        for uid, user_preds in user_predictions.items():
            if not user_ground_truth[uid]:
                continue

            user_preds.sort(key=lambda x: x[1], reverse=True)
            recommended_items = {iid for (iid, est) in user_preds[:k]}
            
            ground_truth_items = set(user_ground_truth[uid])
            hits = len(recommended_items.intersection(ground_truth_items))

            all_precisions[uid] = hits / k
            all_recalls[uid] = hits / len(ground_truth_items)

        avg_precision = sum(prec for prec in all_precisions.values()) / len(all_precisions)
        avg_recall = sum(rec for rec in all_recalls.values()) / len(all_recalls)

        logging.info(f"SVD Model Evaluation (k={k}):")
        logging.info(f"  - RMSE: {rmse:.4f}")
        logging.info(f"  - Average Precision: {avg_precision:.4f}")
        logging.info(f"  - Average Recall: {avg_recall:.4f}")

        return {"rmse": rmse, "precision": avg_precision, "recall": avg_recall}

    # Helper function to predict a single rating for ALS
    @staticmethod
    def predict_als_rating(user_idx, item_idx, model):
        user_vector = model.user_factors[user_idx]
        item_vector = model.item_factors[item_idx]
        return user_vector.dot(item_vector)

    #als model evaluation     
    @staticmethod
    def evaluate_als_model(model, train_matrix_T, test_matrix, user_item_matrix, k=10, rating_threshold=8.0):
        all_precisions, all_recalls = [], []

        # --- 1. Calculate Precision and Recall @k ---
        for user_idx in range(test_matrix.shape[0]):
            user_id = user_item_matrix.index[user_idx]
            relevant_items = set(user_item_matrix.columns[test_matrix[user_idx].indices[test_matrix[user_idx].data >= rating_threshold]])
            if not relevant_items:
                continue
            
            # Get Top-K recommendations using the ALS function
            recs = TrainModel.user_based_als_prediction(user_id, model, user_item_matrix, train_matrix_T, num_recommendation=k)
            recommended_items = {isbn for isbn, score in recs}

            hits = len(recommended_items.intersection(relevant_items))
            all_precisions.append(hits / k)
            all_recalls.append(hits / len(relevant_items))

        avg_precision = np.mean(all_precisions) if all_precisions else 0
        avg_recall = np.mean(all_recalls) if all_recalls else 0
        
        # --- 2. Calculate RMSE ---
        test_user_indices, test_item_indices = test_matrix.nonzero()
        actual_ratings = test_matrix.data
        predicted_ratings = [TrainModel._predict_als_rating(u, i, model) for u, i in zip(test_user_indices, test_item_indices)]
            
        mse = mean_squared_error(actual_ratings, predicted_ratings)
        rmse = np.sqrt(mse)
        
        logging.info(f"ALS Model Evaluation (k={k}):")
        logging.info(f"  - RMSE: {rmse:.4f}")
        logging.info(f"  - Average Precision: {avg_precision:.4f}")
        logging.info(f"  - Average Recall: {avg_recall:.4f}")
        
        return {"rmse": rmse, "precision": avg_precision, "recall": avg_recall}



"""
create class with functions to improve the models
"""

class ModelsHyperparametersImprovment:

##improve context base models##

    # random forest regression hyperparameters improvement
    @staticmethod
    def context_based_radom_forest_hyperparameters_improvement( x_train: pd.DataFrame, y_train:pd.Series):
        param_grid = {
            "n_estimators":[100,200,300,400,500,600,700,800],
            'max_depth': [10, 20, 30, None],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestRegressor(random_state=42)
        randon_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=8, 
                                   cv=3, scoring='neg_mean_squared_error', verbose=2, random_state=42, n_jobs=-1)
        randon_search.fit(x_train,y_train)
        logging.info(f"the best n_estimator for the model is: {randon_search.best_params_}")
        return randon_search.best_estimator_
    
    # extrem gradient boosting hyperparameters improvment
    @staticmethod
    def context_based_XBGgradient_boosting_hyperparameters_improvment( x_train:pd.DataFrame, y_train:pd.Series):
        param_grid = {
                "n_estimators": [100, 200, 300, 400, 500],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9]
        }
        model = XGBRegressor(random_state=42)
        random_search_XGBoost = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=20,
                                                   cv=3, scoring="neg_mean_squared_error", verbose=2, random_state=42, n_jobs=-1)
        random_search_XGBoost.fit(x_train, y_train)
        logging.info(f"model trained successfully, best parameters {random_search_XGBoost.best_params_}")
        return random_search_XGBoost.best_estimator_
    
    #light gradient boosting hyperparameters improvment
    @staticmethod
    def context_based_LGBgradient_boostin_hyperparameters_improvment(x_train:pd.DataFrame, y_train:pd.Series):
        param_grid = {
                "n_estimators": [100, 200, 300, 400, 500],
                "learning_rate": [0.05, 0.1, 0.2],
                "num_leaves": [20, 31, 40, 50],
                "max_depth": [3, 5, 7],
                "colsample_bytree": [0.7, 0.8, 0.9]
        }
        model = LGBMRegressor(random_state=42)
        random_search_LBGoost = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=20,
                                                   cv=3, scoring="neg_mean_squared_error", verbose=2, random_state=42, n_jobs=-1)
        random_search_LBGoost.fit(x_train, y_train)
        logging.info(f"model trained successfully, best parameters {random_search_LBGoost.best_params_}")
        return random_search_LBGoost.best_estimator_

    ### user based models hyperparameters improvment ###

    #knn hyperparameters improvment 
    @staticmethod
    def tune_knn_model(data):
        """
        Finds the best 'k' for a KNN model by iterating and checking RMSE.
        'data' is the full dataset loaded by the surprise Reader.
        """
        trainset, testset = surprise_split(data, test_size=0.2)
        
        k_values = [10, 20, 30, 40, 50]
        rmse_results = {}
        
        for k in k_values:
            model = KNNBasic(k=k, sim_options={'user_based': True}, verbose=False)
            model.fit(trainset)
            predictions = model.test(testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            rmse_results[k] = rmse
        
        best_k = min(rmse_results, key=rmse_results.get)
        logging.info(f"Best k found: {best_k} with RMSE: {rmse_results[best_k]:.4f}")
        
        # Create and retrain the final model with the best k on all data
        final_model = KNNBasic(k=best_k, sim_options={'user_based': True})
        final_model.fit(data.build_full_trainset())
        return final_model

    #svd model hyperparameters improvment
    @staticmethod
    def tune_svd_model(data):
        param_grid = {
            'n_factors': [50, 100, 150],
            'n_epochs': [20, 30],
            'lr_all': [0.005, 0.01],
            'reg_all': [0.02, 0.1]
        }
        
        gs = SurpriseGridSearch(SVD, param_grid, measures=['rmse'], cv=3)
        gs.fit(data)
        
        best_params = gs.best_params['rmse']
        logging.info(f"Best SVD RMSE score: {gs.best_score['rmse']:.4f}")
        logging.info(f"Best SVD parameters: {best_params}")
        
        # Create and retrain the final model with the best parameters
        final_model = SVD(**best_params)
        final_model.fit(data.build_full_trainset())
        return final_model

    #als tuning model
    @staticmethod
    def tune_als_model(train_matrix):
        # Split data for validation
        train, validate = implicit_split(train_matrix, split_count=2, split_by='user')
        
        param_grid = {
            'factors': [30, 50, 80],
            'regularization': [0.01, 0.1],
            'iterations': [15, 20]
        }
        
        best_score = -1
        best_params = {}
        
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

        for params in all_params:
            model = AlternatingLeastSquares(**params)
            model.fit(train.T)
            
            # Evaluate using Precision@10
            score = precision_at_k(model, train_user_items=train, test_user_items=validate, K=10)
            
            if score > best_score:
                best_score = score
                best_params = params

        logging.info(f"Best ALS Precision@10: {best_score:.4f}")
        logging.info(f"Best ALS parameters: {best_params}")

        # Create and retrain the final model on all data
        final_model = AlternatingLeastSquares(**best_params)
        final_model.fit(train_matrix.T)
        return final_model

"""
create a class to orgenize and compare the models 
and pick up the best out of them train him and build 
a pipeline to the main with him
"""

class ModelOrganaize:
    
    #regression models comparison
    @staticmethod
    def compare_regression_model(linear_regression:BaseEstimator, random_tree_regression:BaseEstimator, extrem_gradient_boosting:BaseEstimator, light_gradient_boosting:BaseEstimator,
                                y_test:pd.Series):

        min_mse = float('inf')
        min_rmse = float('inf')
        mse_best_model = ''
        rmse_best_model = ''

        linear_regression_model = linear_regression[0]
        linear_regression_prediction = linear_regression[1]

        random_tree_regression_model = random_tree_regression[0]
        random_tree_regression_prediction = random_tree_regression[1]

        extrem_gradient_boosting_model = extrem_gradient_boosting[0]
        extrem_gradient_boosting_prediction = extrem_gradient_boosting[1]

        light_gradient_boosting_model = light_gradient_boosting[0]
        light_gradient_boosting_prediction = light_gradient_boosting[1]

        comparation = {
            "linear_regression" : TrainModel.context_based_models_evaluetion(linear_regression_prediction, y_test),
            "random_tree_regression" : TrainModel.context_based_models_evaluetion(random_tree_regression_prediction, y_test),
            "extrem_gradient_boosting" : TrainModel.context_based_models_evaluetion(extrem_gradient_boosting_prediction, y_test),
            "light_gradient_boosting" : TrainModel.context_based_models_evaluetion(light_gradient_boosting_prediction, y_test)
        }

        for model, eval in comparation.items():
            for eval_type, evaluation in eval.items():
                if eval_type == "mse":
                    if evaluation < min_mse:
                        min_mse = evaluation
                        mse_best_model = model
                elif eval_type == "rmse":
                    if evaluation < min_rmse:
                        min_rmse = evaluation
                        rmse_best_model = model
                
        logging.info(f"the best mse modle is: {mse_best_model}, and the best rmse model is {rmse_best_model}")
        if rmse_best_model == "linear_regression":
            joblib.dump(linear_regression_model,"linear_regression_model")
            logging.info(f"saved model {rmse_best_model}")
            return linear_regression
        elif rmse_best_model == "random_tree_regression":
            joblib.dump(random_tree_regression_model, "random_tree_regression_model")  
            logging.info(f"saved model {rmse_best_model}")
            return random_tree_regression
        elif rmse_best_model == "extrem_gradient_boosting":
            joblib.dump(extrem_gradient_boosting_model ,"extrem_gradient_boosting_model")
            logging.info(f"saved model {rmse_best_model}")
            return extrem_gradient_boosting
        elif rmse_best_model == "light_gradient_boosting":
            joblib.dump(light_gradient_boosting_model, "light_gradient_boosting_model")
            logging.info(f"saved model {rmse_best_model}")
            return light_gradient_boosting
        
    
    #user based model comparison
    @staticmethod
    def compare_user_based_models(ratings_df: pd.DataFrame, train_matrix: csr_matrix, test_matrix: csr_matrix, user_item_matrix: pd.DataFrame, k: int = 10):
        """
        Trains, evaluates, and compares user-based models (KNN, SVD, ALS),
        saves the best one based on Precision@k, and returns it.

        Args:
            ratings_df (pd.DataFrame): The full ratings dataframe for SVD.
            train_matrix (csr_matrix): The training user-item sparse matrix for KNN/ALS.
            test_matrix (csr_matrix): The testing user-item sparse matrix for KNN/ALS.
            user_item_matrix (pd.DataFrame): The full user-item matrix (with indices) for context.
            k (int): The number of recommendations to consider for precision/recall.

        Returns:
            The best performing trained model object.
        """
        logging.info("--- Starting User-Based Model Comparison ---")

        # 1. Train all three models
        logging.info("Training KNN model...")
        knn_model = TrainModel.knn_user_based_model(train_matrix)

        logging.info("Training SVD model...")
        svd_model, _, svd_testset = TrainModel.svd_user_based_model(ratings_df)

        logging.info("Training ALS model...")
        als_model = TrainModel.als_user_based_model(train_matrix)

        # 2. Evaluate all three models
        logging.info("\n--- Evaluating Models ---")
        knn_scores = TrainModel.evaluate_knn_model(knn_model, train_matrix, test_matrix, user_item_matrix, k=k)
        svd_scores = TrainModel.evaluate_svd_model(svd_model, svd_testset, k=k)
        als_scores = TrainModel.evaluate_als_model(als_model, train_matrix.T, test_matrix, user_item_matrix, k=k)

        # 3. Store models and their evaluation results
        trained_models = {
            "KNN": knn_model,
            "SVD": svd_model,
            "ALS": als_model
        }
        
        evaluation_results = {
            "KNN": knn_scores,
            "SVD": svd_scores,
            "ALS": als_scores
        }

        # 4. Compare models based on Precision@k to find the best one
        best_precision = -1.0
        best_model_name = None

        logging.info("\n--- Model Performance Summary ---")
        for name, scores in evaluation_results.items():
            precision = scores.get('precision', 0)
            recall = scores.get('recall', 0)
            rmse = scores.get('rmse', float('inf'))
            logging.info(f"Model: {name:<10} | Precision@{k}: {precision:.4f} | Recall@{k}: {recall:.4f} | RMSE: {rmse:.4f}")
            
            if precision > best_precision:
                best_precision = precision
                best_model_name = name

        # 5. Save the best model and return it
        if best_model_name:
            best_model_object = trained_models[best_model_name]
            filename = f"best_user_based_{best_model_name}_model.joblib"
            joblib.dump(best_model_object, filename)
            
            logging.info(f"\nBest user-based model is '{best_model_name}' with a Precision@{k} of {best_precision:.4f}.")
            logging.info(f"Model saved successfully as '{filename}'")
            return best_model_object
        else:
            logging.warning("Could not determine the best user-based model.")
            return None


"""
building  a hybride class that combained both the context based model
and the user based models. by giving more weight to each model by
the amount of rating that a user have
"""
class HybridRecommender:

    """
    building a constructor to the class
    get all the required parameters for the models 
    use to handle the recommends functions of the class 
    """
    #get the data frame from context_base_df in DataHandler
    def __init__(self, context_based_model_and_prediction:Tuple[BaseEstimator,np.array], user_based_model:BaseEstimator,
                  user_item_matrix:csr_matrix, train_df:pd.DataFrame, test_df:BaseEstimator, data_df:pd.DataFrame, user_id:int, isbn:int):
        self.context_based_model = context_based_model_and_prediction[0]
        self.context_based_prediction = context_based_model_and_prediction[1]
        self.user_based_model = user_based_model
        self.user_item_matrix = user_item_matrix
        self.train_df = train_df
        self.test_df = test_df
        self.data_df = data_df
        self.user_id = user_id
        self.isbn = isbn

    #helper function that calaulat the weight for user/book by rating
    def custom_growth_curved(x, midpoint=10, steepness=0.3):
        def shifted_sigmoid(val):
            return 1/(1+np.exp(-steepness * (val - midpoint)))
        
        y_offset = shifted_sigmoid(0)        
        value = shifted_sigmoid(x)        
        final_value = (value - y_offset) / (1 - y_offset)        

        return np.clip(final_value, 0, 1)

    #helper function create a recommendation of the top 300 book that fit to the user by context base model
    def book_sample_recommmend(self)->pd.DataFrame:
        user_books_df = self.data_df[self.data_df["user_id"] == self.user_id]["isbn"].unique()    
        all_books = self.data_df["isbn"].unique()
        unread_books = np.setdiff1d(all_books, user_books_df)
        filtered_df = self.data_df[self.data_df["isbn"].isin(unread_books)]
        preprocess_df = DataPreProcess.hybride_model_sample(filtered_df)
        prediction_df = FeaturesEngineer.hybrid_models_features_engineer(preprocess_df)

        prediction = self.context_based_model.predict(prediction_df)
        prediction_df["prediction"] = prediction
        final_df = prediction_df.sort_values("prediction", ascending=False).head(300)

        return final_df

    #helper function get the data frame from  book_sample_recommmend and return the original data frame with the 300
    #books with user id and isbn
    def get_data_frame(self, df:pd.DataFrame)->pd.DataFrame:
        books = df["isbn"].unique()
        filtered_df = self.data_df[self.data_df["isbn"].isin(books)]
        return filtered_df

    #get how much weight you need to give to each model in the evaluation
    #get data frame from get_data_frame
    def weight_per_model(self, df:pd.DataFrame):
        user_rating = len(self.data_df[self.data_df["user_id"] == self.user_id])
        user_estimate = HybridRecommender.custom_growth_curved(user_rating)

        book_rating = df.groupby("isbn")["book_rating"].count()
        book_df = book_rating.to_frame()
        book_df["estimate"] = book_rating.apply(HybridRecommender.custom_growth_curved)
        book_df["final_estimate"] = book_df["estimate"] * user_estimate

        return book_df

        
    #create recommendation by using hybrid model
    #get a data frame from get_data_frame of the 300 books
    def recommend(self, df:pd.DataFrame):
        context_based_prediction = self.context_based_model.predict(df)
        
        if isinstance(self.user_based_model,KNeighborsRegressor):
            pass
        elif isinstance(self.user_based_model,SVD):
            pass
        else: #its als
            pass
            
        
        
