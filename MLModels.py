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
                                x_train:pd.DataFrame, x_test:pd.Series, y_train:pd.DataFrame, y_test:pd.Series):

        min_mse = 10
        min_rmse = 10
        min_r2 = 0

        linear_regression_train = TrainModel.context_based_linear_regression_model(x_train, x_test, y_train)
        random_tree_regression_train = TrainModel.context_based_radom_tree_regression(x_train, x_test, y_train)
        extrem_gradient_boosting_train = TrainModel.XBG_gradient_boosting_model(x_train, x_test, y_train)
        light_gradient_boosting_train = TrainModel.Light_gradient_boosting_model(x_train, x_test, y_train)

        linear_regression_model = linear_regression_train[0]
        linear_regression_prediction = linear_regression_train[1]

        random_tree_regression_model = random_tree_regression_train[0]
        random_tree_regression_prediction = random_tree_regression_train[1]

        extrem_gradient_boosting_model = extrem_gradient_boosting_train[0]
        extrem_gradient_boosting_prediction = extrem_gradient_boosting_train[1]

        light_gradient_boosting_model = light_gradient_boosting_train[0]
        light_gradient_boosting_prediction = light_gradient_boosting_train[1]

        comparation = {
            "linear_regression" : TrainModel.context_based_models_evaluetion(linear_regression_prediction, y_test),
            "random_tree_regression" : TrainModel.context_based_models_evaluetion(random_tree_regression_prediction, y_test),
            "extrem_gradient_boosting" : TrainModel.context_based_models_evaluetion(extrem_gradient_boosting_prediction, y_test),
            "light_gradient_boosting" : TrainModel.context_based_models_evaluetion(light_gradient_boosting_prediction, y_test)
        }

    