from sklearn.linear_model import LinearRegression
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import GridSearchCV as SurpriseGridSearch
from surprise.model_selection import train_test_split as surprise_split
from implicit.evaluation import train_test_split as implicit_split
from implicit.evaluation import precision_at_k
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.model_selection import RandomizedSearchCV
from typing import Tuple
from collections import defaultdict
import itertools
import joblib
from DataHandler import DataPreProcess, FeaturesEngineer

logging.basicConfig(level=logging.INFO)

class TrainModel:
    @staticmethod
    # Trains and evaluates a Linear Regression model for context-based recommendations.
    def context_based_linear_regression_model(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series):
        model = LinearRegression()
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        return model, prediction
    
    @staticmethod
    # Trains and evaluates a Random Forest Regressor for context-based recommendations.
    def context_based_radom_tree_regression(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        return model, prediction
    
    @staticmethod
    # Trains and evaluates an XGBoost Regressor for context-based recommendations.
    def XBG_gradient_boosting_model(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series):
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        return model, prediction
    
    @staticmethod
    # Trains and evaluates a LightGBM Regressor for context-based recommendations.
    def Light_gradient_boosting_model(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series):
        model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        return model, prediction
    
    @staticmethod
    # Calculates R2, MSE, and RMSE scores for regression models.
    def context_based_models_evaluetion(prediction: np.ndarray, y_test: pd.Series):
        mse = mean_squared_error(y_test, prediction)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, prediction)
        return {'r2': r2, 'mse': mse, 'rmse': rmse}

    @staticmethod
    # Trains a Nearest Neighbors model using cosine similarity for user-based filtering.
    def knn_user_based_model(train_matrix):
        model = NearestNeighbors(metric="cosine", algorithm="brute")
        model.fit(train_matrix)
        return model
    
    @staticmethod
    # Trains an Alternating Least Squares (ALS) model for collaborative filtering.
    def als_user_based_model(train_matrix):
        model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=20)
        model.fit(train_matrix.T)
        return model
    
    @staticmethod
    # Trains a Singular Value Decomposition (SVD) model using the Surprise library.
    def svd_user_based_model(rating_df: pd.DataFrame):
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(rating_df[["user_id", "isbn", "book_rating"]], reader)
        train_set, testset = surprise_split(data, test_size=0.2, random_state=42)
        model = SVD(n_factors=50, random_state=42)
        model.fit(train_set)
        return model, train_set, testset

    @staticmethod
    # Generates book recommendations for a user using the KNN model.
    def user_based_knn_prediction(user_id: int, model: NearestNeighbors, user_item_matrix: pd.DataFrame, train_matrix: csr_matrix, k=5) -> pd.Series:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = train_matrix[user_index]
        distances, indices = model.kneighbors(user_vector, n_neighbors=k + 1)
        neighbor_indices = indices.flatten()[1:]  
        neighbor_vectors = train_matrix[neighbor_indices]    
        recommendation_scores_array = neighbor_vectors.mean(axis=0)    
        recommendation_scores = pd.Series(recommendation_scores_array.A1, index=user_item_matrix.columns)
        user_rated_books = user_item_matrix.iloc[user_index]    
        recommendation_scores = recommendation_scores[user_rated_books == 0]    
        return recommendation_scores.sort_values(ascending=False)

    @staticmethod
    # Generates book recommendations for a user using the ALS model.
    def user_based_als_prediction(user_id: int, model: AlternatingLeastSquares, user_item_matrix: pd.DataFrame, train_matrix_T: csr_matrix) -> pd.Series:
        num_items = train_matrix_T.shape[1]
        user_index = user_item_matrix.index.get_loc(user_id)        
        item_indices, scores = model.recommend(user_index, train_matrix_T, N=num_items) 
        isbns = user_item_matrix.columns[item_indices]
        return pd.Series(scores, index=isbns)

    @staticmethod
    # Generates book recommendations for a user using the SVD model.
    def user_based_svd_prediction(user_id: int, model: SVD, train_df: pd.DataFrame) -> pd.Series:
        all_isbns = set(train_df['isbn'].unique())
        rated_isbns = set(train_df[train_df['user_id'] == user_id]['isbn'].unique())
        unrated_books = all_isbns - rated_isbns
        predictions = [(model.predict(uid=user_id, iid=isbn).iid, model.predict(uid=user_id, iid=isbn).est) for isbn in unrated_books]
        isbns = [pred[0] for pred in predictions]
        scores = [pred[1] for pred in predictions]
        recommendations = pd.Series(scores, index=isbns)
        return recommendations.sort_values(ascending=False)

    @staticmethod
    # Predicts a specific rating for a user-item pair using KNN neighbors.
    def predict_knn_rating(user_idx, item_idx, model, train_matrix, k):
        user_vector = train_matrix[user_idx]
        _, indices = model.kneighbors(user_vector, n_neighbors=k + 1)
        neighbor_indices = indices.flatten()[1:]
        neighbor_ratings = train_matrix[neighbor_indices, item_idx].toarray().flatten()
        valid_ratings = neighbor_ratings[neighbor_ratings > 0]
        return np.mean(valid_ratings) if valid_ratings.size > 0 else train_matrix.data.mean()

    @staticmethod
    # Evaluates KNN model performance using Precision@K, Recall, and RMSE.
    def evaluate_knn_model(model, train_matrix, test_matrix, user_item_matrix, k=10, rating_threshold=8.0):
        all_precisions, all_recalls = [], []
        actual_ratings, predicted_ratings = [], []
        for user_idx in range(test_matrix.shape[0]):
            user_id = user_item_matrix.index[user_idx]
            relevant_items = set(user_item_matrix.columns[test_matrix[user_idx].indices[test_matrix[user_idx].data >= rating_threshold]])
            if not relevant_items: continue
            recs = TrainModel.user_based_knn_prediction(user_id, model, user_item_matrix, train_matrix, k=k)
            recommended_items = {isbn for isbn, score in recs.items()}
            hits = len(recommended_items.intersection(relevant_items))
            all_precisions.append(hits / k)
            all_recalls.append(hits / len(relevant_items))
            
        avg_precision = np.mean(all_precisions) if all_precisions else 0
        avg_recall = np.mean(all_recalls) if all_recalls else 0
        test_user_indices, test_item_indices = test_matrix.nonzero()
        for u_idx, i_idx in zip(test_user_indices, test_item_indices):
            actual_ratings.append(test_matrix[u_idx, i_idx])
            predicted_ratings.append(TrainModel.predict_knn_rating(u_idx, i_idx, model, train_matrix, k))
            
        mse = mean_squared_error(actual_ratings, predicted_ratings) if actual_ratings else 0
        return {"rmse": np.sqrt(mse), "precision": avg_precision, "recall": avg_recall}

    @staticmethod
    # Evaluates SVD model performance using Precision@K, Recall, and RMSE.
    def evaluate_svd_model(model: SVD, testset, k=10, rating_threshold=8.0):
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        user_predictions = defaultdict(list)
        user_ground_truth = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            user_predictions[uid].append((iid, est))
            if true_r >= rating_threshold:
                user_ground_truth[uid].append(iid)

        all_precisions, all_recalls = dict(), dict()
        for uid, user_preds in user_predictions.items():
            if not user_ground_truth[uid]: continue
            user_preds.sort(key=lambda x: x[1], reverse=True)
            recommended_items = {iid for (iid, est) in user_preds[:k]}
            ground_truth_items = set(user_ground_truth[uid])
            hits = len(recommended_items.intersection(ground_truth_items))
            all_precisions[uid] = hits / k
            all_recalls[uid] = hits / len(ground_truth_items)
            
        avg_precision = sum(prec for prec in all_precisions.values()) / max(1, len(all_precisions))
        avg_recall = sum(rec for rec in all_recalls.values()) / max(1, len(all_recalls))
        return {"rmse": rmse, "precision": avg_precision, "recall": avg_recall}

    @staticmethod
    # Predicts a specific rating for a user-item pair using ALS latent factors.
    def predict_als_rating(user_idx, item_idx, model):
        user_vector = model.user_factors[user_idx]
        item_vector = model.item_factors[item_idx]
        return user_vector.dot(item_vector)

    @staticmethod
    # Evaluates ALS model performance using Precision@K, Recall, and RMSE.
    def evaluate_als_model(model, train_matrix_T, test_matrix, user_item_matrix, k=10, rating_threshold=8.0):
        all_precisions, all_recalls = [], []
        actual_ratings, predicted_ratings = [], []
        for user_idx in range(test_matrix.shape[0]):
            user_id = user_item_matrix.index[user_idx]
            relevant_items = set(user_item_matrix.columns[test_matrix[user_idx].indices[test_matrix[user_idx].data >= rating_threshold]])
            if not relevant_items: continue
            recs = TrainModel.user_based_als_prediction(user_id, model, user_item_matrix, train_matrix_T)
            recommended_items = {isbn for isbn, score in recs.head(k).items()}
            hits = len(recommended_items.intersection(relevant_items))
            all_precisions.append(hits / k)
            all_recalls.append(hits / len(relevant_items))
            
        avg_precision = np.mean(all_precisions) if all_precisions else 0
        avg_recall = np.mean(all_recalls) if all_recalls else 0
        test_user_indices, test_item_indices = test_matrix.nonzero()
        for u_idx, i_idx in zip(test_user_indices, test_item_indices):
            actual_ratings.append(test_matrix[u_idx, i_idx])
            predicted_ratings.append(TrainModel.predict_als_rating(u_idx, i_idx, model))
            
        mse = mean_squared_error(actual_ratings, predicted_ratings) if actual_ratings else 0
        return {"rmse": np.sqrt(mse), "precision": avg_precision, "recall": avg_recall}


class ModelsHyperparametersImprovment:
    @staticmethod
    # Performs randomized search to find optimal hyperparameters for Random Forest.
    def context_based_radom_forest_hyperparameters_improvement(x_train: pd.DataFrame, y_train: pd.Series):
        param_grid = {"n_estimators": [100, 200, 300], 'max_depth': [10, 20, None]}
        model = RandomForestRegressor(random_state=42)
        randon_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=8, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
        randon_search.fit(x_train, y_train)
        return randon_search.best_estimator_
    
    @staticmethod
    # Performs randomized search to find optimal hyperparameters for XGBoost.
    def context_based_XBGgradient_boosting_hyperparameters_improvment(x_train: pd.DataFrame, y_train: pd.Series):
        param_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        model = XGBRegressor(random_state=42)
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=8, cv=3, scoring="neg_mean_squared_error", random_state=42, n_jobs=-1)
        search.fit(x_train, y_train)
        return search.best_estimator_
    
    @staticmethod
    # Performs randomized search to find optimal hyperparameters for LightGBM.
    def context_based_LGBgradient_boostin_hyperparameters_improvment(x_train: pd.DataFrame, y_train: pd.Series):
        param_grid = {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "num_leaves": [20, 31]}
        model = LGBMRegressor(random_state=42)
        search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=8, cv=3, scoring="neg_mean_squared_error", random_state=42, n_jobs=-1)
        search.fit(x_train, y_train)
        return search.best_estimator_

    @staticmethod
    # Finds the optimal 'k' value and retrains the KNNBasic model.
    def tune_knn_model(data):
        trainset, testset = surprise_split(data, test_size=0.2)
        best_rmse = float('inf')
        best_k = 10
        for k in [10, 20, 30]:
            model = KNNBasic(k=k, sim_options={'user_based': True}, verbose=False)
            model.fit(trainset)
            predictions = model.test(testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            if rmse < best_rmse:
                best_rmse, best_k = rmse, k
        final_model = KNNBasic(k=best_k, sim_options={'user_based': True})
        final_model.fit(data.build_full_trainset())
        return final_model

    @staticmethod
    # Performs grid search to find optimal hyperparameters for the SVD model.
    def tune_svd_model(data):
        param_grid = {'n_factors': [50, 100], 'n_epochs': [20, 30]}
        gs = SurpriseGridSearch(SVD, param_grid, measures=['rmse'], cv=3)
        gs.fit(data)
        final_model = SVD(**gs.best_params['rmse'])
        final_model.fit(data.build_full_trainset())
        return final_model

    @staticmethod
    # Optimizes ALS model parameters using precision-based evaluation.
    def tune_als_model(train_matrix):
        train, validate = implicit_split(train_matrix, split_count=2, split_by='user')
        param_grid = {'factors': [30, 50], 'iterations': [15, 20]}
        best_score = -1
        best_params = {}
        for params in [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]:
            model = AlternatingLeastSquares(**params)
            model.fit(train.T)
            score = precision_at_k(model, train_user_items=train, test_user_items=validate, K=10)
            if score > best_score:
                best_score, best_params = score, params
        final_model = AlternatingLeastSquares(**best_params)
        final_model.fit(train_matrix.T)
        return final_model

class ModelOrganaize:
    @staticmethod
    # Compares multiple regression models and saves the best performing one.
    def compare_regression_model(linear_regression, random_tree_regression, extrem_gradient_boosting, light_gradient_boosting, y_test: pd.Series):
        comparation = {
            "linear_regression": TrainModel.context_based_models_evaluetion(linear_regression[1], y_test),
            "random_tree_regression": TrainModel.context_based_models_evaluetion(random_tree_regression[1], y_test),
            "extrem_gradient_boosting": TrainModel.context_based_models_evaluetion(extrem_gradient_boosting[1], y_test),
            "light_gradient_boosting": TrainModel.context_based_models_evaluetion(light_gradient_boosting[1], y_test)
        }
        best_rmse_val, best_model_name = float('inf'), None
        for model_name, eval_dict in comparation.items():
            if eval_dict['rmse'] < best_rmse_val:
                best_rmse_val, best_model_name = eval_dict['rmse'], model_name
                
        mapping = {
            "linear_regression": linear_regression,
            "random_tree_regression": random_tree_regression,
            "extrem_gradient_boosting": extrem_gradient_boosting,
            "light_gradient_boosting": light_gradient_boosting
        }
        joblib.dump(mapping[best_model_name][0], f"{best_model_name}_model.joblib")
        return mapping[best_model_name]
    
    @staticmethod
    # Compares KNN, SVD, and ALS models and saves the best performing one.
    def compare_user_based_models(ratings_df: pd.DataFrame, train_matrix: csr_matrix, test_matrix: csr_matrix, user_item_matrix: pd.DataFrame, k: int = 10):
        knn_model = TrainModel.knn_user_based_model(train_matrix)
        svd_model, _, svd_testset = TrainModel.svd_user_based_model(ratings_df)
        als_model = TrainModel.als_user_based_model(train_matrix)

        knn_scores = TrainModel.evaluate_knn_model(knn_model, train_matrix, test_matrix, user_item_matrix, k=k)
        svd_scores = TrainModel.evaluate_svd_model(svd_model, svd_testset, k=k)
        als_scores = TrainModel.evaluate_als_model(als_model, train_matrix.T, test_matrix, user_item_matrix, k=k)

        trained_models = {"KNN": knn_model, "SVD": svd_model, "ALS": als_model}
        evaluations = {"KNN": knn_scores, "SVD": svd_scores, "ALS": als_scores}
        
        best_precision, best_model_name = -1.0, None
        for name, scores in evaluations.items():
            if scores['precision'] > best_precision:
                best_precision, best_model_name = scores['precision'], name
                
        joblib.dump(trained_models[best_model_name], f"best_user_based_{best_model_name}_model.joblib")
        return trained_models[best_model_name]

class HybridRecommender:
    # Initializes the HybridRecommender with context and user models and data.
    def __init__(self, context_based_model_and_prediction: Tuple[BaseEstimator, np.array], user_based_model: BaseEstimator,
                  user_item_matrix: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, data_df: pd.DataFrame, user_id: int):
        self.context_based_model = context_based_model_and_prediction[0]
        self.user_based_model = user_based_model
        self.user_item_matrix = user_item_matrix
        self.data_df = data_df
        self.user_id = user_id

    @staticmethod
    # Calculates a sigmoid-based weight for hybrid recommendation balancing.
    def custom_growth_curved(x, midpoint=10, steepness=0.3):
        return np.clip((1/(1+np.exp(-steepness * (x - midpoint))) - 1/(1+np.exp(steepness * midpoint))) / (1 - 1/(1+np.exp(steepness * midpoint))), 0, 1)

    # Identifies unread books as candidates for recommendation.
    def book_sample_recommmend(self) -> pd.DataFrame:
        user_books_df = self.data_df[self.data_df["user_id"] == self.user_id]["isbn"].unique()
        all_books = self.data_df["isbn"].unique()
        unread_books = np.setdiff1d(all_books, user_books_df)
        return pd.DataFrame({"isbn": unread_books})
    
    # Prepares and scales candidate book features for the context model.
    def get_data_frame(self, sample_df: pd.DataFrame) -> pd.DataFrame:
        ratings = joblib.load("cleaned_data_dict.joblib")
        books = ratings["Books"]
        users = ratings["Users"]
        
        df = sample_df.merge(books, on="isbn", how="left")
        df["user_id"] = self.user_id
        df = df.merge(users, on="user_id", how="left")
        
        prediction_df = FeaturesEngineer.hybrid_context_based_features_engineer(df)
        
        encoder = joblib.load("hashing_encoder.joblib")
        scaler = joblib.load("standard_scaler.joblib")
        cat_cols = prediction_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            prediction_df = encoder.transform(prediction_df)
        scaled_features = scaler.transform(prediction_df)
        
        sample_df["context_score"] = self.context_based_model.predict(scaled_features)
        return sample_df

    # Determines the hybrid model weight based on user interaction history.
    def weight_per_model(self, df: pd.DataFrame) -> float:
        user_ratings_count = self.data_df[self.data_df["user_id"] == self.user_id].shape[0] if "user_id" in self.data_df.columns else 0
        return self.custom_growth_curved(user_ratings_count)

    # Combines context and user scores into a final weighted recommendation list.
    def recommend(self, alpha: float, df: pd.DataFrame) -> pd.DataFrame:
        # User Based Logic
        try:
            train_matrix = joblib.load("train_matrix.joblib")
            user_recs = TrainModel.user_based_knn_prediction(self.user_id, self.user_based_model, self.user_item_matrix, train_matrix)
        except Exception:
            user_recs = pd.Series(dtype=float)
            
        df["user_score"] = df["isbn"].map(user_recs).fillna(0)
        df["final_score"] = alpha * df["user_score"] + (1 - alpha) * df["context_score"]
        return df.sort_values("final_score", ascending=False).head(10)
