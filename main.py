import logging
import joblib

from DataHandler import importData, DataClean, DataPreProcess, FeaturesEngineer
from MLModels import TrainModel, ModelsHyperparametersImprovment, ModelOrganaize, HybridRecommender
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import NearestNeighbors
from surprise import SVD
from implicit.als import AlternatingLeastSquares

logging.basicConfig(level=logging.INFO)

class User:
    def __init__(self, user_id, location="unknown", age=30):
        self.user_id = user_id
        self.location = location
        self.age = age

def main():

    data_dict = importData.import_data_pipeline()
    clean_data = DataClean.cleaning_data_pipeline(data_dict)
    
    # Save clean data
    joblib.dump(clean_data, "cleaned_data_dict.joblib")
    
    hybride_df = DataPreProcess.context_based_df(clean_data)

    svd_data = DataPreProcess.user_based_data_svd(clean_data)
    als_knn_data = DataPreProcess.user_based_data_als_knn(clean_data)
    regression_data = DataPreProcess.context_based_data_preprocessing_pipeline(clean_data)

    x_train_als_knn, x_test_als_knn, user_item_matrix = als_knn_data
    x_train_svd, x_test_svd = svd_data
    x_train_regression, x_test_regression, y_train_regression, y_test_regression = regression_data

    # Save user item matrix
    joblib.dump(user_item_matrix, "user_item_matrix.joblib")

    # train and predict the context based models
    linear_regression = TrainModel.context_based_linear_regression_model(x_train_regression, x_test_regression, y_train_regression)
    random_tree_forest = TrainModel.context_based_radom_tree_regression(x_train_regression, x_test_regression, y_train_regression)
    XGB_gradient_boosting = TrainModel.XBG_gradient_boosting_model(x_train_regression, x_test_regression, y_train_regression)
    light_gradient_boosting = TrainModel.Light_gradient_boosting_model(x_train_regression, x_test_regression, y_train_regression)

    linear_regression_model, linear_regression_prediction = linear_regression
    random_tree_forest_model, random_tree_forest_prediction = random_tree_forest
    XGB_gradient_boosting_model, XGB_gradient_boosting_prediction = XGB_gradient_boosting
    light_gradient_boosting_model, light_gradient_boosting_prediction = light_gradient_boosting

    # user based models evaluation
    svd_df = clean_data["Ratings"]
    choosen_context_based_model = ModelOrganaize.compare_regression_model(
        linear_regression, random_tree_forest, XGB_gradient_boosting,
        light_gradient_boosting, y_test_regression
    )
    
    choosen_user_based_model = ModelOrganaize.compare_user_based_models(
        svd_df, x_train_als_knn, x_test_als_knn, user_item_matrix
    )

    # context based_model improvment
    context_improved_model = None
    if isinstance(choosen_context_based_model[0], RandomForestRegressor):
        context_improved_model = ModelsHyperparametersImprovment.context_based_radom_forest_hyperparameters_improvement(x_train_regression, y_train_regression)
    elif isinstance(choosen_context_based_model[0], XGBRegressor):
        context_improved_model = ModelsHyperparametersImprovment.context_based_XBGgradient_boosting_hyperparameters_improvment(x_train_regression, y_train_regression)
    else: 
        context_improved_model = ModelsHyperparametersImprovment.context_based_LGBgradient_boostin_hyperparameters_improvment(x_train_regression, y_train_regression)

    logging.info("context based model hyper parameters improved")
    joblib.dump(context_improved_model, "best_context_model.joblib")

    # user based hyperparameters improvment
    user_improved_model = None
    if isinstance(choosen_user_based_model, NearestNeighbors):
        # pass dummy reader data for surprise or sparse matrix
        user_improved_model = choosen_user_based_model 
    elif isinstance(choosen_user_based_model, AlternatingLeastSquares):
        user_improved_model = ModelsHyperparametersImprovment.tune_als_model(x_train_als_knn)
    else:
        # SVD model
        from surprise import Dataset, Reader
        reader = Reader(rating_scale=(1,10))
        data = Dataset.load_from_df(svd_df[["user_id", "isbn", "book_rating"]], reader)
        user_improved_model = ModelsHyperparametersImprovment.tune_svd_model(data)

    joblib.dump(user_improved_model, "best_user_based_model.joblib")

    logging.info("Pipeline Execution Completed.")

if __name__ == "__main__":
    main()