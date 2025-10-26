from DataHandler import importData, DataClean, DataPreProcess, GetDataInfo
from MLModels import TrainModel, ModelsHyperparametersImprovment, ModelOrganaize, HybridRecommender
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import NearestNeighbors
from surprise import SVD
from implicit.als import AlternatingLeastSquares
import logging

logging.basicConfig(level=logging.INFO)


def main():

    data_dict = importData.import_data_pipeline()#import the data as a dictionary
    clean_data = DataClean.cleaning_data_pipeline(data_dict)#clean the data
    hybride_df = DataPreProcess.context_based_df(clean_data)

    svd_data = DataPreProcess.user_based_data_svd(clean_data)# data for svd model
    als_knn_data = DataPreProcess.user_based_data_als_knn(clean_data)# data for als and knn models
    regression_data = DataPreProcess.context_based_data_preprocessing_pipeline(clean_data)# data for regression models

    #access the data
    x_train_als_knn = als_knn_data[0]
    x_test_als_knn = als_knn_data[1]
    user_item_matrix = als_knn_data[2]

    x_train_svd = svd_data[0]
    x_test_svd = svd_data[1]

    x_train_regression = regression_data[0]
    x_test_regression = regression_data[1]
    y_train_regression = regression_data[2]
    y_test_regression = regression_data[3]

    ##train and predict the models##

    #train and predict the context based models
    linear_regression = TrainModel.context_based_linear_regression_model(x_train_regression, y_train_regression)
    random_tree_forest =TrainModel.context_based_radom_tree_regression(x_train_regression, y_train_regression)
    XGB_gradient_boosting = TrainModel.XBG_gradient_boosting_model(x_train_regression, y_train_regression)
    light_gradient_boosting = TrainModel.Light_gradient_boosting_model(x_train_regression, y_train_regression)

    linear_regression_model = linear_regression[0]
    linear_regression_prediction = linear_regression[1]
    random_tree_forest_model = random_tree_forest[0]
    random_tree_forest_prediction = random_tree_forest[1]
    XGB_gradient_boosting_model = XGB_gradient_boosting[0]
    XGB_gradient_boosting_prediction = XGB_gradient_boosting[1]
    light_gradient_boosting_model = light_gradient_boosting[0]
    light_gradient_boosting_prediction = light_gradient_boosting[1]

    #train and predict user based models
    knn_model = TrainModel.user_based_knn_prediction(x_train_als_knn)
    svd_model = TrainModel.svd_user_based_model(x_train_svd)
    als_model = TrainModel.als_user_based_model(x_train_als_knn)

    """
    to create the user based models predictions i need to create user class
    that handle user in the system
    """

    ### evaluate the models ###

    # context based models evaluation
    linear_regression_evaluate = TrainModel.context_based_models_evaluetion(linear_regression_prediction, y_test_regression)
    random_tree_forest_evaluate = TrainModel.context_based_models_evaluetion(random_tree_forest_prediction, y_test_regression)
    XGB_gradient_boosting_evaluate = TrainModel.context_based_models_evaluetion(XGB_gradient_boosting_prediction, y_test_regression)
    light_gradient_boosting_evaluate = TrainModel.context_based_models_evaluetion(light_gradient_boosting_prediction, y_test_regression)

    # user based models evaluation
    """
    to evaluate the models create user class 
    that handle with register and create a new user for the system 
    """
    

    ### get the best models of each ###
    svd_df = clean_data["Ratings"]
    
    choosen_context_based_model = ModelOrganaize.compare_regression_model(linear_regression_model, random_tree_forest_model, XGB_gradient_boosting_model,
                                                                          light_gradient_boosting_model, y_test_regression)
    
    choosen_user_based_model = ModelOrganaize.compare_user_based_models(svd_df, x_train_als_knn, x_test_als_knn, user_item_matrix)

    ### models hyperparameters improvment ###

    #context based_model improvment
    if isinstance(choosen_context_based_model, RandomForestRegressor):
        context_improved_model = ModelsHyperparametersImprovment.context_based_radom_forest_hyperparameters_improvement(x_train_regression, y_train_regression)
    elif isinstance(choosen_context_based_model, XGBRegressor):
        context_improved_model = ModelsHyperparametersImprovment.context_based_XBGgradient_boosting_hyperparameters_improvment(x_train_regression, y_train_regression)
    else: # its light grdient boosting
        context_improved_model = ModelsHyperparametersImprovment.context_based_LGBgradient_boostin_hyperparameters_improvment(x_train_regression, y_train_regression)

    logging.info("context based model hyper parameters improved")

    # user based hyperparameters improvment
    if isinstance(choosen_user_based_model, NearestNeighbors):
        user_improved_model = ModelsHyperparametersImprovment.tune_knn_model()
        # complete the row above
    elif isinstance(choosen_user_based_model, AlternatingLeastSquares):
        user_improved_model = ModelsHyperparametersImprovment.tune_als_model()
        #complete the row above
    else: #its svd
        user_improved_model = ModelsHyperparametersImprovment.tune_svd_model()
        #complete the row above

    ### reavaluate the models ###

    #context based model reavaluation
    if isinstance(choosen_context_based_model, RandomForestRegressor):
      pass
    elif isinstance(choosen_context_based_model, XGBRegressor):
        pass
    else: #its light gradient boosting
        pass 

    #user based model reavaluation
    if isinstance(choosen_user_based_model, NearestNeighbors):
        user_improved_model = ModelsHyperparametersImprovment.tune_knn_model()
        # complete the row above
    elif isinstance(choosen_user_based_model, AlternatingLeastSquares):
        user_improved_model = ModelsHyperparametersImprovment.tune_als_model()
        #complete the row above
    else: #its svd
        user_improved_model = ModelsHyperparametersImprovment.tune_svd_model()
        #complete the row above


    ### create hybride recommend
    hybrid_recommender = HybridRecommender(choosen_context_based_model, choosen_user_based_model, user_item_matrix, 
                                           x_train_regression, x_test_regression, hybride_df)

    book_sample = hybrid_recommender.book_sample_recommmend()
    df_300 = hybrid_recommender.get_data_frame(hybride_df)
    alpha = hybrid_recommender.weight_per_model(df_300)
    recommendation = hybrid_recommender.recommend(alpha, df_300)

if __name__ == "__main__":
    main()