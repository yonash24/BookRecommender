from DataHandler import importData, DataClean, DataPreProcess, GetDataInfo
from MLModels import TrainModel, ModelsHyperparametersImprovment, ModelOrganaize, HybridRecommender

def main():

    data_dict = importData.import_data_pipeline()#import the data as a dictionary
    clean_data = DataClean.cleaning_data_pipeline(data_dict)#clean the data

    svd_data = DataPreProcess.user_based_data_svd(clean_data)# data for svd model
    als_knn_data = DataPreProcess.user_based_data_als_knn(clean_data)# data for als and knn models
    regression_data = DataPreProcess.context_based_data_preprocessing_pipeline(clean_data)# data for regression models

    #access the data
    x_train_als_knn = als_knn_data[0]
    x_test_als_knn = als_knn_data[1]

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





if __name__ == "__main__":
    main()