from DataHandler import importData, DataClean, DataPreProcess, GetDataInfo

def main():

    data_dict = importData.import_data_pipeline()#import the data as a dictionary
    clean_data = DataClean.cleaning_data_pipeline(data_dict)#clean the data

    svd_data = DataPreProcess.user_based_data_svd(clean_data)# data for svd model
    als_knn_data = DataPreProcess.user_based_data_als_knn(clean_data)# data for als and knn models
    regression_data = DataPreProcess.context_based_data_preprocessing_pipeline(clean_data)# data for regression models

    x_train_als_knn = als_knn_data[0]
    x_test_als_knn = als_knn_data[1]

    x_train_svd = svd_data[0]
    x_test_svd = svd_data[1]

    x_train_regression = regression_data[0]
    x_test_regression = regression_data[1]
    y_train_regression = regression_data[2]
    y_test_regression = regression_data[3]

    







if __name__ == "__main__":
    main()