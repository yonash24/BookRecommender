import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from typing import Dict, Tuple
import logging
from category_encoders import HashingEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)


"""
create a class that import the data into 
a new directory and return as a dictionary of dataFrames
"""
class importData:
    
    #func that import the Goodreads-books dataset from kaggle into new dictionary named data_dir
    @staticmethod
    def  import_data():
        Path("data_dir").mkdir(exist_ok=True)
        cur_path = Path.cwd()
        dest_path = cur_path / "data_dir"
        if dest_path.exists():
            print("data_dir already exist")
            return
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("arashnic/book-recommendation-dataset",path=dest_path,unzip=True)
        

    #delete unnecesarry files (like images)
    @staticmethod
    def delet_unnececatty_files():
        dir_path = Path("data_dir")
        if not dir_path:
            print(f"theres no such sirectory{dir_path}")
            return
        for file in dir_path.glob("*png"):
                try:
                    file.unlink()
                except FileNotFoundError:
                    print("there is no files in the directory")
                except Exception as e:
                    print(f"an error occured{e}")

    #transfer the actuall files we will work with into dataframes in a dictionary
    @staticmethod
    def to_dataFrme():
        data_dict = {}
        dir_path = Path("data_dir")
        csv_files = list(dir_path.glob("*.csv"))
        for file in csv_files:
            file_name = file.stem
            df = pd.read_csv(file,encoding="latin-1")
            data_dict[file_name] = df
        return data_dict
    
    #create pupeline to import the data
    @staticmethod
    def import_data_pipeline():
        importData.import_data()
        importData.delet_unnececatty_files()
        data_dict = importData.to_dataFrme()
        return data_dict




"""
create class to get info on the data 
to understand the data and handling it
the function in the class get a dictionary of data frames from the class importData
"""
class GetDataInfo:

### Core Data Characteristics

    #get the data shape
    @staticmethod
    def get_data_shap(data_dict:Dict[str,pd.DataFrame]):
        for file,df in data_dict.items():
            df_shap = df.shape
            print(f"the dataFrame shape of {file} is {df_shap}")

    #get the the data info
    @staticmethod
    def get_data_info(data_dict:Dict[str,pd.DataFrame]):
        for file, df in data_dict.items():
            print(f"the dataFrame {file} info is:")
            df.info()

    #get info about missing values in each col
    @staticmethod
    def get_missing_vals(data_dict:Dict[str,pd.DataFrame]):
        for file,df in data_dict.items():
                missing_val = df.isnull().sum()
                missing_df = missing_val[missing_val > 0]
                if missing_df.empty:
                    print("no missing values")
                else:
                    for col,val in missing_df.items():
                        print(f"int col {col} there are {val} missing values")

### Ratings Analysis

    #see the Distribution of All Rating Values by bar chart 
    @staticmethod
    def get_rating_destribution(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Ratings"]
        rating_dist = df["book_rating"].value_counts()
        final_data = rating_dist.sort_index()
        final_data.plot(kind="bar")
        plt.title("rating values distribution")
        plt.xlabel("rating values")
        plt.ylabel("vals distribution")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    #see the mean rate for each book
    @staticmethod
    def book_mean_rate(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Ratings"]
        plot_df = df.groupby("isbn")["book_rating"].mean()
        plt.hist(plot_df, bins=20, color="skyblue", edgecolor="black")
        plt.title("book mean rate")
        plt.xlabel("mean rating")
        plt.ylabel("frequency")
        plt.grid(True,linestyle="--", alpha=0.7)
        plt.show()

    #see the median rate for each book
    @staticmethod
    def book_median_rate(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Ratings"]
        plot_df = df.groupby("isbn")["book_rating"].median()
        plt.hist(plot_df, bins=20, color="skyblue", edgecolor="black")
        plt.title("book median rate")
        plt.xlabel("book isbn")
        plt.ylabel("frequency")
        plt.grid(True,linestyle="--", alpha=0.7)
        plt.show()

### User Analysis
    
    #get user age distribution
    @staticmethod
    def user_mean_age(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Users"]
        plot_df = df["age"].dropna()
        plt.hist(plot_df, bins=20, color="skyblue", edgecolor="black")
        plt.title("user age distribution")
        plt.xlabel("age")
        plt.ylabel("num of users")
        plt.grid(True,linestyle="--", alpha=0.7)
        plt.show()

    #get the locations with the highest amount of raters
    def highest_raters_location(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Users"]
        location_df = df["location"].value_counts().head(20)
        location_df.plot(kind="bar",color="skyblue", edgecolor="black")
        plt.title("top raters location")
        plt.xlabel("location")
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("amount of raters")
        plt.tight_layout() 
        plt.grid(axis='y',linestyle="--", alpha=0.7)
        plt.show()

### Book Analysis

    # get the  Range of Publication Years 
    @staticmethod
    def year_range(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Books"]
        min_year = df["year_of_publication"].min()
        max_year = df["year_of_publication"].max()
        logging.info(f"the min year is:{min_year} n\ the max year is: {max_year}")
        return min_year, max_year

    #get the Number of Unique Authors and publishers
    @staticmethod
    def unique_authors_and_publishers(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Books"]
        publishers_amount = df["publisher"].nunique()
        writers_amount = df["book_author"].nunique()
        logging.info(f"unique publishers amount {publishers_amount}, unique writers amount {writers_amount}")
        return publishers_amount, writers_amount

### Popularity & Quality Insights

    #get the most popular books
    def highest_raters_location(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Ratings"]
        rates_df = df["isbn"].value_counts().head(20)
        rates_df.plot(kind="bar",color="skyblue", edgecolor="black")
        plt.title("most popular books")
        plt.xlabel("books")
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("amount of readers")
        plt.tight_layout() 
        plt.grid(axis='y',linestyle="--", alpha=0.7)
        plt.show()

    #see the top 20 rated books by mean rate for each book
    @staticmethod
    def book_mean_rate(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Ratings"]
        plot_df = df.groupby("isbn")["book_rating"].mean()
        top_books = plot_df.sort_values(ascending=False).head(20)
        top_books.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("book mean rate")
        plt.xlabel("book")
        plt.ylabel("mean rate")
        plt.grid(axis='y',linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    #see the most actives users
    @staticmethod
    def most_active_users(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Ratings"]
        rates_df = df["user_id"].value_counts().head(20)
        rates_df.plot(kind="bar",color="skyblue", edgecolor="black")
        plt.title("most acite users")
        plt.xlabel("user id")
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("amount of ratings")
        plt.tight_layout() 
        plt.grid(axis='y',linestyle="--", alpha=0.7)
        plt.show()

"""
create a class that clean the data 
"""
class DataClean:

    #drop unrelevant columns
    @staticmethod
    def drop_unrelevat_cols(data_dict:Dict[str,pd.DataFrame])->dict:
        cols_to_keep = ["isbn","book_title","book_author","year_of_publication","publisher"]
        if "Books" in data_dict:
            data_dict["Books"] = data_dict["Books"][cols_to_keep]
            logging.info("drop unrelevat cols from Books, ratings and users dont have unrelevat cols")
        else:
            logging.warning("Books not found in data_dict")
        return data_dict
    
    
    #standarize all the cols heads making them lower case without special chars
    @staticmethod
    def cols_heads_standart(data_dict:Dict[str,pd.DataFrame])->dict:
        for file, df in data_dict.items():
            original_col = df.columns.to_list()
            cols = df.columns
            cols = cols.str.lower()
            cols = cols.str.replace(' ', '_', regex=False)
            cols = cols.str.replace('-', '_', regex=False)
            cols = cols.str.replace(r'[^a-z0-9_]', '', regex=True)
            df.columns = cols
        logging.info("cols head transfefr to lower case seperate by '_' and witohut specil chars")
        return data_dict
    
    #clean the data from Books dataFrame 
    #handle missing values
    #handle with incorrect vals in a row

    ### make sure to use it before the func cols_heads_standart!!!!!! ###
    @staticmethod
    def clean_books_df(data_dict:Dict[str,pd.DataFrame])->dict:
        fill_vals = {
            "isbn" : "unKnown",
            "book_title" : "unKnown",
            "book_author" : "unKnown",
            "year_of_publication" : 0,
            "publisher" : "unKnown"
        }
        data_dict["Books"] = data_dict["Books"].fillna(value=fill_vals)
        logging.info("filled missing values in Books df")

        df = data_dict["Books"]
        four_digit_year_pattern = r"^\d{4}$"
        year_df = df["year_of_publication"].astype(str)
        is_valid_year_mask = year_df.str.match(four_digit_year_pattern)
        df.loc[~is_valid_year_mask, 'year_of_publication'] = 0
        logging.info("Corrected invalid entries in 'year_of_publication' column.")
        df['year_of_publication'] = pd.to_numeric(df['year_of_publication'])
        data_dict['Books'] = df
        return data_dict
    

    #clean the data from Users dataFrame 
    #handle missing values
    #remove unrizenable ages below 5 and above 100

    ### make sure to use it before the func cols_heads_standart!!!!!! ###
    def clean_users_df(data_dict:Dict[str,pd.DataFrame])->dict:
        fill_vals = {
            "user_id" : "unKnown",
            "location" : "unKnown",
            "age" : 0
        }
        data_dict["Users"] = data_dict["Users"].fillna(value=fill_vals)
        logging.info("filled missing values in Users df")

        df = data_dict["Users"]
        fileted_df = df[(df["age"] < 100) & (df["age"] > 5)]
        data_dict["Users"] = fileted_df
        logging.info("filter unrizenable ages from the dataFrame")
        return data_dict
    
    
    #clean the data from Ratings dataFrame 
    #handle missing values
    #remove unrizenable ages below 5 and above 100

    ### make sure to use it before the func cols_heads_standart!!!!!! ###
    def clean_ratings_df(data_dict:Dict[str,pd.DataFrame])->dict:
        fill_vals = {
            "user_id" : 0,
            "isbn" : "unKnown",
            "book_rating" : 0
        }
        data_dict["Ratings"] = data_dict["Ratings"].fillna(value=fill_vals)
        logging.info("filled missing values in Ratings df")

        df = data_dict["Ratings"]
        fileted_df = df[df["book_rating"] > 0]
        data_dict["Ratings"] = fileted_df
        logging.info("filter unrated books from the dataFrame")
        return data_dict
    
    #make every column in the dataframe to a specific type 
    #ensure a numeric col is numeric a string col is object etc
    ### use right after drop_unrelevat_cols ###
    @staticmethod
    def data_ensure_type(data_dict:Dict[str,pd.DataFrame]):
        col_type = {
            "Users" : {"user_id" : "Int64", "location" : "object", "age" : "Int64"},
            "Books" : {"isbn" : "object", "book_title" : "object", "book_author" : "object", "year_of_publication" : "Int64", "publisher" : "object"},
            "Ratings" : {"user_id" : "Int64", "isbn" : "object", "book_rating" : "Int64"}
        }

        for file, df in data_dict.items():
            if file in col_type:
                try:
                    cur_type = col_type[file]
                    data_dict[file] = data_dict[file].astype(cur_type)
                    logging.info(f"the cols in file {file} have successfully enforced")
                except Exception as e:
                    logging.error(f"Could not enforce types for '{file}'. Error: {e}")


        return data_dict
    
    #delete duplicate rows
    @staticmethod
    def delete_dups(data_dict: Dict[str,pd.DataFrame])->dict:
        for file, df in data_dict.items():
            data_dict[file] = df.drop_duplicates()
            logging.info("erased all duplicades in th data frames")
        return data_dict
    
    #uniform the text in object type cols to be uniformity
    #lower case letters seperated by "_" and witohut special chars
    @staticmethod
    def uniform_object_cols(data_dict:Dict[str,pd.DataFrame])->Dict[str,pd.DataFrame]:
        for file, df in data_dict.items():
            for col in df:
                if df[col].dtype == "object":
                        try:
                            df[col] = (df[col].str.lower()
                            .str.replace('-', '_')
                            .str.replace(' ', '_')
                            .str.replace(',', '_')
                            .str.replace(r'[^a-z0-9_]', '', regex=True)
                            .str.strip('_'))  
                            logging.info(f"uniformed the text in col {col} at data frame {file}")
                        except Exception as e:
                            logging.error(f"error occured in col {col} at data frame {file} coulnt fit the text")

        return data_dict
    
    #create a data cleaning pipeline with all the functions in the class
    @staticmethod
    def cleaning_data_pipeline(data_dict:Dict[str,pd.DataFrame])->Dict[str,pd.DataFrame]:
        data_dict = DataClean.cols_heads_standart(data_dict)
        data_dict = DataClean.drop_unrelevat_cols(data_dict)
        data_dict = DataClean.clean_users_df(data_dict)
        data_dict = DataClean.clean_books_df(data_dict)
        data_dict = DataClean.clean_ratings_df(data_dict)
        data_dict = DataClean.delete_dups(data_dict)
        data_dict = DataClean.uniform_object_cols(data_dict)
        data_dict = DataClean.data_ensure_type(data_dict)
        logging.info("data cleaned successfully")
        return data_dict
    

    """
    create a class to preprocess the data and preper
    the data to fit to the ML models
    """

class DataPreProcess:
    
### preprocess the data for context based model ###
    
    #create user item matrix with rows user_id cols isbn and rating calues that calculate the mean rating
    @staticmethod
    def user_item_matrix(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Ratings"]
        pivote_df = df.pivot_table(
            index="user_id",
            columns="isbn",
            values="book_rating",
            aggfunc="mean",
            fill_value=0
        )
        logging.info("successfully created user item matrix")
        return pivote_df

### preprocess context based model ###

    #create the context based data frame for the model
    @staticmethod
    def context_based_df(data_dict:Dict[str,pd.DataFrame])->pd.DataFrame:
        try:
            rating_df = data_dict["Ratings"]
            book_df = data_dict["Books"]
            user_df = data_dict["Users"]
            merged_df1 = pd.merge(rating_df,user_df,on="user_id")
            merged_df2 = pd.merge(merged_df1,book_df, on="isbn")
            logging.info("merged the 3 clean data frames in the data dictionary")

            merged_df2["book_mean_rate"] = merged_df2.groupby("isbn")["book_rating"].transform("mean")
            merged_df2["rating_count"] = merged_df2.groupby("isbn")["book_rating"].transform("count")
            logging.info("added to the data rfame mean moovie rate and rating count")

            return merged_df2
        
        except Exception as e:
            logging.error("possibly the dict keys are lower cases, returning empty data frame")
            return pd.DataFrame()
        
    #filter the data frame from books how got less then 3 rates and users how rates less then 2 books
    @staticmethod
    def filter_df(df:pd.DataFrame)->pd.DataFrame:
        min_user_rate = 3
        min_book_rate = 3
        user_count = df["user_id"].value_counts()
        books_count = df["isbn"].value_counts()
        
        active_user = user_count[user_count >= min_user_rate].index        
        active_book = books_count[books_count >= min_book_rate].index        
        
        df = df[df["user_id"].isin(active_user)]
        logging.info("removed all users who rated less then 3 books")
        df = df[df["isbn"].isin(active_book)]
        logging.info("removed all books who have less then 3 rates")

        cols_to_drop = ["user_id","isbn","book_title"]
        final_df = df.drop(cols_to_drop, axis=1)
        logging.info("drop unrelevant columns for the next step")
        return final_df
    
    #encode the data frame from context_based_df
    #col "location" woth one hot encoding
    #cols "publishers" and "author" with hashing encoding
    @staticmethod
    def context_based_encoding(df:pd.DataFrame)->pd.DataFrame:
        location_encoded_df = pd.get_dummies(df,columns=["location"], drop_first=True)
        logging.info("encoded loaction with one hot encoding")
        encoder = HashingEncoder(n_components=64,cols=["book_author","publisher"])
        hash_encoded_df = encoder.fit_transform(location_encoded_df)
        logging.info("hash encoding successfully")
        return hash_encoded_df
    
    #split the data into data frame features and target vector based on context based dataf rame
    #get data frame from context_based_df
    @staticmethod
    def featurs_target_splitting(df:pd.DataFrame)->Tuple[pd.DataFrame,pd.Series]:
        target_vector = df["book_rating"]
        features_df = df.drop("book_rating",axis=1)
        logging.info("split the data into features data frame and target vector")
        return features_df, target_vector
        
    #creating a pipeline for the context based model data preprocessing
    @staticmethod
    def context_based_data_preprocessing_pipeline(data_dict:Dict[str,pd.DataFrame])->Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
        context_dict = DataPreProcess.context_based_df(data_dict)
        filter_df = DataPreProcess.filter_df(context_dict)
        encoding_df = DataPreProcess.context_based_encoding(filter_df)
        initial_splitting = DataPreProcess.featurs_target_splitting(encoding_df)
        final_data = DataPreProcess.split_data(initial_splitting)
        return final_data

### general preprocessing functions for all the data that need to be preprocessed ###

    #split data into train and test 
    @staticmethod
    def split_data(data:Tuple[pd.DataFrame, pd.Series])->Tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
        try:    
            features = data[0]
            target = data[1]
            x_train, x_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            logging.info("splitted the data into train and tets data")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error("failed to split the data into train and test")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
        
    