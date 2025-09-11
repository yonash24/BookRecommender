import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from typing import Dict
import logging

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
        rating_dist = df["Book-Rating"].value_counts()
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
        plot_df = df.groupby("ISBN")["Book-Rating"].mean()
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
        plot_df = df.groupby("ISBN")["Book-Rating"].median()
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
        plot_df = df["Age"].dropna()
        plt.hist(plot_df, bins=20, color="skyblue", edgecolor="black")
        plt.title("user age distribution")
        plt.xlabel("age")
        plt.ylabel("num of users")
        plt.grid(True,linestyle="--", alpha=0.7)
        plt.show()

    #get the locations with the highest amount of raters
    def highest_raters_location(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Users"]
        location_df = df["Location"].value_counts().head(20)
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
        min_year = df["Year-Of-Publication"].min()
        max_year = df["Year-Of-Publication"].max()
        logging.info(f"the min year is:{min_year} n\ the max year is: {max_year}")
        return min_year, max_year

    #get the Number of Unique Authors and Publishers
    @staticmethod
    def unique_authors_and_publishers(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Books"]
        publishers_amount = df["Publisher"].nunique()
        writers_amount = df["Book-Author"].nunique()
        logging.info(f"unique publishers amount {publishers_amount}, unique writers amount {writers_amount}")
        return publishers_amount, writers_amount

### Popularity & Quality Insights

    #get the most popular books
    def highest_raters_location(data_dict:Dict[str,pd.DataFrame]):
        df = data_dict["Ratings"]
        rates_df = df["ISBN"].value_counts().head(20)
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
        plot_df = df.groupby("ISBN")["Book-Rating"].mean()
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
        rates_df = df["User-ID"].value_counts().head(20)
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
        cols_to_keep = ["ISBN","Book-Title","Book-Author","Year-Of-Publication","Publisher"]
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
            "ISBN" : "unKnown",
            "Book-Title" : "unKnown",
            "Book-Author" : "unKnown",
            "Year-Of-Publication" : 0,
            "Publisher" : "unKnown"
        }
        data_dict["Books"] = data_dict["Books"].fillna(value=fill_vals)
        logging.info("filled missing values in Books df")

        df = data_dict["Books"]
        four_digit_year_pattern = r"^\d{4}$"
        year_df = df["Year-Of-Publication"].astype(str)
        is_valid_year_mask = year_df.str.match(four_digit_year_pattern)
        df.loc[~is_valid_year_mask, 'Year-Of-Publication'] = 0
        logging.info("Corrected invalid entries in 'Year-Of-Publication' column.")
        df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'])
        data_dict['Books'] = df
        return data_dict
    

    #clean the data from Users dataFrame 
    #handle missing values
    #remove unrizenable ages below 5 and above 100

    ### make sure to use it before the func cols_heads_standart!!!!!! ###
    def clean_users_df(data_dict:Dict[str,pd.DataFrame])->dict:
        fill_vals = {
            "User-ID" : "unKnown",
            "Location" : "unKnown",
            "Age" : 0
        }
        data_dict["Users"] = data_dict["Users"].fillna(value=fill_vals)
        logging.info("filled missing values in Users df")

        