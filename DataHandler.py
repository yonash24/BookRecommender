import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from typing import Dict, Tuple
import logging
from category_encoders import HashingEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO)

class importData:
    @staticmethod
    # Downloads the book recommendation dataset from Kaggle.
    def import_data():
        Path("data_dir").mkdir(exist_ok=True)
        cur_path = Path.cwd()
        dest_path = cur_path / "data_dir"
        if dest_path.exists():
            print("data_dir already exist")
            return
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("arashnic/book-recommendation-dataset",path=dest_path,unzip=True)
        
    @staticmethod
    # Deletes unnecessary PNG files from the data directory.
    def delet_unnececatty_files():
        dir_path = Path("data_dir")
        if not dir_path.exists():
            print(f"theres no such directory {dir_path}")
            return
        for file in dir_path.glob("*png"):
            try:
                file.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"an error occured {e}")

    @staticmethod
    # Loads CSV files from the data directory into a dictionary of DataFrames.
    def to_dataFrme():
        data_dict = {}
        dir_path = Path("data_dir")
        csv_files = list(dir_path.glob("*.csv"))
        for file in csv_files:
            file_name = file.stem
            df = pd.read_csv(file, encoding="latin-1")
            data_dict[file_name] = df
        return data_dict
    
    @staticmethod
    # Runs the full pipeline to import and load the data.
    def import_data_pipeline():
        importData.import_data()
        importData.delet_unnececatty_files()
        data_dict = importData.to_dataFrme()
        return data_dict


class GetDataInfo:
    @staticmethod
    # Prints the shape of each DataFrame in the data dictionary.
    def get_data_shap(data_dict: Dict[str, pd.DataFrame]):
        for file, df in data_dict.items():
            print(f"the dataFrame shape of {file} is {df.shape}")

    @staticmethod
    # Prints the info summary of each DataFrame in the data dictionary.
    def get_data_info(data_dict: Dict[str, pd.DataFrame]):
        for file, df in data_dict.items():
            print(f"the dataFrame {file} info is:")
            df.info()

    @staticmethod
    # Prints the count of missing values for each column in the DataFrames.
    def get_missing_vals(data_dict: Dict[str, pd.DataFrame]):
        for file, df in data_dict.items():
            missing_val = df.isnull().sum()
            missing_df = missing_val[missing_val > 0]
            if missing_df.empty:
                print("no missing values")
            else:
                for col, val in missing_df.items():
                    print(f"in col {col} there are {val} missing values")

    @staticmethod
    # Visualizes the distribution of book ratings using a bar chart.
    def get_rating_destribution(data_dict: Dict[str, pd.DataFrame]):
        df = data_dict["Ratings"]
        rating_dist = df["book_rating"].value_counts().sort_index()
        rating_dist.plot(kind="bar")
        plt.title("rating values distribution")
        plt.xlabel("rating values")
        plt.ylabel("vals distribution")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    @staticmethod
    # Visualizes the distribution of mean ratings per book using a histogram.
    def book_mean_rate(data_dict: Dict[str, pd.DataFrame]):
        df = data_dict["Ratings"]
        plot_df = df.groupby("isbn")["book_rating"].mean()
        plt.hist(plot_df, bins=20, color="skyblue", edgecolor="black")
        plt.title("book mean rate")
        plt.xlabel("mean rating")
        plt.ylabel("frequency")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()

    @staticmethod
    # Visualizes the distribution of median ratings per book using a histogram.
    def book_median_rate(data_dict: Dict[str, pd.DataFrame]):
        df = data_dict["Ratings"]
        plot_df = df.groupby("isbn")["book_rating"].median()
        plt.hist(plot_df, bins=20, color="skyblue", edgecolor="black")
        plt.title("book median rate")
        plt.xlabel("book isbn")
        plt.ylabel("frequency")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()

    @staticmethod
    # Visualizes the age distribution of users using a histogram.
    def user_mean_age(data_dict: Dict[str, pd.DataFrame]):
        df = data_dict["Users"]
        plot_df = df["age"].dropna()
        plt.hist(plot_df, bins=20, color="skyblue", edgecolor="black")
        plt.title("user age distribution")
        plt.xlabel("age")
        plt.ylabel("num of users")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()

    @staticmethod
    # Visualizes the top 20 locations with the most users.
    def highest_raters_location(data_dict: Dict[str, pd.DataFrame]):
        df = data_dict["Users"]
        location_df = df["location"].value_counts().head(20)
        location_df.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("top raters location")
        plt.xlabel("location")
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("amount of raters")
        plt.tight_layout() 
        plt.grid(axis='y', linestyle="--", alpha=0.7)
        plt.show()

    @staticmethod
    # Logs and returns the minimum and maximum publication years.
    def year_range(data_dict: Dict[str, pd.DataFrame]):
        df = data_dict["Books"]
        min_year = df["year_of_publication"].min()
        max_year = df["year_of_publication"].max()
        logging.info(f"the min year is: {min_year}\nthe max year is: {max_year}")
        return min_year, max_year

    @staticmethod
    # Logs and returns the count of unique authors and publishers.
    def unique_authors_and_publishers(data_dict: Dict[str, pd.DataFrame]):
        df = data_dict["Books"]
        publishers_amount = df["publisher"].nunique()
        writers_amount = df["book_author"].nunique()
        logging.info(f"unique publishers amount {publishers_amount}, unique writers amount {writers_amount}")
        return publishers_amount, writers_amount

    @staticmethod
    # Visualizes the top 20 most frequently rated books.
    def highest_raters_books(data_dict: Dict[str, pd.DataFrame]):
        df = data_dict["Ratings"]
        rates_df = df["isbn"].value_counts().head(20)
        rates_df.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("most popular books")
        plt.xlabel("books")
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("amount of readers")
        plt.tight_layout() 
        plt.grid(axis='y', linestyle="--", alpha=0.7)
        plt.show()

    @staticmethod
    # Visualizes the top 20 books with the highest mean ratings.
    def top_20_rated_books(data_dict: Dict[str, pd.DataFrame]):
        df = data_dict["Ratings"]
        plot_df = df.groupby("isbn")["book_rating"].mean()
        top_books = plot_df.sort_values(ascending=False).head(20)
        top_books.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("book mean rate")
        plt.xlabel("book")
        plt.ylabel("mean rate")
        plt.grid(axis='y', linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    # Visualizes the top 20 users with the most ratings.
    def most_active_users(data_dict: Dict[str, pd.DataFrame]):
        df = data_dict["Ratings"]
        rates_df = df["user_id"].value_counts().head(20)
        rates_df.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("most active users")
        plt.xlabel("user id")
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("amount of ratings")
        plt.tight_layout() 
        plt.grid(axis='y', linestyle="--", alpha=0.7)
        plt.show()


class DataClean:
    @staticmethod
    # Removes irrelevant columns from the Books DataFrame.
    def drop_unrelevat_cols(data_dict: Dict[str, pd.DataFrame]) -> dict:
        cols_to_keep = ["isbn", "book_title", "book_author", "year_of_publication", "publisher"]
        if "Books" in data_dict:
            data_dict["Books"] = data_dict["Books"][cols_to_keep]
        return data_dict
    
    @staticmethod
    # Standardizes column headers to lowercase with underscores.
    def cols_heads_standart(data_dict: Dict[str, pd.DataFrame]) -> dict:
        for file, df in data_dict.items():
            cols = df.columns.str.lower()
            cols = cols.str.replace(' ', '_', regex=False)
            cols = cols.str.replace('-', '_', regex=False)
            cols = cols.str.replace(r'[^a-z0-9_]', '', regex=True)
            df.columns = cols
        logging.info("columns headers standardized")
        return data_dict
    
    @staticmethod
    # Fills missing values and validates years in the Books DataFrame.
    def clean_books_df(data_dict: Dict[str, pd.DataFrame]) -> dict:
        fill_vals = {
            "isbn": "unKnown", "book_title": "unKnown", "book_author": "unKnown",
            "year_of_publication": 0, "publisher": "unKnown"
        }
        data_dict["Books"] = data_dict["Books"].fillna(value=fill_vals)
        df = data_dict["Books"]
        four_digit_year_pattern = r"^\d{4}$"
        year_df = df["year_of_publication"].astype(str)
        is_valid_year_mask = year_df.str.match(four_digit_year_pattern)
        df.loc[~is_valid_year_mask, 'year_of_publication'] = 0
        df['year_of_publication'] = pd.to_numeric(df['year_of_publication'])
        data_dict['Books'] = df
        return data_dict
    
    @staticmethod
    # Fills missing values and filters realistic ages in the Users DataFrame.
    def clean_users_df(data_dict: Dict[str, pd.DataFrame]) -> dict:
        fill_vals = {"user_id": "unKnown", "location": "unKnown", "age": 0}
        data_dict["Users"] = data_dict["Users"].fillna(value=fill_vals)
        df = data_dict["Users"]
        fileted_df = df[(df["age"] < 100) & (df["age"] > 5)]
        data_dict["Users"] = fileted_df
        return data_dict
    
    @staticmethod
    # Fills missing values and filters out unrated books in the Ratings DataFrame.
    def clean_ratings_df(data_dict: Dict[str, pd.DataFrame]) -> dict:
        fill_vals = {"user_id": 0, "isbn": "unKnown", "book_rating": 0}
        data_dict["Ratings"] = data_dict["Ratings"].fillna(value=fill_vals)
        df = data_dict["Ratings"]
        fileted_df = df[df["book_rating"] > 0]
        data_dict["Ratings"] = fileted_df
        return data_dict
    
    @staticmethod
    # Enforces specific data types for columns in each DataFrame.
    def data_ensure_type(data_dict: Dict[str, pd.DataFrame]):
        col_type = {
            "Users": {"user_id": "Int64", "location": "object", "age": "Int64"},
            "Books": {"isbn": "object", "book_title": "object", "book_author": "object", "year_of_publication": "Int64", "publisher": "object"},
            "Ratings": {"user_id": "Int64", "isbn": "object", "book_rating": "Int64"}
        }
        for file, df in data_dict.items():
            if file in col_type:
                try:
                    data_dict[file] = data_dict[file].astype(col_type[file])
                except Exception as e:
                    logging.error(f"Could not enforce types for '{file}'. Error: {e}")
        return data_dict
    
    @staticmethod
    # Removes duplicate rows from every DataFrame in the dictionary.
    def delete_dups(data_dict: Dict[str, pd.DataFrame]) -> dict:
        for file, df in data_dict.items():
            data_dict[file] = df.drop_duplicates()
        return data_dict
    
    @staticmethod
    # Cleans and standardizes text content in object-type columns.
    def uniform_object_cols(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        for file, df in data_dict.items():
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = (df[col].str.lower()
                                .str.replace('-', '_', regex=False)
                                .str.replace(' ', '_', regex=False)
                                .str.replace(',', '_', regex=False)
                                .str.replace(r'[^a-z0-9_]', '', regex=True)
                                .str.strip('_'))
        return data_dict
    
    @staticmethod
    # Executes the complete data cleaning sequence.
    def cleaning_data_pipeline(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        data_dict = DataClean.cols_heads_standart(data_dict)
        data_dict = DataClean.drop_unrelevat_cols(data_dict)
        data_dict = DataClean.clean_users_df(data_dict)
        data_dict = DataClean.clean_books_df(data_dict)
        data_dict = DataClean.clean_ratings_df(data_dict)
        data_dict = DataClean.delete_dups(data_dict)
        data_dict = DataClean.uniform_object_cols(data_dict)
        data_dict = DataClean.data_ensure_type(data_dict)
        logging.info("Data cleaned successfully")
        return data_dict


class DataPreProcess:
    @staticmethod
    # Merges Ratings, Books, and Users DataFrames for context-based analysis.
    def context_based_df(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        ratings = data_dict["Ratings"]
        books = data_dict["Books"]
        users = data_dict["Users"]
        
        df = ratings.merge(books, on="isbn", how="left")
        df = df.merge(users, on="user_id", how="left")
        return df

    @staticmethod
    # Fully prepares context-based data including encoding and scaling with leakage prevention.
    def context_based_data_preprocessing_pipeline(data_dict: Dict[str, pd.DataFrame]):
        df = DataPreProcess.context_based_df(data_dict)
        df = df.dropna(subset=['book_rating'])
        
        target = df["book_rating"]
        features = df.drop(columns=["book_rating"])
        
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Merge target for mapping calculations to strictly prevent leakage
        x_train_copy = x_train.copy()
        x_train_copy["book_rating"] = y_train
        x_test_copy = x_test.copy()
        
        # Features Engineering computation inside training only
        x_train_eng, x_test_eng = FeaturesEngineer.context_base_models_features_engineer(x_train_copy, x_test_copy)
        
        if "book_rating" in x_train_eng.columns:
            x_train_eng = x_train_eng.drop(columns=["book_rating"])
        
        # Encoders
        cat_cols = x_train_eng.select_dtypes(include=['object', 'category']).columns.tolist()
        encoder = HashingEncoder(cols=cat_cols, n_components=128)
        x_train_encoded = encoder.fit_transform(x_train_eng)
        x_test_encoded = encoder.transform(x_test_eng)
        joblib.dump(encoder, "hashing_encoder.joblib")
        
        # Scalers
        scaler = StandardScaler()
        x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train_encoded), columns=x_train_encoded.columns, index=x_train_encoded.index)
        x_test_scaled = pd.DataFrame(scaler.transform(x_test_encoded), columns=x_test_encoded.columns, index=x_test_encoded.index)
        joblib.dump(scaler, "standard_scaler.joblib")
        
        return x_train_scaled, x_test_scaled, y_train, y_test

    @staticmethod
    # Splits the ratings data into training and testing sets for SVD models.
    def user_based_data_svd(data_dict: Dict[str, pd.DataFrame], test_size=0.2, random_state=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        rating_df = data_dict["Ratings"]
        train_df, test_df = train_test_split(rating_df, test_size=test_size, random_state=random_state)
        return train_df, test_df

    @staticmethod
    # Creates sparse user-item matrices and splits them for ALS and KNN models.
    def user_based_data_als_knn(data_dict, test_size=0.2, random_state=42):
        rating_df = data_dict["Ratings"]
        users = rating_df["user_id"].astype('category')
        books = rating_df["isbn"].astype('category')
        sparse_matrix = csr_matrix((rating_df["book_rating"].values, (users.cat.codes, books.cat.codes)), shape=(len(users.cat.categories), len(books.cat.categories)))
        user_item_matrix = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, index=users.cat.categories, columns=books.cat.categories)
        rows, cols = sparse_matrix.nonzero()
        data = sparse_matrix.data
        
        train_data, test_data, train_rows, test_rows, train_cols, test_cols = train_test_split(
            data, rows, cols, test_size=test_size, random_state=random_state, stratify=rows
        )
        train_matrix = csr_matrix((train_data, (train_rows, train_cols)), shape=sparse_matrix.shape)
        test_matrix = csr_matrix((test_data, (test_rows, test_cols)), shape=sparse_matrix.shape)
        return train_matrix, test_matrix, user_item_matrix


class FeaturesEngineer:
    @staticmethod
    # Generates and maps statistical features from training data to prevent leakage.
    def context_base_models_features_engineer(x_train: pd.DataFrame, x_test: pd.DataFrame):
        # Maps logic preventing leakage to be computed only on train
        feature_maps = {
            "book_rating_mean": x_train.groupby("isbn")["book_rating"].mean().to_dict(),
            "rating_count": x_train.groupby("isbn")["book_rating"].count().to_dict(),
            "book_rating_var": x_train.groupby("isbn")["book_rating"].var().to_dict(),
            "writer_mean_rate": x_train.groupby("book_author")["book_rating"].mean().to_dict(),
            "writer_book_count": x_train.groupby("book_author")["isbn"].nunique().to_dict(),
            "user_mean_rate": x_train.groupby("user_id")["book_rating"].mean().to_dict(),
            "user_rating_count": x_train.groupby("user_id")["book_rating"].count().to_dict()
        }
        joblib.dump(feature_maps, "feature_maps.joblib")

        maps = feature_maps
        x_train["book_mean_rate"] = x_train["isbn"].map(maps["book_rating_mean"])
        x_train["rating_count"] = x_train["isbn"].map(maps["rating_count"])
        x_train["book_age"] = 2026 - x_train["year_of_publication"] # updated dynamic year mapping
        x_train["book_rating_var"] = x_train["isbn"].map(maps["book_rating_var"])
        x_train["writer_mean_rate"] = x_train["book_author"].map(maps["writer_mean_rate"])
        x_train["writer_book_count"] = x_train["book_author"].map(maps["writer_book_count"])
        x_train["user_mean_rate"] = x_train["user_id"].map(maps["user_mean_rate"])
        x_train["user_rating_count"] = x_train["user_id"].map(maps["user_rating_count"])
        x_train = x_train.fillna(0)

        x_test["book_mean_rate"] = x_test["isbn"].map(maps["book_rating_mean"])
        x_test["rating_count"] = x_test["isbn"].map(maps["rating_count"])
        x_test["book_age"] = 2026 - x_test["year_of_publication"]
        x_test["book_rating_var"] = x_test["isbn"].map(maps["book_rating_var"])
        x_test["writer_mean_rate"] = x_test["book_author"].map(maps["writer_mean_rate"])
        x_test["writer_book_count"] = x_test["book_author"].map(maps["writer_book_count"])
        x_test["user_mean_rate"] = x_test["user_id"].map(maps["user_mean_rate"])
        x_test["user_rating_count"] = x_test["user_id"].map(maps["user_rating_count"])
        x_test = x_test.fillna(0)

        cols_to_drop = ["user_id", "isbn", "book_title"]
        return x_train.drop(cols_to_drop, axis=1), x_test.drop(cols_to_drop, axis=1)

    @staticmethod
    # Maps pre-calculated features to a DataFrame for hybrid model inference.
    def hybrid_context_based_features_engineer(df: pd.DataFrame) -> pd.DataFrame:
        """ Uses loaded artifacts so no calculation leak happens on inference. """
        feature_maps = joblib.load("feature_maps.joblib")
        
        df["book_mean_rate"] = df["isbn"].map(feature_maps["book_rating_mean"])
        df["rating_count"] = df["isbn"].map(feature_maps["rating_count"])
        df["book_age"] = 2026 - df["year_of_publication"]
        df["book_rating_var"] = df["isbn"].map(feature_maps["book_rating_var"])
        df["writer_mean_rate"] = df["book_author"].map(feature_maps["writer_mean_rate"])
        df["writer_book_count"] = df["book_author"].map(feature_maps["writer_book_count"])
        df["user_mean_rate"] = df["user_id"].map(feature_maps["user_mean_rate"])
        df["user_rating_count"] = df["user_id"].map(feature_maps["user_rating_count"])
        df = df.fillna(0)

        cols_to_drop = ["user_id", "isbn", "book_title"]
        if cols_to_drop[0] in df.columns:
            df = df.drop(cols_to_drop, axis=1, errors='ignore')
        return df

    @staticmethod
    # Converts a dense DataFrame into a sparse user-item matrix for hybrid models.
    def hybrid_knn_als_data(df: pd.DataFrame, test_size=0.2, random_state=42) -> Tuple[pd.DataFrame, csr_matrix]:
        users = df["user_id"].astype('category')
        books = df["isbn"].astype('category')
        sparse_matrix = csr_matrix((df["book_rating"].values, (users.cat.codes, books.cat.codes)), shape=(len(users.cat.categories), len(books.cat.categories)))
        user_item_matrix = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, index=users.cat.categories, columns=books.cat.categories)
        return user_item_matrix, sparse_matrix
