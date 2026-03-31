from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import logging
from typing import List, Dict
from MLModels import HybridRecommender

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Book Recommender API")

context_model = None
user_model = None
user_item_matrix = None
clean_data = None
ratings_df = None
encoder = None
scaler = None

@app.on_event("startup")
async def startup_event():
    global context_model, user_model, user_item_matrix, clean_data, ratings_df, encoder, scaler
    try:
        context_model = joblib.load("best_context_model.joblib")
        user_model = joblib.load("best_user_based_model.joblib")
        user_item_matrix = joblib.load("user_item_matrix.joblib")
        clean_data = joblib.load("cleaned_data_dict.joblib")
        ratings_df = clean_data.get("Ratings", pd.DataFrame())
        encoder = joblib.load("hashing_encoder.joblib")
        scaler = joblib.load("standard_scaler.joblib")
        logging.info("Successfully loaded all models and data artifacts")
    except Exception as e:
        logging.error(f"Failed to load engine components: {e}")

@app.get("/")
async def root():
    return {"status": "online", "message": "Book Recommendation Engine API"}

@app.get("/recommend/{user_id}", response_model=List[Dict])
async def get_recommendations(user_id: int):
    try:
        if user_model is None or context_model is None:
            raise HTTPException(status_code=503, detail="Models are not fully loaded.")
            
        # Cold Start Fallback
        if user_id not in user_item_matrix.index:
            logging.info(f"User {user_id} not in matrix. Fallback to popularity logic.")
            books_df = clean_data["Books"]
            top_10 = ratings_df['isbn'].value_counts().head(10).index
            popular_books = books_df[books_df['isbn'].isin(top_10)].to_dict(orient="records")
            return popular_books
            
        recommender = HybridRecommender(
            context_based_model_and_prediction=(context_model, None),
            user_based_model=user_model,
            user_item_matrix=user_item_matrix,
            train_df=None,
            test_df=None,
            data_df=ratings_df,
            user_id=user_id
        )

        sample_300 = recommender.book_sample_recommmend()
        df_300 = recommender.get_data_frame(sample_300)
        alpha = recommender.weight_per_model(df_300)
        top_10 = recommender.recommend(alpha, df_300)

        return top_10.to_dict(orient="records")
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error generating recommendation for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during recommendation generation")

@app.get("/book/{isbn}")
async def get_book_info(isbn: str):
    books_df = clean_data["Books"]
    book = books_df[books_df["isbn"] == isbn]
    if book.empty:
        raise HTTPException(status_code=404, detail="Book ISBN not found")
    return book.iloc[0].to_dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
