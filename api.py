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
feature_maps = None
encoder = None
scaler = None

@app.on_event("startup")
async def startup_event():
    global context_model, user_model, user_item_matrix, clean_data, ratings_df, feature_maps, encoder, scaler
    try:
        context_model = joblib.load("best_context_model.joblib")
        user_model = joblib.load("best_user_based_model.joblib")
        user_item_matrix = joblib.load("user_item_matrix.joblib")
        clean_data = joblib.load("cleaned_data_dict.joblib")
        ratings_df = clean_data.get("Ratings", pd.DataFrame())
        feature_maps = joblib.load("feature_maps.joblib")
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
        if user_model is None or user_item_matrix is None:
            raise HTTPException(status_code=503, detail="Models are not fully loaded.")
            
        if user_id not in user_item_matrix.index:
            # Cold start logic handled inside HybridRecommender or here
            logging.info(f"User {user_id} not in matrix. Fallback to popularity logic.")
            pass # the HybridRecommender could handle this gracefully with fallback
            
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
