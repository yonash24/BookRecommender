from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import logging
from typing import List, Dict
from MLModels import HybridRecommender

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Book Recommender API")

"""
load the required models and data objects for inference
ensures they are in memory before any request
"""
try:
    context_model = joblib.load("best_context_model.joblib")
    user_model = joblib.load("best_user_based_model.joblib")
    user_item_matrix = joblib.load("user_item_matrix.joblib")
    clean_data = joblib.load("cleaned_data_dict.joblib")
    ratings_df = clean_data["Ratings"]
    logging.info("Successfully loaded all models and data artifacts")
except Exception as e:
    logging.error(f"Failed to load engine components: {e}")

"""
root endpoint to check api status
"""
@app.get("/")
async def root():
    return {"status": "online", "message": "Book Recommendation Engine API"}

"""
endpoint to retrieve top 10 hybrid recommendations for a specific user
gets user_id as path parameter and returns a list of recommended books
"""
@app.get("/recommend/{user_id}", response_model=List[Dict])
async def get_recommendations(user_id: int):
    if user_id not in user_item_matrix.index:
        raise HTTPException(status_code=404, detail="User ID not found in system")

    try:
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
    except Exception as e:
        logging.error(f"Error generating recommendation for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during recommendation generation")

"""
endpoint to get metadata about a specific book by isbn
"""
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
