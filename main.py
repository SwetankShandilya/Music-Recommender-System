from fastapi import FastAPI, HTTPException, Query
from typing import Optional
from recommender import recommend_songs_collaborative, recommend_songs_content, recommend_songs_hybrid
from recommender import user_to_index, track_to_index 
import traceback
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Music Recommendation System API!"}


@app.get("/recommend/collaborative/")
def get_recommendations_collab(user_id: str,top_n: int = 10):
    try:
        recommendations = recommend_songs_collaborative(user_id, top_n)
        return {"recommendations": recommendations}

    except KeyError as e:
        
        raise HTTPException(status_code=404, detail={
            "error": str(e),
            "total_users": len(user_to_index),
            "sample_users": list(user_to_index.keys())[:5]
        })
    
    except Exception as e:
        print("Unexpected error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail={
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.get("/recommend/content/")
def get_recommendations_content(song_name: str, top_n: int = 10):
    return {"recommendations": recommend_songs_content(song_name, top_n)}

@app.get("/recommend/hybrid/")
def get_recommendations_hybrid(user_id: str, song_name: str, top_n: int = 10):
    return {"recommendations": recommend_songs_hybrid(user_id, song_name, top_n)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
