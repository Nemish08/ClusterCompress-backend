from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from app.models import MultiCompressionResult
from app.services import image_processing_service

app = FastAPI(
    title="ClusterCompress API",
    description="An API to compress images using K-Means and analyze the color palette.",
    version="1.0.0"
)

origins = [
    "https://cluster-compress-frontend.vercel.app/"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/compress-all", response_model=MultiCompressionResult)
async def compress_all_algorithms(
    image: UploadFile = File(..., description="The image file to compress."),
    k: int = Form(16, ge=1, le=64, description="Number of colors for K-Means & Hierarchical.")
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    image_bytes = await image.read()
    original_size = len(image_bytes)
    
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        kmeans_result = image_processing_service.compress_with_kmeans(pil_image, k)
        hierarchical_result = image_processing_service.compress_with_hierarchical(pil_image, k)
        dbscan_result = image_processing_service.compress_with_dbscan(pil_image, eps=k/300) # eps might need tuning

        return {
            "original_size": original_size,
            "kmeans": kmeans_result,
            "hierarchical": hierarchical_result,
            "dbscan": dbscan_result
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

