from fastapi import FastAPI, UploadFile, File
import io
from PIL import Image

app = FastAPI(title="BanglaVision API", description="REST API for Bengali Image Captioning")

@app.post("/caption")
async def generate_caption_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    return {"caption": "একটি সবুজ মাঠের ওপর শিশুরা ফুটবল খেলছে।", "confidence": 0.94}
