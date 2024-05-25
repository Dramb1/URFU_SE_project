from io import BytesIO
import base64

from PIL import Image
import dlib
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_face_embedding


app = FastAPI()

detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor(
    "./pretrained/shape_predictor_68_face_landmarks.dat"
)
facerec = dlib.face_recognition_model_v1(
    "./pretrained/dlib_face_recognition_resnet_model_v1.dat"
)


@app.get("/")
async def home_page():
    """Домашняя страница для загрузки сравнения изображений

    Returns:
        HTMLResponse: html page layout
    """

    html_content = """
          <form method="post" enctype="multipart/form-data">
          <div>
              <label>Upload Images</label>
              <input name="file" type="file" multiple>
              <input name="file1" type="file" multiple>
          </div>
          <button type="submit">Submit</button>
          </form>
    """

    return HTMLResponse(content=html_content, status_code=200)


@app.post("/")
async def processing_request(
    file: UploadFile = File(...), file1: UploadFile = File(...)
):
    """Post a query comparing two images of people

    Args:
        file (UploadFile): Images of a human face
        file1 (UploadFile): Images of a human face

    Returns:
        HTMLResponse: html page layout with two images and cosine similarity of people
    """

    image1 = np.asarray(Image.open(BytesIO(await file.read())))
    image2 = np.asarray(Image.open(BytesIO(await file1.read())))

    embedding1 = get_face_embedding(image1, detector, landmarks_predictor, facerec)
    embedding2 = get_face_embedding(image2, detector, landmarks_predictor, facerec)

    cosine_distance = cosine_similarity([embedding1], [embedding2])

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    _, image1_buffer = cv2.imencode(".png", image1)
    _, image2_buffer = cv2.imencode(".png", image2)
    image1_base64 = base64.b64encode(image1_buffer).decode("utf-8")
    image2_base64 = base64.b64encode(image2_buffer).decode("utf-8")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>HTML Content</title>
    </head>
    <body>
        <h1>Изображения для сравнения</h1>
        <img src="data:image/jpeg;base64,{image1_base64}" alt="Image 1" width="300">
        <img src="data:image/jpeg;base64,{image2_base64}" alt="Image 2" width="300">
        <p>{'На изображениях один человек' if cosine_distance[0][0] > 0.95 else 'На изображениях разные люди'}</p>
        <p>cosine_distance: {cosine_distance[0][0]}</p>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    app_str = "server_reid:app"
    uvicorn.run(app_str, host="localhost", port=8000, reload=True, workers=1)
