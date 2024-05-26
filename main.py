from io import BytesIO
import base64
import os

from PIL import Image
import dlib
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_face_embedding, get_embs_db


app = FastAPI()

WORKSPACE_PATH = os.path.dirname(os.getcwd())
persons_emb, persons_id = get_embs_db()
detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor(
    "./pretrained/shape_predictor_68_face_landmarks.dat"
)
facerec = dlib.face_recognition_model_v1(
    "./pretrained/dlib_face_recognition_resnet_model_v1.dat"
)


@app.get("/")
async def home_page():
    """Home page

    Returns:
        HTMLResponse: html page layout
    """

    html_content = """
        <form method="get" action="/reidentification" enctype="multipart/form-data">
            <button type="submit">Reid person</button>
        </form> 
        <form method="get" action="/add_person" enctype="multipart/form-data">
            <button type="submit">Add new person</button>
        </form>
    """

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/reidentification")
async def reid_page():
    """Page for uploading an image for reidentification

    Returns:
        HTMLResponse: html page layout
    """

    html_content = """
        <form method="post" action="/reidentification" enctype="multipart/form-data">
            <div>
                <label>Upload Image</label>
                <input name="file" type="file" multiple>
            </div>
            <button type="submit">Submit</button>
        </form>

        <form method="get" action="/" enctype="multipart/form-data">
            <button type="submit">Back to home page</button>
        </form>
    """

    return HTMLResponse(content=html_content, status_code=200)


@app.post("/reidentification")
async def processing_reid_request(file: UploadFile = File(...)):
    """Post a query for person reidentification

    Args:
        file (UploadFile): Image of a human face

    Returns:
        HTMLResponse: html page layout with image and id person in database
    """

    image = np.asarray(Image.open(BytesIO(await file.read())))

    embedding = get_face_embedding(image, detector, landmarks_predictor, facerec)

    cosine_distance = cosine_similarity([embedding], persons_emb)
    max_same_person_id = np.argmax(cosine_distance)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, image_buffer = cv2.imencode(".png", image)
    image_base64 = base64.b64encode(image_buffer).decode("utf-8")

    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>HTML Content</title>
        </head>
        <body>
            <h1>Изображение для повторной идентификации</h1>
            <img src="data:image/jpeg;base64,{image_base64}" alt="Image 1" width="300">
            <p>{f'person ID: {persons_id[max_same_person_id]}' 
                if cosine_distance[0, max_same_person_id] > 0.945 else 'person ID: Unknown'}</p>
        </body>

        <form method="get" action="/" enctype="multipart/form-data">
            <button type="submit">Back to home page</button>
        </form>
    </html>
    """

    return HTMLResponse(content=html_content)


@app.get("/add_person")
async def page_add_person():
    """Page for adding a person from the database

    Returns:
        HTMLResponse: html page layout
    """

    html_content = """
        <form method="post" action="/add_person" enctype="multipart/form-data">
            <div>
                <label>Upload Image</label>
                <input name="file" type="file">
            </div>
            <div>
                <label>Id person</label>
                <input name="id_person" type="text" placeholder="Enter save path">
            </div>
            <button type="submit">Save person</button>
        </form>

        <form method="get" action="/" enctype="multipart/form-data">
            <button type="submit">Back to home page</button>
        </form>
    """

    return HTMLResponse(content=html_content, status_code=200)


@app.post("/add_person")
async def page_add_person(file: UploadFile = File(...), id_person: str = Form(...)):
    global persons_emb, persons_id

    """Post a query for adding a person from the database

    Args:
        file (UploadFile): Uploaded file to save
        id_person (str): Path to save the image

    Raises:
        HTTPException: If there is an error saving the person

    Returns:
        HTMLResponse: Success form if Pesron is saved successfully
    """
    try:
        image = np.asarray(Image.open(BytesIO(await file.read())))

        embedding = get_face_embedding(image, detector, landmarks_predictor, facerec)

        embs_db_path = os.path.join(WORKSPACE_PATH, "embeddings_database")
        embs_person_folder_path = os.path.join(embs_db_path, id_person)
        if not os.path.exists(embs_person_folder_path):
            os.mkdir(embs_person_folder_path)
        else:
            with open(os.path.join(embs_person_folder_path, "mean_emb.npy"), "rb") as f:
                emb = np.load(f)
                embedding = np.vstack((embedding.reshape(1, -1), emb))
                embedding = np.mean(embedding, axis=0)
        with open(os.path.join(embs_person_folder_path, "mean_emb.npy"), "wb") as f:
            np.save(f, embedding)

        persons_emb, persons_id = get_embs_db()

        html_content = f"""
            <form method="get" action="/" enctype="multipart/form-data">
                <div>
                    <p>Person saved successfully at path: {embs_person_folder_path}</p>
                </div>
                <button type="submit">Back to home page</button>
            </form>
        """
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error saving person: " + str(e))


if __name__ == "__main__":
    import uvicorn

    app_str = "main:app"
    uvicorn.run(app_str, host="localhost", port=8000, reload=True, workers=1)
