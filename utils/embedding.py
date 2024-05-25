import os
import numpy as np


WORKSPACE_PATH = os.path.dirname(os.getcwd())
EMB_SIZE = 128


def get_face_embedding(image, detector, landmarks_predictor, facerec):
    """Obtaining the embedding of a person's face in an image

    Args:
        image numpy.ndarray: _description_
        detector (_dlib_pybind11.fhog_object_detector): face detector
        landmarks_predictor (_dlib_pybind11.shape_predictor): face landmarks predictor
        facerec (_dlib_pybind11.face_recognition_model_v1): face recognition model

    Returns:
        numpy.ndarray: face embedding
    """
    dets = detector(image, 1)
    detection = dets[0]
    shape = landmarks_predictor(image, detection)
    face_descriptor = facerec.compute_face_descriptor(image, shape)
    return np.array(face_descriptor)


def get_embs_db():
    """gaining an embedding base

    Returns:
        (numpy.ndarray, list): array of person embeddings and list of identifiers
    """
    embs_db_path = os.path.join(WORKSPACE_PATH, "embeddings_database")
    persons_emb = np.zeros((0, EMB_SIZE))
    persons_id = []

    person_folders = os.listdir(embs_db_path)
    for person_folder in person_folders:
        embs_person_folder_path = os.path.join(embs_db_path, person_folder)

        with open(os.path.join(embs_person_folder_path, "mean_emb.npy"), "rb") as f:
            emb = np.load(f)
            persons_emb = np.vstack((persons_emb, emb))
            persons_id.append(person_folder)
    
    return persons_emb, persons_id
