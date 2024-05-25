import os
import argparse

import cv2
import numpy as np
import dlib

from utils import get_face_embedding


WORKSPACE_PATH = os.path.dirname(os.getcwd())

detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor(
    "./pretrained/shape_predictor_68_face_landmarks.dat"
)
facerec = dlib.face_recognition_model_v1(
    "./pretrained/dlib_face_recognition_resnet_model_v1.dat"
)


def process_images_in_folder(database_path):
    embs_db_path = os.path.join(WORKSPACE_PATH, "embeddings_database")
    if not os.path.exists(embs_db_path):
        os.mkdir(embs_db_path)

    person_folders = os.listdir(database_path)
    for folder_path in person_folders:
        embs = []
        embs_person_folder_path = os.path.join(embs_db_path, folder_path)
        if not os.path.exists(embs_person_folder_path):
            os.mkdir(embs_person_folder_path)

        person_folder_path = os.path.join(database_path, folder_path)
        _, _, files  = next(os.walk(person_folder_path))
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(os.path.join(person_folder_path, file))
                embedding = get_face_embedding(image, detector, landmarks_predictor, facerec)
                embs.append(embedding)
        
        with open(os.path.join(embs_person_folder_path, f'mean_emb.npy'), 'wb') as f:
            np.save(f, np.mean(embs, axis=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating embeddings database.')
    parser.add_argument('database_path', type=str, help='Path to the folder containing images')
    args = parser.parse_args()

    process_images_in_folder(args.database_path)
