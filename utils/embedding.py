import numpy as np


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
