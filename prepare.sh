sudo apt-get install lbzip2

mkdir pretrained
cd pretrained

wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

lbzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2
lbzip2 -d shape_predictor_68_face_landmarks.dat.bz2