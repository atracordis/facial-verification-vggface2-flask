import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from PIL import Image
from scipy.spatial.distance import cosine
from keras import backend as K

def preprocess_input(x):
    x_temp = np.copy(x)
    data_format = K.image_data_format()

    if data_format == 'channels_first':
        x_temp = x_temp[:, ::-1, ...]
        x_temp[:, 0, :, :] -= 91.4953
        x_temp[:, 1, :, :] -= 103.8827
        x_temp[:, 2, :, :] -= 131.0912
    else:
        x_temp = x_temp[..., ::-1]
        x_temp[..., 0] -= 91.4953
        x_temp[..., 1] -= 103.8827
        x_temp[..., 2] -= 131.0912

    return x_temp


# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames, model):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = np.asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples)
    # perform prediction
    yhat = model.predict(samples)
    return yhat


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        return '>face is a Match (%.3f <= %.3f)' % (score, thresh)
    else:
        return '>face is NOT a Match (%.3f > %.3f)' % (score, thresh)

try :
    model = load_model("./model/model_vggface2.keras")
except:
    from keras_vggface.vggface import VGGFace
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    model.save("./model/model_vggface2.keras")

