# run: python run_keras_server.py 
# 	   curl -X POST -F image=@image_name.jpg 'http://localhost:5000/predict'

# import the necessary packages
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
from PIL import Image
import cv2
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
graph = None
sess = None

CATEGORIES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
NAME = 'flowers-cnn-64x2-model'
MODEL_NAME_JSON = NAME + '.json'
WEIGHTS_NAME_H5 = NAME + '.h5'
IMAGE_SIZE = 128

def decode_predictions(pred):
	# pred is an array of percentages [0. 1. 0. 0. 0.]
	result = []
	for i in range(len(pred)):
		# result = [['daisy', 0], ['dandelion', 1], ['rose', 0]....]
		label = CATEGORIES[i]
		percentage = str(pred[i])
		result.append([label, percentage])

	print(result)
	return result


def load_model():
    # load the pre-trained Keras model

	global model
	global graph
	global sess

	tf_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2)
	sess = tf.Session(config=tf_config)
	graph = tf.get_default_graph()

	set_session(sess)

	print("Loading model from disk........")

	# load json and create model
	json_file = open(MODEL_NAME_JSON, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into new model
	loaded_model.load_weights(WEIGHTS_NAME_H5)
	print("Loaded model from disk")

	loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model = loaded_model

@app.route("/predict", methods=["POST"])
def predict():
    print('-------------  /predict -------------')
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):

        	# prepare image
            imagestr = flask.request.files['image'].read()
            npimg = np.fromstring(imagestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))
            image = np.array(image).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            global sess
            global graph
            with graph.as_default():
                set_session(sess)
                preds = model.predict(image)

            result = decode_predictions(preds[0])
            data["predictions"] = result

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()

