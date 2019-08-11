# Flower_Classification

Download the dataset and unzip from https://www.kaggle.com/alxmamaev/flowers-recognition

I have trained a CNN written in Keras to predict the type of flower from the dataset

You can also run the FLASK server by runnig "python run_keras_server.py"
Then "curl -X POST -F image=@**some_flower_image_name.jpg** 'http://localhost:5000/predict'" to get a prediction
