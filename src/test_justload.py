import keras

model = keras.models.load_model('models/facial_emotion_classifier.h5')

print(model.summary())