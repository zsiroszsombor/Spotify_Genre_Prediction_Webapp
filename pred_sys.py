import numpy as np
import pickle

#loading the model
loaded_model = pickle.load(open("C:/Users/zsiro/Desktop/Python scripts/spotify_model/trained_model.sav",'rb'))


#testing input for loaded model
input_data=(140, 57, 34, -7, 42, 37, 257.0, 6, 3, 30)
input_as_numpy = np.asarray(input_data)
input_as_numpy_reshaped = input_as_numpy.reshape(1,-1)

pred = loaded_model.predict(input_as_numpy_reshaped)
print(pred)