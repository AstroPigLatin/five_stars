from keras.models import model_from_json
import numpy as np


class HandGestureModel(object):
    hand_gestures = ["0", "1","2", "3", "4",
                     "5", "6",
                     "7", "8",
                     "9"]
    
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open('model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
            
        # load weights into the new model
        self.loaded_model.load_weights('model_weights.h5')
#         self.loaded_model._make_predict_function()
        
        
    def predict_number(self, img):
        self.preds = self.loaded_model.predict(img)
        return HandGestureModel.hand_gestures[np.argmax(self.preds)]
    
    
#  ValueError: Input 0 of layer conv1 is incompatible with the layer: expected axis -1 of input shape to have value 3 but received input with shape [None, 49, 49, 1]    
# Before: (48, 48 size) for variable "roi" -- data pixel of grayscale image of face
#  ValueError: Input 0 of layer conv1 is incompatible with the layer: expected axis -1 of input shape to have value 3 but received input with shape [None, 50, 50, 1]
