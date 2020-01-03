#### import kearas libraries and VGG16


from keras.preprocessing import image
from keras.applications.vgg16 import VGG16 , preprocess_input
from keras.models import Model

## set this default to avoid any errors with tenserflow backend
import tensorflow as tf
graph = tf.get_default_graph()

import numpy as np

### extract features and build simple model using VGG16 as base model and outputs as fully connected layer

class feature_extract:
	 def __init__ (self):
	 	base_model = VGG16(weights = 'imagenet')
	 	self.model = Model(inputs = base_model.input , outputs = base_model.get_layer('fc1').output)

	 ### extract image features and predict using VGG16 model
	 def extract(self, img):
	 	img = img.resize((224,224))
	 	img = img.convert('RGB')
	 	x =image.img_to_array(img)
	 	x = np.expand_dims(x, axis = 0)
	 	x = preprocess_input(x)
	 	with graph.as_default():
	 		feature = self.model.predict(x)[0]
	 	return feature/np.linalg.norm(feature)




	 	