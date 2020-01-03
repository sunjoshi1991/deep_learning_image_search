
### extract image fetures and saved as pickle files (image weights)to compare with new image


import glob
import os, pickle

from PIL import Image
from feature_extractor import feature_extract

fe = feature_extract()

for image_path in sorted(glob.glob('static/images/*.jpg')):
	#print(image_path)
	img = Image.open(image_path)
	feature = fe.extract(img)
	feature_path = 'static/features/'+os.path.splitext(os.path.basename(image_path))[0] + '.pkl'
	pickle.dump(feature, open(feature_path, 'wb'))
