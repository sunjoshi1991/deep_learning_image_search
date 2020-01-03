#### import Flask libraries

import os
import numpy as np
from PIL import Image
from feature_extractor import feature_extract
import glob , pickle 
from datetime import datetime
from flask import Flask , request, render_template

app =Flask(__name__)

### read image features from feature extractor class

fe = feature_extract()

features = []
image_paths = []

for feat in glob.glob("static/features/*"):
	features.append(pickle.load(open(feat, 'rb')))
	image_paths.append('static/images/' + os.path.splitext(os.path.basename(feat))[0] + '.jpg')


## flask app to predcit the image with scores (distances)
@app.route('/',methods = ['GET', 'POST'])

def image_search():
	if request.method=='POST':
		file = request.files['query_image']
		img = Image.open(file.stream)
		upload_image_path = "static/uploads/" + datetime.now().isoformat() + "_" + file.filename
		img.save(upload_image_path)
		query = fe.extract(img)
		distance = np.linalg.norm(features-query, axis =1)
		print(distance)
		ids = np.argsort(distance)[:20]
		for id in ids:
			scores = (distance[id], image_paths[id])

		return render_template('index.html',query_path = upload_image_path, scores = scores)
	else:
		return render_template('index.html')




if __name__ == '__main__':
	app.run("0.0.0.0")

	




