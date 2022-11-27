from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import List
import io
import numpy as np
import sys
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
 
# load model
#filepath = "./saved_model"
filepath = 'G:/plant disease detection/model/2.h5'
# model = load_model(filepath, compile=True)
model = load_model(filepath)
from IPython.display import FileLink
FileLink(r'class_indices.json')
classes= 	  ["Apple Scab", "Apple Black Rot", "Cedar Apple Rust", 
               "Apple Healthy", "Blueberry Healthy", 
               "Cherry Powdery Mildew", "Cherry Healthy", 
               "Corn Cercospora Leaf spot", "Corn Common Rust", 
               "Corn Northern Leaf Blight", "Corn Healthy", 
               "Grape Black Rot", "Grape Black Measles", 
               "Grape Leaf Blight", "Grape Healthy", 
               "Orange Haunglongbing", "Peach Bacterial spot", 
               "Peach Healthy", "Pepper Bell Bacterial spot", "Pepper Bell healthy", 
               "Potato Early blight", "Potato Late blight", "Potato Healthy", 
               "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew", 
               "Strawberry Leaf scorch", "Strawberry Healthy", "Tomato Bacterial spot", 
               "Tomato Early blight", "Tomato Late blight", "Tomato Leaf Mold", 
               "Tomato Septoria leaf spot", "Tomato Spider mites ", 
               "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", 
               "Tomato Mosaicvirus", "Tomato Healthy"]

# get the input shape for the model layer
input_shape = model.layers[0].input_shape
 


# define the fastAPI
app = FastAPI()
#allowing backend process to run on localhost:3000(CORS)
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# define response
@app.get("/")
def root_route():
	return {'error': 'Use GET /prediction instead of the root route!'}

# define the prediction route
@app.post("/prediction/")
async def prediction_route(file: UploadFile = File(...)):
	 #ensure that this is an image
	if file.content_type.startswith("image/") is False:
		raise HTTPException(status_code=400,
		                    detail=f'File \'{file.filename}\' is not an image.'
		)

	try:
		# read image contain
		contents = await  file.read()
		pil_image = Image.open(io.BytesIO(contents))

		# resize image to expected input shape
		pil_image = pil_image.resize((input_shape[1], input_shape[2]))
		 
		
		# convert image into grayscale
		if input_shape[3] and input_shape[3] == 1:
			pil_image = pil_image.convert('L')

		# convert imgae to numpy format
		numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))

		# scale data
		numpy_image = numpy_image / 255.0
		img_batch = np.expand_dims(numpy_image, 0)
		# generate prediction
		prediction_array = np.array([numpy_image])
		predictions = model.predict(prediction_array)
		y_class = np.argmax(predictions,axis=1)
		y_class[0]
		prediction = predictions[0]
		likely_class = np.argmax(prediction)
		prediction=classes[y_class[0]]
		confidence = np.max(predictions[0])
    	 
        
		return {
			
			 
			"class": prediction,
			"confidence":float(confidence)
		  
			 
             
        }
	except:
		e = sys.exc_info()[1]
		raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)