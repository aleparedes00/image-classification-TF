from googleapiclient import discovery
from googleapiclient import errors
import sys
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import json

	
def create_dict(image_path):
    gfile = tf.gfile.FastGFile(image_path, 'rb').read()
    nparr = np.fromstring(gfile, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR).astype(np.float32)/255.0
    resi = cv2.resize(img_np, (224, 224))
    
    return {"instances": [resi.tolist()] }



#set a connection to the services
ml = discovery.build('ml','v1')

# Store your full project ID in a variable in the format the API needs.
projectID = 'sandbox-arch'
model_name = 'image_classification_SystemeU'
version_name = 'image_classification_mobilenet'

#Dict with the tensor of the image
request_dict = create_dict(sys.argv[1])
  
#Create a request
request = ml.projects().predict(name='projects/{projectID}/models/{model_name}'.format(projectID=projectID, model_name=model_name), body=request_dict)
  
response = request.execute()
print(response)

