import sys
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.image as mpimg
import cv2
import numpy as np
from skimage import io
from PIL import Image
import json


def new_shit(image_path):
	
	# OpenCV
    print("OpenCV")
    gfile = tf.gfile.FastGFile(image_path, 'rb').read()
    nparr = np.fromstring(gfile, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR).astype(np.float32)/255.0
    resi = cv2.resize(img_np, (224, 224))

    
    return {"image": resi.tolist() }

#tensor_image_32 = convert_image_in_32b(sys.argv[1])
tensor_image_32 = new_shit(sys.argv[1])

file = open("image_32.json", "w")
file.write(
    json.dumps(tensor_image_32)
)
file.close()

#tensor_image_32 = new_convertion_type(sys.argv[1])
#tensor_to_string = tf.as_string(tensor_image_32)
#print("type:", type(tensor_to_string))

#tf.write_file("image_predict.json", tensor_to_string)
