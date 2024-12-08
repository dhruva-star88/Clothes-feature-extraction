from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle

with open('./model.p', 'rb') as f:
    model = pickle.load(f)

# Create Instance
img2vec = Img2Vec()

# Load Image Path
image_path = "watch.jpg"

# Open the image using PIL library
img = Image.open(image_path)

# Get Features
features = img2vec.get_vec(img)

# Predict
pred = model.predict([features])

print(pred)