from img2vec_pytorch import Img2Vec
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Initialize Img2Vec
img2vec = Img2Vec()

# Define directories
data_dir = 'data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

data = {}

# Iterate over train and validation directories
for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):  # Loop through categories
        category_dir = os.path.join(dir_, category)
        for img_path in os.listdir(category_dir):  # Loop through image files in category
            img_path_ = os.path.join(category_dir, img_path)
            
            try:
                # Open image and ensure it's in RGB format
                img = Image.open(img_path_).convert("RGB")
                
                # Compute image features
                img_features = img2vec.get_vec(img)
                
                # Append features and labels
                features.append(img_features)
                labels.append(category)
            except Exception as e:
                print(f"Error processing {img_path_}: {e}")
    
    # Store features and labels for train/validation
    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels


# Train the model
model = RandomForestClassifier()
model.fit(data['training_data'], data['training_labels'])

# Test Performance
y_pred = model.predict(data["validation_data"])
score = accuracy_score(y_pred, data['validation_labels'])

print(score)

# Save the Model
# 'wb' - write binary, './'- opens file in current directory
with open('./model.p', 'wb') as f:
    pickle.dump(model, f)
    f.close()