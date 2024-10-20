import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import pandas as pd
import os
import torch.nn.functional as F

model = torch.load('model.pth')
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(),
])

image_folder = 'path_to_images'
name = []
clas = []

for image_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, image_name)
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    name.append(image_name)
    clas.append(predicted.item())

data = {
    'image_name': name,
    'predicted_class': clas
}

df = pd.DataFrame(data)
df.to_csv('test.csv', index=False)
