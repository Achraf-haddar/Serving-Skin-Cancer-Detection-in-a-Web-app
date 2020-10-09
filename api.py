# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
import torch

import albumentations
import pretrainedmodels

import numpy as np
import torch.nn as nn

from flask import Flask
from flask import request
from flask import render_template
from torch.nn import functional as F

from dataset import ClassificationDataset
import engine
 

app = Flask(__name__)
UPLOAD_FOLDER = "/home/achraf/app/static"
DEVICE = "cpu"
MODEL = None

class SEResNex50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNex50_32x4d, self).__init__()
        self.base_model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=pretrained)
        # To check the number of output features
        # Run this line and check the in_features 
        # pretrained.__dict__["se_resnext50_32_x4d"]()
        self.l0 = nn.Linear(2048, 1)
    
    def forward(self, image):
        bs, _, _, _ = image.shape
        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = torch.sigmoid(self.l0(x))
        return out 

def predict(image_path, model):
    mean = (0.458, 0.456, 0.406)  # mean for this model
    std = (0.229, 0.224, 0.225)  # std for this model

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )
    test_images = [image_path]
    test_targets = [0]

    # Test data loader
    test_dataset = ClassificationDataset(
        image_path=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    predictions = engine.evaluate(
        data_loader=test_loader,
        model=model,
        device=DEVICE
    )
    return np.vstack((predictions)).ravel()



@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"] # name provided in index.html
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename # give the name of the image .jpg
            )
            image_file.save(image_location)
            # generate the prediction
            pred = predict(image_location, MODEL)[0]
            return render_template("index.html", prediction=pred, image_loc=image_file.filename)        
    return render_template("index.html", prediction=0, image_loc=None)

if __name__ == "__main__":
    MODEL = SEResNex50_32x4d(pretrained=None)
    MODEL.load_state_dict(torch.load("model.bin", map_location=torch.device(DEVICE)))
    MODEL.to(DEVICE)

    app.run(host="0.0.0.0", port=12000, debug=True)