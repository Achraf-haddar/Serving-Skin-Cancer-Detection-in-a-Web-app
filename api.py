# way to upload image: endpoint
# way to save the image
# function to make prediction on the image
# show the results
import os
from flask import Flask
from flask import request
from flask import render_template

app = Flask(__name__)
UPLOAD_FOLDER = "/home/achraf/Desktop/workspace/SkinCancerDetection/static"

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
            return render_template("index.html", prediction=1)        
    return render_template("index.html", prediction=0)

if __name__ == "__main__":
    app.run(port=12000, debug=True)