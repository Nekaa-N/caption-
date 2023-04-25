

import os
from rich.progress import track
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from flask import Flask , render_template  , request , send_file

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT




def predict_step(image_paths):
  images = []
 
  i_image = Image.open(image_paths)
  if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

  images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values


  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds




@app.route('/')
def home():
        return render_template("home.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
       
            
        if (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename
                print(img_path)
                class_result  = predict_step(img_path)

                predictions = {
                      "class1":class_result,
                        
                        
                        
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                print(1)
                return  render_template('home.html' , predictions = class_result[0])
            else:
                return render_template('home.html' , error = error)

    else:
        return render_template('home.html')






if __name__ == "__main__":
    app.run(debug = True)