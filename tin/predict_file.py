import os
from flask import Flask,request,redirect,url_for,render_template
from werkzeug.utils import secure_filename

from keras.models import Sequential,load_model
import keras,sys
import numpy as np
from PIL import Image

import tensorflow as tf

import base64
from io import BytesIO

classes = ["bottle","chimney","mushroom","sausage","snake"]
num_classes = len(classes)
image_size = 50

graph = tf.get_default_graph()
model = load_model("./tin_cnn.h5")

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png','jpg','gif','jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET','POST'])
def upload_file():
    global graph
    with graph.as_default():

        if request.method == 'POST':
            if 'file' not in request.files:
                 flash('ファイルがありません')
                 return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                 flash('ファイルがありません')
                 return redirect(request.url)
            if file and  allowed_file(file.filename):
                 filename = secure_filename(file.filename)
                 file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                 filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)

                 image = Image.open(filepath)
                 image = image.convert("RGB")
                 image = image.resize((image_size,image_size))
                 data = np.asarray(image)
                 X = []
                 X.append(data)
                 X = np.array(X)

                 result = model.predict([X])[0]
                 predicted = result.argmax()
                 percentage = int(result[predicted]*100)

                 if classes[predicted] == "bottle":
                     return render_template("tin_bottle.html")
                 if classes[predicted] == "chimney":
                     return render_template("tin_chimney.html")
                 if classes[predicted] == "mushroom":
                     return render_template("tin_mushroom.html")
                 if classes[predicted] == "sausage":
                     return render_template("tin_sausage.html")
                 if classes[predicted] == "snake":
                     return render_template("tin_snake.html")

         # 画像書き込み用バッファを確保して画像データをそこに書き込む
        img_in = "templates/zouu.jpeg"
        img_pil = Image.open(img_in)
        buf = BytesIO()
        img_pil.save(buf,format="jpeg")

        # バイナリデータをbase64でエンコードし、それをさらにutf-8でデコードしておく
        zoustr = base64.b64encode(buf.getvalue()).decode("utf-8")

        # image要素のsrc属性に埋め込めこむために、適切に付帯情報を付与する
        zoudata = "data:image/jpeg;base64,{}".format(zoustr)

        #return redirect(url_for('uploaded_file',filename=filename))
        return render_template("tintin.html",zoudata=zoudata)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FORDER'],filename)

#set FLASK_APP=predict_file.py　入力後　python -m flask run で起動
