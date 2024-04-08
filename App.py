from flask import Flask, render_template, flash, request, session

import numpy as np
from keras.preprocessing import image
import warnings
import os

app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/Predict")
def Predict():
    return render_template('Predict.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        import tensorflow as tf
        import cv2

        file = request.files['file']
        file.save('static/upload/Test.png')
        fname = 'static/upload/Test.png'

        img1 = cv2.imread('static/upload/Test.png')

        dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
        noi = 'static/upload/noi.png'

        cv2.imwrite(noi, dst)

        import warnings
        warnings.filterwarnings('ignore')

        classifierLoad = tf.keras.models.load_model('model.h5')
        test_image = image.load_img('static/upload/Test.png', target_size=(200, 200))
        test_image = np.expand_dims(test_image, axis=0)
        result = classifierLoad.predict(test_image)
        print(result)
        ind = np.argmax(result)

        if ind == 0:
            print("ImpureOvarionCyst")
            out = "ImpureOvarionCyst"
            pre = "Tri-Sprintec, Estarylla, Mono-Linyah"
        elif ind == 1:
            print("NormalPelvic")
            out = "NormalPelvic"
            pre = "Nil"
        elif ind == 1:
            print("OvarianCyst")
            out = "OvarianCyst"
            pre = "Afinitor (Everolimus) Afinitor Disperz (Everolimus)"

        return render_template('Predict.html', pre=pre, result=out, org=fname, noi=noi)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
