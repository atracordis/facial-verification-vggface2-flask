from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from utils.utils import *

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f1 = request.files['pic_reference']
        f1.save("./photos/" + secure_filename(f1.filename))
        f2 = request.files['pic_challenger']
        f2.save("./photos/" + secure_filename(f2.filename))

        filenames = [f1.filename, f2.filename]
        filenames = ["./photos/" + file for file in filenames]
        embeddings = get_embeddings(filenames, model)
        text = is_match(embeddings[0], embeddings[1])

        return render_template('index.html', prediction_text=text)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=False, threaded=False)
