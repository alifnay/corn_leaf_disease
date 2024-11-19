from flask import Flask, request, render_template, redirect, url_for
import os
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Tentukan folder untuk menyimpan file sementara
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Max size 10MB

# Memuat model CNN
model = load_model('models/model.h5')

# Definisikan label kelas
class_names = ['Blight', 'Common Rust', 'Gray Leaf', 'Healthy']

# Fungsi untuk memproses gambar
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Pastikan folder upload ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':
        # Periksa apakah file diunggah
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('detection.html', error="No file selected")
        
        try:
            # Simpan file sementara
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Proses gambar
            image = Image.open(file_path)
            processed_image = preprocess_image(image)

            # Prediksi
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_probability = predictions[0][predicted_class_index] * 100  # Akurasi untuk kelas prediksi terpilih
            predicted_class_name = class_names[predicted_class_index]

            # Hapus file setelah selesai
            os.remove(file_path)

            # Render hasil di halaman yang sama
            return render_template(
                'detection.html',
                predicted_class=predicted_class_name,
                predicted_probability=predicted_probability,
                probabilities={class_names[i]: predictions[0][i] * 100 for i in range(len(class_names))}
            )
        except Exception as e:
            return render_template('detection.html', error=f"Error processing image: {str(e)}")

    # Jika GET request, tampilkan formulir upload
    return render_template('detection.html')

if __name__ == '__main__':
    app.run(debug=True)
