from flask import Flask, request, render_template, redirect, url_for
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)

# Memuat model CNN
model = load_model('models/model.h5')

# Definisikan label kelas
CLASS_NAMES = ['Blight', 'Common Rust', 'Gray Leaf', 'Healthy']
HANDLING_TIPS = {
    'Common Rust': {
        "title": "Tips Mengatasi Common Rust",
        "tips": [
            "1. Gunakan varietas jagung tahan: Pilih benih yang tahan terhadap common rust (BISI-18, Pioneer P27, NK7328).",
            "2. Pengendalian kimia: Semprotkan fungisida berbasis strobilurin atau triazol jika penyakit terdeteksi dini.",
            "3. Rotasi tanaman: Hindari menanam jagung di lahan yang sama secara berturut-turut untuk mengurangi inokulum patogen.",
            "4. Pola tanam yang baik: Jaga jarak tanam agar sirkulasi udara meningkat dan kelembapan berkurang."
        ]
    },
    'Gray Leaf': {
        "title": "Tips Mengatasi Gray Leaf Spot",
        "tips": [
            "1. Varietas tahan penyakit: Tanam varietas jagung yang memiliki ketahanan terhadap gray leaf spot (BISI-2, NK7328).",
            "2. Pemangkasan daun yang terinfeksi: Segera buang daun yang menunjukkan gejala untuk mencegah penyebaran.",
            "3. Fungisida preventif: Gunakan fungisida berbasis azoksistrobin atau propikonazol saat kondisi mendukung infeksi (kelembapan tinggi dan suhu hangat).",
            "4. Hindari irigasi atas yang menyebabkan daun basah.",
            "5. Lakukan pengelolaan sisa tanaman dengan membajak atau membakar sisa tanaman yang terinfeksi."
        ]
    },
    'Blight': {
        "title": "Tips Mengatasi Blight",
        "tips": [
            "1. Gunakan benih unggul: Pilih varietas jagung tahan terhadap blight (Pioneer P27, BIMA 20 URI).",
            "2. Fungisida: Semprotkan fungisida sistemik seperti mankozeb, flutriafol, atau propikonazol saat gejala awal muncul.",
            "3. Tanam tepat waktu: Hindari menanam jagung terlalu awal di musim hujan, karena kondisi lembap mendukung infeksi.",
            "4. Pengelolaan sisa tanaman: Hancurkan sisa tanaman yang terinfeksi agar tidak menjadi sumber inokulum.",
            "5. Rotasi tanaman: Tanam tanaman selain jagung (seperti kedelai) untuk memutus siklus hidup patogen."
        ]
    },
    'Healthy': {
        "title": "Tanaman Anda Sehat!",
        "tips": [
            "Selamat, tanaman Anda sehat! Pastikan untuk terus merawatnya dengan pemupukan dan pengendalian hama secara berkala."
        ]
    }
}

# Fungsi untuk memproses gambar
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html', contents=['base.html'])

@app.route('/about')
def about():
    return render_template('index.html', contents=['about.html'])

@app.route('/information')
def information():
    return render_template('index.html', contents=['information.html'])

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':
        # Periksa apakah file diunggah
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('detection.html', error="No file selected")
        
        try:
            # Proses gambar langsung dari request.files tanpa menyimpannya ke disk
            image = Image.open(file.stream)  # Membaca gambar dari stream (memori)
            processed_image = preprocess_image(image)

            # Prediksi
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_probability = predictions[0][predicted_class_index] * 100  # Akurasi untuk kelas prediksi terpilih
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            handling_tips = HANDLING_TIPS[predicted_class_name]
            confidence = float(np.max(predictions))

            # Konversi gambar ke format base64 agar bisa ditampilkan di web
            img_io = io.BytesIO()
            image.save(img_io, 'PNG')  # Menyimpan gambar dalam format PNG
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

            # Render hasil di halaman yang sama
            return render_template(
                'index.html',
                contents=['detection.html'],
                predicted_class=predicted_class_name,
                handling_tip = handling_tips,
                confidence_score = confidence,
                file_uploaded=True,
                img_base64=img_base64,  # Gambar dalam base64 untuk ditampilkan
                predicted_probability=predicted_probability,
                probabilities={CLASS_NAMES[i]: predictions[0][i] * 100 for i in range(len(CLASS_NAMES))}
            )
        except Exception as e:
            return render_template('index.html', error=f"Error processing image: {str(e)}")

    # Jika GET request, tampilkan formulir upload
    return render_template('index.html', contents=['detection.html'])

if __name__ == '__main__':
    app.run(debug=True)
