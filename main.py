from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')  # Halaman detection
    
@app.route('/result')
def result():
    return render_template('result.html')  # Halaman result

if __name__ == '__main__':
    app.run(debug=True)