# backend/app.py
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import io
import os
import traceback
from style_transfer import stylize_bytes

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "https://neural-style-transfer-two.vercel.app"
    ])

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Backend is running"})

@app.route('/stylize', methods=['POST'])
def stylize():
    try:
        if 'content' not in request.files or 'style' not in request.files:
            return jsonify({'error': 'Missing file(s). Please upload both `content` and `style`.'}), 400

        content_file = request.files['content'].read()
        style_file = request.files['style'].read()

        print("Files received, running fast style transfer...")
        output_image = stylize_bytes(content_file, style_file)  # PIL Image

        # Save to bytes and return
        img_io = io.BytesIO()
        output_image.save(img_io, format='PNG')
        img_io.seek(0)
        print("Stylization done, sending response.")
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        tb = traceback.format_exc()
        print("Error during /stylize:", str(e))
        print(tb)
        return jsonify({'error': str(e), 'trace': tb}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)