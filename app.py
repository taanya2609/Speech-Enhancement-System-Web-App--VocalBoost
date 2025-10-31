import os
import numpy
import matplotlib
import torch
import uuid
import subprocess
from flask import Flask, render_template, request, jsonify, send_file
import soundfile as sf
import librosa
from pydub import AudioSegment
from utils.enhance_audio import enhance_audio
from utils.plot_spectrogram import plot_spectrogram

# --- FIX: Set FFmpeg path explicitly to bypass system PATH issues ---
# This is the most reliable way to ensure the Flask app can find FFmpeg.
# You must replace the placeholder path with your actual path.
# Example: r"C:\path\to\your\ffmpeg.exe" or "/usr/local/bin/ffmpeg"
# Double-check that the path points directly to the executable file itself.
os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\ffmpeg-2025-07-28-git-dc8e753f32-full_build\bin\ffmpeg.exe"

app = Flask(__name__)

# Ensure necessary directories exist
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, 'images'), exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

@app.route('/')
def index():
    """Renders the main landing page."""
    return render_template('index.html')

@app.route('/enhance')
def enhance():
    """Renders the audio enhancement page."""
    return render_template('enhance.html')

@app.route('/howitworks')
def howitworks():
    """Renders the 'How It Works' page."""
    return render_template('howitworks.html')

@app.route('/faqs')
def faqs():
    """Renders the FAQs page."""
    return render_template('faqs.html')

@app.route('/about')
def about():
    """Renders the 'About' page."""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Renders the 'Contact' page."""
    return render_template('contact.html')

@app.route('/features')
def features():
    """Renders the 'Features' page."""
    return render_template('features.html')



@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Handles audio file uploads and starts the enhancement process."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_id = str(uuid.uuid4())
        
        # Determine file extension
        original_filename = file.filename
        file_ext = os.path.splitext(original_filename)[1].lower()

        # Save the original file with its correct extension
        temp_input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}{file_ext}")
        file.save(temp_input_path)

        # Path for the WAV file that will be used for processing
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.wav")

        # Convert to WAV if necessary
        if file_ext != '.wav':
            try:
                # Use subprocess to call ffmpeg directly for conversion
                # The -i flag specifies the input file, and the output is the WAV file.
                command = [os.environ["FFMPEG_BINARY"], '-i', temp_input_path, '-acodec', 'pcm_s16le', '-ar', '44100', input_path]
                subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                # Provides detailed error message if ffmpeg command fails
                error_output = e.stderr
                return jsonify({'error': f"FFmpeg conversion failed: {error_output}"}), 500
            except FileNotFoundError:
                # This error means ffmpeg is not found
                return jsonify({'error': "FFmpeg executable not found. Please ensure it's in your system's PATH."}), 500
            except Exception as e:
                return jsonify({'error': f"File conversion failed: {str(e)}"}), 500
        else:
            input_path = temp_input_path

        # Get enhancement level from form data
        enhancement_level = request.form.get('enhancement_level', 'normal')

        # Run enhancement and generate spectrograms
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{file_id}_enhanced.wav")
        
        try:
            enhanced_path, sr, confidence = enhance_audio(input_path, output_path, level=enhancement_level)
        except Exception as e:
            return jsonify({'error': f'Audio enhancement failed: {str(e)}'}), 500

        if enhanced_path:
            # Generate spectrograms
            original_spectrogram_path = os.path.join(app.config['STATIC_FOLDER'], 'images', f"{file_id}_original_spec.png")
            enhanced_spectrogram_path = os.path.join(app.config['STATIC_FOLDER'], 'images', f"{file_id}_enhanced_spec.png")
            
            try:
                plot_spectrogram(input_path, original_spectrogram_path)
                plot_spectrogram(enhanced_path, enhanced_spectrogram_path)
            except Exception as e:
                return jsonify({'error': f'Spectrogram plotting failed: {str(e)}'}), 500
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'confidence': confidence
            })
        
        return jsonify({'error': 'Audio enhancement failed.'}), 500

@app.route('/result/<file_id>')
def result(file_id):
    """Renders the result page with enhanced audio and spectrograms."""
    original_audio_path = f"/uploads/{file_id}.wav"
    enhanced_audio_path = f"/processed/{file_id}_enhanced.wav"
    
    # Check if files exist
    if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.wav")):
        return "File not found", 404

    return render_template('result.html', 
        file_id=file_id, 
        original_audio=original_audio_path, 
        enhanced_audio=enhanced_audio_path
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the original uploaded audio file."""
    file_id = os.path.splitext(filename)[0]
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        if file.startswith(file_id):
            return send_file(os.path.join(app.config['UPLOAD_FOLDER'], file))
    return "File not found", 404

@app.route('/processed/<filename>')
def processed_file(filename):
    """Serves the enhanced audio file."""
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)