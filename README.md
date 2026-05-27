# Deepfake Detection System

Welcome to the Deepfake Detection System. This project uses an advanced AI model combining EfficientNet and BiLSTM to identify whether a video or image is a deepfake (manipulated) or real. It also includes Explainable AI (XAI) features to visualize and explain why the model reached its decision.

---

## Quick Start: Explore via Web App

The easiest way to use this system is through the built-in Flask web interface. Follow these steps to set up a Python virtual environment, install the required packages, and run the dashboard.

### 1. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies securely. Open your terminal in the project root directory and follow the commands below.

**Create the virtual environment:**
```bash
python -m venv venv
```

**Activate the virtual environment:**
- **On Windows (PowerShell):**
  ```powershell
  venv\Scripts\Activate
  ```
- **On Windows (Command Prompt):**
  ```cmd
  venv\Scripts\activate.bat
  ```
- **On macOS and Linux:**
  ```bash
  source venv/bin/activate
  ```

### 2. Install Dependencies
Once the virtual environment is activated, install all required machine learning and web dependencies:
```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
pip install flask
```

### 3. Run the Web Dashboard
Start the Flask web application inside the virtual environment using the following command:
```bash
python web_app.py
```

### 4. Access the Webpage
Open your web browser and navigate to `http://localhost:5000` (or `http://127.0.0.1:5000`).

## Web Application Features

The Flask-based deepfake detection web application provides a comprehensive forensic analysis interface with the following features:

### User Interface
- **Modern Dark Theme UI**: Professional dashboard with gradient backgrounds, glassmorphism effects, and smooth animations
- **Separate Upload Buttons**: Dedicated cards for image and video uploads with drag-and-drop support
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Real-Time Status Indicators**: Shows model loading status and device type (CUDA/CPU) in the header

### Upload & Analysis
- **Image Analysis**: Upload PNG, JPG, or JPEG images for single-frame deepfake detection
- **Video Analysis**: Upload MP4, AVI, or MOV videos for sequence-based forensic analysis
- **Progress Tracking**: Live scanning progress bar with percentage completion
- **Context-Aware Pipeline Logs**: Real-time terminal-style log showing processing stages:
  - Stage 1: Payload initialization and parsing
  - Stage 2: MTCNN face detection and alignment
  - Stage 3: EfficientNet spatial feature extraction
  - Stage 4: FFT frequency spectrum analysis
  - Stage 5: BiLSTM temporal sequence processing (videos only)
  - Stage 6: GradCAM spatial attention generation
  - Stage 7: Integrated Gradients pixel saliency computation
  - Stage 8: Composite report consolidation

### XAI (Explainable AI) Visualizations
The application provides comprehensive explainable AI outputs to help understand model decisions:

1. **Suspicious Frames Tab**:
   - Displays top 3 frames with highest neural network activations
   - Hover-to-reveal GradCAM heatmaps showing spatial attention
   - Frame indices with attention weight percentages

2. **Pixel Saliency (Integrated Gradients) Tab**:
   - Pixel-level attribution maps showing which regions influenced the decision
   - Hot colormap highlighting anomalous boundaries and manipulation artifacts
   - Identifies sub-pixel edges and blending anomalies

3. **Attention Weights Tab** (videos only):
   - Temporal attention curve from BiLSTM recurrence layer
   - Interactive Chart.js visualization with peak highlighting
   - Shows which frames triggered the highest suspicion scores

4. **Composite Report Tab**:
   - Multi-tier explanation sheet combining all XAI visualizations
   - Downloadable PNG report for documentation
   - Comprehensive forensic summary

### Analysis Results
- **Verdict Gauge**: Circular progress indicator showing deepfake probability
- **Classification Labels**: AUTHENTIC, SUSPICIOUS, or DEEPFAKE with color coding
- **Confidence Scores**: Detailed confidence metrics with engine information
- **Payload Preview**: Real-time preview of uploaded image or video

### Technical Features
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Test-Time Augmentation**: 5-version TTA for improved accuracy
- **Temperature Scaling**: Calibrated probability outputs
- **Automated Cleanup**: Immediate deletion of uploaded files after processing
- **Error Handling**: Graceful error messages and system alerts
- **Missing Checkpoint Detection**: Banner alert when model checkpoint is not found

---

## Advanced Usage (For Developers & Training)

For developers looking to train the model, extract face crops, or run batch processing, a command-line interface is available via `main.py`.

### Training on Google Colab (Free T4 GPU)
For training on large datasets, we recommend using Google Colab.
1. Run `python zip_project.py` to compress the project directory.
2. Upload the `deepfake__1.zip` file to your Google Drive.
3. Open Google Colab, mount your drive, and extract the zip file.
4. Install requirements: `!pip install -r requirements.txt && pip install flask`
5. Extract faces: `!python main.py --mode extract --input_dir data/raw --output_dir data/faces`
6. Train the model: `!python main.py --mode train --data_dir data/faces`

### Running Commands Locally
You can run inference directly from your command-line interface:

**For a single image:**
```bash
python main.py --mode predict --input path/to/image.jpg --checkpoint checkpoints/best_model.pth
```

**For a single video:**
```bash
python main.py --mode predict --input path/to/video.mp4 --checkpoint checkpoints/best_model.pth
```
(The XAI visualizations will be saved to the `xai_samples/` directory.)

---

## Project Structure

- `web_app.py`: The Flask web application backend and prediction serving layer.
- `templates/index.html`: Responsive HTML/CSS/JS frontend dashboard.
- `app.py`: Streamlit-based legacy web application explorer.
- `main.py`: Core entry point for command-line execution (Training/Extracting/Predictions).
- `config.py`: Global configuration and hyperparameters.
- `models/`: Neural network architecture modules (EfficientNet and LSTM).
- `training/`: Training and validation scripts.
- `xai/`: Explainable AI algorithms (GradCAM, Integrated Gradients) and visualization tools.
- `checkpoints/`: Directory where trained models (e.g., `best_model.pth`) are saved and loaded.
