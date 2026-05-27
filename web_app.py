from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
from inference.predict import DeepfakePredictor
import uuid
import torch

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Detect and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize predictor
try:
    predictor = DeepfakePredictor(checkpoint_path="checkpoints/best_model.pth", device=device)
    model_loaded = True
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    predictor = None
    model_loaded = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Process an image and return results"""
    if not model_loaded:
        return {'error': 'Model not loaded'}
    
    try:
        result = predictor.predict_image(image_path, compute_xai=True)
        return result
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def process_video(video_path):
    """Process a video and return results"""
    if not model_loaded:
        return {'error': 'Model not loaded'}
    
    try:
        result = predictor.predict_video(video_path, compute_xai=True)
        return result
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def serialize_result(result, original_filename):
    """Serialize numpy arrays in results, saving them as files and returning clean JSON"""
    if 'error' in result:
        return result

    unique_id = uuid.uuid4().hex[:8]
    base_name = os.path.splitext(original_filename)[0]
    safe_base = "".join([c if c.isalnum() else "_" for c in base_name])
    safe_name = f"{safe_base}_{unique_id}"

    serialized = {
        'prediction': float(result['prediction']),
        'label': result['label'],
        'confidence': result['confidence'],
    }

    # Save and reference composite visualization
    if 'visualization' in result and isinstance(result['visualization'], np.ndarray):
        vis_filename = f"vis_{safe_name}_composite.png"
        vis_path = os.path.join(app.config['UPLOAD_FOLDER'], vis_filename)
        cv2.imwrite(vis_path, result['visualization'])
        serialized['visualization_url'] = f"/uploads/{vis_filename}"

    # Save and reference Integrated Gradients (IG) attribution heatmap
    if 'ig_attribution' in result and isinstance(result['ig_attribution'], np.ndarray):
        ig_filename = f"vis_{safe_name}_ig.png"
        ig_path = os.path.join(app.config['UPLOAD_FOLDER'], ig_filename)
        cv2.imwrite(ig_path, result['ig_attribution'])
        serialized['ig_attribution_url'] = f"/uploads/{ig_filename}"

    # Handle temporal attention weights
    if 'attention_weights' in result:
        if isinstance(result['attention_weights'], np.ndarray):
            serialized['attention_weights'] = result['attention_weights'].tolist()
        else:
            serialized['attention_weights'] = list(result['attention_weights'])

    # Handle top suspicious frames
    if 'top_suspicious_frames' in result:
        serialized['top_suspicious_frames'] = result['top_suspicious_frames']

    # Handle individual crops and GradCAM overlays for suspicious frames
    if 'top_suspicious_crops' in result and 'top_suspicious_overlays' in result:
        suspicious_list = []
        for i, idx in enumerate(result.get('top_suspicious_frames', [])):
            item = {'index': idx}
            
            # Save original frame crop
            if i < len(result['top_suspicious_crops']) and isinstance(result['top_suspicious_crops'][i], np.ndarray):
                crop_filename = f"vis_{safe_name}_crop_{idx}.png"
                crop_path = os.path.join(app.config['UPLOAD_FOLDER'], crop_filename)
                cv2.imwrite(crop_path, result['top_suspicious_crops'][i])
                item['crop_url'] = f"/uploads/{crop_filename}"
                # Add attention weight for this frame
                if 'attention_weights' in result and idx < len(result['attention_weights']):
                    item['attention_weight'] = float(result['attention_weights'][idx])
                
            # Save GradCAM overlay frame
            if i < len(result['top_suspicious_overlays']) and isinstance(result['top_suspicious_overlays'][i], np.ndarray):
                overlay_filename = f"vis_{safe_name}_overlay_{idx}.png"
                overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_filename)
                cv2.imwrite(overlay_path, result['top_suspicious_overlays'][i])
                item['overlay_url'] = f"/uploads/{overlay_filename}"

            suspicious_list.append(item)
        serialized['suspicious_details'] = suspicious_list

    return serialized

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded, device=device if model_loaded else 'None')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Generate secure UUID filename to prevent path traversal and collision
        original_filename = file.filename
        file_ext = original_filename.rsplit('.', 1)[1].lower()
        unique_id = uuid.uuid4().hex[:12]
        filename = f"upload_{unique_id}.{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save uploaded file
            file.save(file_path)
            
            # Determine file type
            is_video = file_ext in ['mp4', 'avi', 'mov']
            
            # Process file
            if is_video:
                result = process_video(file_path)
                result['file_type'] = 'video'
            else:
                result = process_image(file_path)
                result['file_type'] = 'image'
            
            # Add metadata
            result['filename'] = original_filename
            
            # Serialize results
            serialized_result = serialize_result(result, original_filename)
            serialized_result['file_type'] = result['file_type']
            serialized_result['filename'] = original_filename
            serialized_result['device'] = device
            
            return jsonify(serialized_result)
        
        except Exception as e:
            return jsonify({'error': f'Inference error: {str(e)}'})
            
        finally:
            # Clean up the original uploaded file to prevent storage leaks
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as cleanup_err:
                    print(f"Warning: Failed to clean up file {file_path}: {cleanup_err}")
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Securely serve uploaded visualizations and templates"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)