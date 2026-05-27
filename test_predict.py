from inference.predict import DeepfakePredictor
import os
import cv2
import numpy as np
import torch

print("Step 1: Initializing DeepfakePredictor...")
predictor = DeepfakePredictor(checkpoint_path="checkpoints/best_model.pth", device="cpu")
print("Step 1 completed: Predictor initialized.")

# Create a blank black image to use as test payload
img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.imwrite("test_debug.jpg", img)

image_path = "test_debug.jpg"

try:
    print("Step 2: Loading image and converting to RGB...")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("Step 2 completed.")

    print("Step 3: Running MTCNN face detection...")
    if predictor.mtcnn is not None:
        boxes, probs, _ = predictor.mtcnn.detect(img_rgb, landmarks=True)
        print(f"MTCNN boxes found: {boxes}, probs: {probs}")
    print("Step 3 completed.")

    print("Step 4: Resizing image and preparing single frame...")
    face = cv2.resize(img_rgb, (224, 224))
    frames = [face]
    fft_frames = [cv2.resize(cv2.imread(image_path), (224, 224))] # dummy fft resize
    print("Step 4 completed.")

    print("Step 5: Building sequence tensor...")
    T = predictor.config['sequence_length']
    rgb_tensor, fft_tensor, mask = predictor._build_sequence(frames, fft_frames, T)
    print(f"Sequence tensors ready. RGB shape: {rgb_tensor.shape}, FFT shape: {fft_tensor.shape}")
    print("Step 5 completed.")

    print("Step 6: Running Model forward pass...")
    with torch.no_grad():
        logit = predictor.model(
            rgb_tensor.unsqueeze(0).to(predictor.device),
            fft_tensor.unsqueeze(0).to(predictor.device),
            mask=mask.unsqueeze(0).to(predictor.device),
        )
    print(f"Model forward pass completed. Logit value: {logit.item()}")
    print("Step 6 completed.")

    print("Step 7: Generating GradCAM visualizations...")
    rgb = rgb_tensor.unsqueeze(0)
    fft = fft_tensor.unsqueeze(0)
    m = mask.unsqueeze(0)
    per_frame_cam, aggregate_cam = predictor.gradcam.compute_per_frame_gradcam(rgb, fft, m)
    print(f"GradCAM completed. per_frame_cam length: {len(per_frame_cam)}")
    print("Step 7 completed.")

    print("Step 8: Computing Integrated Gradients (IG)...")
    # This might take some time
    ig_attr, ig_delta = predictor.ig_explainer.compute_ig(rgb, fft, m)
    print(f"IG completed. ig_attr shape: {ig_attr.shape if hasattr(ig_attr, 'shape') else type(ig_attr)}")
    print("Step 8 completed.")

    print("Step 9: Generating final report composite...")
    # ...
    print("All steps succeeded!")

except Exception as e:
    print(f"Exception raised: {e}")
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists("test_debug.jpg"):
        os.remove("test_debug.jpg")
