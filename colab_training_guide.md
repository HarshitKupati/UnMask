# Google Colab T4 GPU Training Guide

Training a model on 142,000 images requires high-speed storage and a GPU. A Colab T4 GPU is perfect for this, but reading 142k loose images directly from Google Drive will be extremely slow. 

Follow these steps to efficiently migrate your project and train on Colab without I/O bottlenecks.

## Step 1: Zip the Project Locally
First, we need to package your codebase and extracted data. We must **exclude** the `.venv` folder, as Colab will install its own Linux-based dependencies.

Since your `data` folder contains over 140,000 files, standard Windows zipping tools often freeze or crash. To safely zip the project, open your terminal in the `deepfake__1` folder and run the custom Python script:

```bash
python zip_project.py
```
*(Note: This might take a few minutes since `data/faces` contains 142k files. The cursor will blink while it compresses the images—this is normal!)*

## Step 2: Upload to Google Drive
1. Go to your Google Drive (`drive.google.com`).
2. Upload the `deepfake__1.zip` file directly to the root of your My Drive.

## Step 3: Setup the Colab Notebook
1. Open Google Colab (`colab.research.google.com`) and create a **New Notebook**.
2. Go to **Runtime > Change runtime type** and select **T4 GPU**.
3. Create a new code cell and mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 4: Transfer and Unzip Data (Crucial for Speed)
**WARNING:** Do **not** train the model directly off Google Drive (e.g. `%cd /content/drive/MyDrive/...`). Google Drive has severe rate limits for reading thousands of small files, which will bottleneck the GPU. Always unzip to `/content/` first!

Create a new code cell and run this to copy the zip to the fast Colab disk and extract it:

```bash
# Copy zip to fast local storage
!cp /content/drive/MyDrive/deepfake__1.zip /content/

# Unzip quietly (-q) to prevent output crashing the browser
!unzip -q /content/deepfake__1.zip -d /content/deepfake__1

# Move into the project folder
%cd /content/deepfake__1
```

## Step 5: Install Dependencies
Install your exact Python packages using the requirements file. Note that we use `--upgrade` to resolve any conflicts with Colab's pre-installed packages.

```bash
!pip install -r requirements.txt --upgrade
!pip install wandb shap retinaface-pytorch --upgrade
```

> [!WARNING]
> After running the installation, you may see a warning about previously imported packages (like `numpy`). **You must Restart the Session** (Runtime -> Restart session) in Colab before moving on to the next step, or you will encounter errors!

## Step 6: Create Experiment Split (Recommended for T4)
To drastically speed up training, use a subset of your dataset for an initial functional model.

```bash
!python data_utils/create_splits.py --max_samples 2500
```

## Step 7: Start Training!
Run the main script. Your GPU will now tear through the dataset!

```bash
!python main.py --mode train --data_dir data/faces
```

## Step 8: Saving Checkpoints
As your model trains, it saves `.pth` checkpoints to the `checkpoints/` directory inside Colab. **Since Colab deletes all files when it disconnects**, you should copy your trained models back to Google Drive periodically!

You can add this to a new cell and run it while training happens, or wait until it finishes:
```bash
# Copy best model back to Google Drive
!cp /content/deepfake__1/checkpoints/best_model.pth /content/drive/MyDrive/
```
