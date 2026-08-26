"""
Main processing pipeline for fashion trend forecasting.

This module orchestrates the full pipeline:
1. Data loading and preprocessing
2. Image segmentation (YOLO + SAM)
3. Feature extraction via autoencoder
4. Clustering
5. Results saving for dashboard
"""

import os
import sys
import zipfile
import pandas as pd
import numpy as np
import cv2
import requests
import time
import torch
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Optional imports (may not be installed in all environments)
try:
    from segment_anything import sam_model_registry, SamPredictor
    from ultralytics import YOLO
    import supervision as sv
    HAS_CV_DEPS = True
except ImportError:
    HAS_CV_DEPS = False


def extract_zip(zip_file_path: str, extraction_path: str) -> bool:
    """Extract zip file to target directory."""
    if not os.path.exists(zip_file_path):
        print(f"Error: The file '{zip_file_path}' does not exist.")
        return False
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)
        print(f"Successfully extracted to '{extraction_path}'")
        return True
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        return False


def replace_emojis_with_text(text: str) -> str:
    """Replace emojis with textual descriptions."""
    if not isinstance(text, str):
        return text
    try:
        import emoji
        return emoji.demojize(text, delimiters=(" ", " "))
    except ImportError:
        return text


def read_data(path: str) -> pd.DataFrame:
    """Read JSON data file."""
    return pd.read_json(path)


def download_images(df: pd.DataFrame, batch_size: int = 50, delay: int = 5) -> None:
    """Download images in batches with delay between batches."""
    images_dir = PROJECT_ROOT / "images" / "original_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        for _, row in batch.iterrows():
            url = row.get('image_url') or row.get('url')
            img_id = row.get('id') or row.get('post_id')
            if url and img_id:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        img_path = images_dir / f"{img_id}.jpg"
                        with open(img_path, 'wb') as f:
                            f.write(response.content)
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
        if i + batch_size < len(df):
            time.sleep(delay)


def load_yolo_model(model_name: str = "yolov8n.pt"):
    """Load YOLO model for object detection."""
    if not HAS_CV_DEPS:
        raise ImportError("Computer vision dependencies not installed")
    return YOLO(model_name)


def load_sam_model(model_type: str = "vit_h", checkpoint_path: str = None):
    """Load SAM model for segmentation."""
    if not HAS_CV_DEPS:
        raise ImportError("Computer vision dependencies not installed")
    
    if checkpoint_path is None:
        checkpoint_path = PROJECT_ROOT / "models" / "sam" / "sam_vit_h_4b8939.pth"
    
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    return SamPredictor(sam)


def convert_bbox_x1y1x2y2_to_xywh(x1, y1, x2, y2):
    """Convert bounding box from (x1, y1, x2, y2) to (x, y, w, h) format."""
    return x1, y1, x2 - x1, y2 - y1


def segment_clothing(yolo_model, sam_predictor, image_path: str, output_dir: str):
    """Segment clothing items from an image using YOLO + SAM."""
    if not HAS_CV_DEPS:
        raise ImportError("Computer vision dependencies not installed")
    
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # YOLO detection
    results = yolo_model(image_rgb)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter for clothing classes (adjust class IDs as needed)
    clothing_classes = [0]  # person class, adjust based on your model
    detections = detections[np.isin(detections.class_id, clothing_classes)]
    
    if len(detections) == 0:
        return []
    
    sam_predictor.set_image(image_rgb)
    segmented_images = []
    
    for i, (xyxy, _, _, _) in enumerate(detections):
        x1, y1, x2, y2 = xyxy.astype(int)
        masks, scores, _ = sam_predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False
        )
        mask = masks[0]
        
        # Create segmented image (black background, only clothing)
        segmented = np.zeros_like(image_rgb)
        segmented[mask] = image_rgb[mask]
        
        output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_seg_{i}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
        segmented_images.append(output_path)
    
    return segmented_images


def get_images_and_filenames(path: str):
    """Get all images and filenames from a directory."""
    image_list = []
    filename_list = []
    
    for filename in os.listdir(path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                image_list.append(img)
                filename_list.append(filename)
    
    return np.array(image_list), filename_list


def extract_autoencoder_features(autoencoder_path: str, images_path: str, output_path: str):
    """Extract latent space features using trained autoencoder."""
    try:
        from tensorflow.keras.models import Model, load_model
    except ImportError:
        print("TensorFlow/Keras not installed. Skipping feature extraction.")
        return
    
    autoencoder = load_model(autoencoder_path)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)
    
    images, filenames = get_images_and_filenames(images_path)
    if len(images) == 0:
        print("No images found")
        return
    
    latent_spaces = encoder.predict(images, verbose=1)
    
    df = pd.DataFrame({
        'filename': filenames,
        'latent_space': list(latent_spaces)
    })
    df.to_hdf(output_path, key='df_items', mode='w')
    print(f"Saved latent spaces to {output_path}")


def run_clustering(latent_spaces_path: str, n_clusters: int = 10, output_path: str = None):
    """Run K-Means clustering on latent spaces."""
    from sklearn.cluster import KMeans
    
    df = pd.read_hdf(latent_spaces_path, key='df_items')
    latent_space = np.stack(df['latent_space'].values)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(latent_space)
    
    df['cluster'] = clusters
    
    if output_path:
        df.to_hdf(output_path, key='df_items', mode='w')
        print(f"Saved clustered data to {output_path}")
    
    return df


def main():
    """Main pipeline execution."""
    print("Starting fashion trend forecasting pipeline...")
    
    # Step 1: Extract data (if zip provided)
    zip_path = PROJECT_ROOT / "Future-Fashion-Trends-Forecasting-main.zip"
    if zip_path.exists():
        extract_zip(str(zip_path), str(PROJECT_ROOT))
    
    # Step 2: Download images (if posts.csv exists)
    posts_csv = PROJECT_ROOT / "Future-Fashion-Trends-Forecasting-main" / "data" / "posts.csv"
    if posts_csv.exists():
        df = pd.read_csv(posts_csv)
        download_images(df)
    
    # Step 3: Image segmentation (requires CV deps)
    if HAS_CV_DEPS:
        try:
            yolo = load_yolo_model()
            sam = load_sam_model()
            
            images_dir = PROJECT_ROOT / "images" / "original_images"
            segmented_dir = PROJECT_ROOT / "images" / "segmented_images"
            segmented_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in os.listdir(images_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = images_dir / img_file
                    segment_clothing(yolo, sam, str(img_path), str(segmented_dir))
        except Exception as e:
            print(f"Segmentation failed: {e}")
    else:
        print("Skipping segmentation - CV dependencies not installed")
    
    # Step 4: Feature extraction (requires trained autoencoder)
    autoencoder_path = PROJECT_ROOT / "models" / "autoencoder.h5"
    segmented_dir = PROJECT_ROOT / "images" / "segmented_images"
    latent_output = PROJECT_ROOT / "Future-Fashion-Trends-Forecasting-main" / "data" / "latent_spaces.h5"
    
    if autoencoder_path.exists() and segmented_dir.exists():
        extract_autoencoder_features(str(autoencoder_path), str(segmented_dir), str(latent_output))
    else:
        print("Skipping feature extraction - autoencoder or segmented images not found")
    
    # Step 5: Clustering
    if latent_output.exists():
        clustered_output = PROJECT_ROOT / "Future-Fashion-Trends-Forecasting-main" / "data" / "clustered_data.h5"
        run_clustering(str(latent_output), n_clusters=10, output_path=str(clustered_output))
    else:
        print("Skipping clustering - latent spaces not found")
    
    print("Pipeline completed!")


if __name__ == "__main__":
    main()
