import numpy as np
from PIL import Image
import io
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from typing import Tuple, Dict, Any
import base64

# --- Helper Function for Resizing ---
def get_resized_pixels(img: Image.Image, max_pixels: int = 10000) -> Tuple[np.ndarray, tuple]:
    """Downscales an image if it has more than max_pixels for faster processing."""
    w, h = img.size
    if w * h > max_pixels:
        aspect_ratio = w / h
        new_h = int((max_pixels / aspect_ratio) ** 0.5)
        new_w = int(new_h * aspect_ratio)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    pixels = np.array(img, dtype=np.float64) / 255.0
    return pixels.reshape(-1, 3), img.size


# --- Main K-Means Compression (high quality) ---
def compress_with_kmeans(img: Image.Image, k: int) -> Dict[str, Any]:
    pixels = np.array(img, dtype=np.float64) / 255.0
    original_shape = pixels.shape
    pixels_flat = pixels.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(pixels_flat)
    new_palette = kmeans.cluster_centers_
    new_pixels_flat = new_palette[kmeans.labels_]
    
    return _recreate_image(new_pixels_flat, original_shape, new_palette)


# --- Hierarchical Compression (on downscaled image) ---
def compress_with_hierarchical(img: Image.Image, k: int) -> Dict[str, Any]:
    pixels_flat, original_size = get_resized_pixels(img)
    original_shape = (original_size[1], original_size[0], 3)

    agg_clustering = AgglomerativeClustering(n_clusters=k).fit(pixels_flat)
    
    # To get the palette, we average the colors in each found cluster
    new_palette = np.array([pixels_flat[agg_clustering.labels_ == i].mean(axis=0) for i in range(k)])
    new_pixels_flat = new_palette[agg_clustering.labels_]
    
    return _recreate_image(new_pixels_flat, original_shape, new_palette)


# --- DBSCAN Compression (on downscaled image) ---
def compress_with_dbscan(img: Image.Image, eps: float = 0.1) -> Dict[str, Any]:
    pixels_flat, original_size = get_resized_pixels(img)
    original_shape = (original_size[1], original_size[0], 3)
    
    dbscan = DBSCAN(eps=eps, min_samples=4).fit(pixels_flat)
    labels = dbscan.labels_
    
    unique_labels = set(labels)
    # Outlier color is black
    outlier_color = np.array([0, 0, 0]) 
    
    # Calculate palette (centroids of found clusters)
    palette_list = []
    for label in unique_labels:
        if label != -1:
            palette_list.append(pixels_flat[labels == label].mean(axis=0))
    new_palette = np.array(palette_list) if palette_list else np.array([])
    
    # Recreate image, coloring outliers black
    new_pixels_flat = np.zeros_like(pixels_flat)
    label_to_palette_idx = {label: i for i, label in enumerate(unique_labels) if label != -1}
    
    for i, label in enumerate(labels):
        if label == -1:
            new_pixels_flat[i] = outlier_color
        else:
            new_pixels_flat[i] = new_palette[label_to_palette_idx[label]]
            
    return _recreate_image(new_pixels_flat, original_shape, new_palette)


# --- Utility to create image bytes from pixel data ---
def _recreate_image(pixels_flat: np.ndarray, shape: tuple, palette: np.ndarray) -> Dict[str, Any]:
    """Helper to convert pixel data back into an image and calculate stats."""
    new_pixels = pixels_flat.reshape(shape)
    new_img_array = (new_pixels * 255).astype(np.uint8)
    new_img = Image.fromarray(new_img_array)
    
    buffer = io.BytesIO()
    new_img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    

    palette_css = [
        f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
        for color in palette
    ] if palette.size > 0 else []

    # And we also need to fix the image_b64 encoding. Let's use the robust base64 library.
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
   

    return {
        "image_b64": f"data:image/png;base64,{encoded_string}",
        "size": len(image_bytes),
        "palette": palette_css,
        "num_colors": len(palette)
    }