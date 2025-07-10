from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage
import numpy as np
from typing import Dict, Any

def get_hierarchical_analysis(palette_rgb: np.ndarray) -> Dict[str, Any]:
    palette_norm = palette_rgb / 255.0
    Z = linkage(palette_norm, method='ward')
    
    return {"linkage_matrix": Z.tolist()}

def get_dbscan_analysis(palette_rgb: np.ndarray) -> Dict[str, Any]:
    palette_norm = palette_rgb / 255.0
    
    dbscan = DBSCAN(eps=0.25, min_samples=2).fit(palette_norm)
    labels = dbscan.labels_ 
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_outliers = np.sum(labels == -1)
    
    return {
        "labels": labels.tolist(),
        "num_clusters": num_clusters,
        "num_outliers": int(num_outliers),
        "eps": 0.25,
        "min_samples": 2
    }