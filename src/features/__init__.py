# Features module for data loading and feature engineering
# 
# Lazy imports to avoid requiring all dependencies when running individual scripts.
# For example, download_data.py only needs 'requests', not 'cv2'.

def __getattr__(name):
    """Lazy import of submodules."""
    if name == 'Comma2k19Loader':
        from .data_loader import Comma2k19Loader
        return Comma2k19Loader
    elif name == 'DataSynchronizer':
        from .synchronizer import DataSynchronizer
        return DataSynchronizer
    elif name == 'FeatureExtractor':
        from .feature_extractor import FeatureExtractor
        return FeatureExtractor
    elif name == 'DrivingLabeler':
        from .labeler import DrivingLabeler
        return DrivingLabeler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['Comma2k19Loader', 'DataSynchronizer', 'FeatureExtractor', 'DrivingLabeler']
