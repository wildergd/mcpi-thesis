from joblib import dump as joblib_dump, load as joblib_load
from skops.io import dump as skops_dump, load as skops_load
from os import path

def get_file_path(filepath: str, ext: str = 'joblib') -> str:
    filename, _ = path.splitext(path.basename(filepath))
    return f'{path.basename(filepath)}/{filename}.{ext}'

def extract_format_from_filename(filepath: str) -> str:
    _, extension = path.splitext(path.basename(filepath))
    if extension in ['joblib', 'skops']:
        return extension
    raise IOError('Invalid model file.')

def persist_model(model, filename: str, format: str = 'joblib'):
    if format == 'skops':
        return skops_dump(model, get_file_path(filename, format))
    
    return joblib_dump(model, get_file_path(filename, format))

def load_model(filename: str):
    format = extract_format_from_filename(filename)
    if format == 'skops':
        return skops_load(filename, trusted = True)
    return joblib_load(filename)

__all__ = [
    'load_model',
    'persist_model'
]