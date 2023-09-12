import joblib
from skops import io as sio
from os import path, makedirs

def get_file_path(filepath: str, ext: str = 'joblib') -> str:
    filename, _ = path.splitext(path.basename(filepath))
    return f'{path.dirname(filepath)}/{filename}.{ext}'

def extract_format_from_filename(filepath: str) -> str:
    _, extension = path.splitext(path.basename(filepath))
    if extension in ['.joblib', '.skops']:
        return extension[1:]
    raise IOError('Invalid model file.')

def persist_model(model, filename: str):
    format = extract_format_from_filename(filename)
    if format == 'skops':
        return sio.dump(model, filename)
    
    return joblib.dump(model, get_file_path(filename, format))

def load_model(filename: str):
    format = extract_format_from_filename(filename)
    if format == 'skops':
        return sio.load(filename, trusted = True)
    return joblib.load(filename)


def save_model(model, filename: str):
    folder = path.dirname(filename)
    if not path.exists(folder):
        makedirs(folder)
    
    persist_model(model, filename)
    

__all__ = [
    'load_model',
    'save_model'
]