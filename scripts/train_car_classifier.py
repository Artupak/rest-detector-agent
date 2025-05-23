import click
import tensorflow as tf
import os
from pathlib import Path
import requests
import tarfile
from tqdm import tqdm
from src.car_classifier import CarClassifier

STANFORD_CARS_URL = "http://ai.stanford.edu/~jkrause/car196/car_ims.tgz"
ANNOTATIONS_URL = "http://ai.stanford.edu/~jkrause/car196/cars_annos.mat"

def download_file(url: str, target_path: Path, desc: str):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def prepare_dataset(dataset_path: Path):
    """Prepare Stanford Cars Dataset for training."""
    # Create directories
    train_dir = dataset_path / 'train'
    val_dir = dataset_path / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset
    if not (dataset_path / 'car_ims.tgz').exists():
        download_file(
            STANFORD_CARS_URL,
            dataset_path / 'car_ims.tgz',
            'Downloading Stanford Cars Dataset'
        )
    
    # Download annotations
    if not (dataset_path / 'cars_annos.mat').exists():
        download_file(
            ANNOTATIONS_URL,
            dataset_path / 'cars_annos.mat',
            'Downloading annotations'
        )
    
    # Extract dataset
    if not (dataset_path / 'car_ims').exists():
        with tarfile.open(dataset_path / 'car_ims.tgz') as tar:
            tar.extractall(path=dataset_path)
    
    # Process annotations and split dataset
    annotations = tf.io.read_file(str(dataset_path / 'cars_annos.mat'))
    # TODO: Process annotations and organize images into train/val splits
    # This requires additional processing of the .mat file
    
    return train_dir, val_dir

@click.command()
@click.option('--dataset-path', type=click.Path(), default='data/stanford_cars',
              help='Path to store/load Stanford Cars Dataset')
@click.option('--epochs', type=int, default=10,
              help='Number of training epochs')
def main(dataset_path, epochs):
    """Fine-tune car classifier on Stanford Cars Dataset."""
    dataset_path = Path(dataset_path)
    
    # Prepare dataset
    click.echo("Preparing dataset...")
    train_dir, val_dir = prepare_dataset(dataset_path)
    
    # Initialize and fine-tune model
    click.echo("Fine-tuning model...")
    classifier = CarClassifier()
    classifier.fine_tune(dataset_path, epochs=epochs)
    
    click.echo("Done! Fine-tuned model saved.")

if __name__ == '__main__':
    main() 