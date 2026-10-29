"""
Pregunta 1: Carga, visualización, transformaciones y partición estratificada de CIFAR-10.

Semilla individual: 30 (dos últimos dígitos del código de matrícula).

Uso:
    python src/dataset.py
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
SEED = 30
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
VAL_SIZE = 0.10  # 10 % del set de entrenamiento -> 5 000 imágenes de validación

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Media y desviación estándar por canal (R, G, B) del set de entrenamiento de CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int = SEED) -> None:
    """Fija todas las fuentes de aleatoriedad para garantizar reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Semilla para cada proceso del DataLoader (aumentos de datos reproducibles)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Pipeline de transformaciones (torchvision.transforms)
# ---------------------------------------------------------------------------
def get_transforms():
    """
    Devuelve (train_transform, eval_transform).

    - Entrenamiento: Data Augmentation (RandomCrop con padding + RandomHorizontalFlip),
      conversión a tensor y normalización de los canales RGB.
    - Validación / prueba: solo conversión a tensor y normalización (sin aumento).
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_transform, eval_transform


# ---------------------------------------------------------------------------
# Carga y partición
# ---------------------------------------------------------------------------
def load_cifar10(data_dir: str = DATA_DIR):
    """Descarga (si no existe) y carga CIFAR-10 con las transformaciones correspondientes."""
    train_transform, eval_transform = get_transforms()

    # Se cargan dos vistas del mismo set de entrenamiento: una con aumento (para train)
    # y otra sin aumento (para validación), de modo que la validación no se vea alterada.
    train_aug = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    train_clean = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=eval_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)
    return train_aug, train_clean, test_set


def stratified_split(targets, val_size: float = VAL_SIZE, seed: int = SEED):
    """Partición estratificada de índices: conserva la proporción de cada clase."""
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=targets,
        shuffle=True,
        random_state=seed,
    )
    return train_idx, val_idx


def get_datasets(data_dir: str = DATA_DIR, val_size: float = VAL_SIZE, seed: int = SEED):
    """Devuelve (train_subset, val_subset, test_set) ya particionados."""
    train_aug, train_clean, test_set = load_cifar10(data_dir)
    train_idx, val_idx = stratified_split(train_aug.targets, val_size, seed)
    train_subset = Subset(train_aug, train_idx)
    val_subset = Subset(train_clean, val_idx)
    return train_subset, val_subset, test_set


def get_dataloaders(batch_size: int = 128, num_workers: int = 2, seed: int = SEED,
                    data_dir: str = DATA_DIR):
    """Crea los DataLoaders de entrenamiento, validación y prueba con mezcla reproducible."""
    train_subset, val_subset, test_set = get_datasets(data_dir, seed=seed)

    g = torch.Generator()
    g.manual_seed(seed)  # determina el orden de la mezcla (shuffle) en cada época

    common = dict(num_workers=num_workers, worker_init_fn=seed_worker,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              generator=g, **common)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def class_distribution(subset_or_dataset):
    """Cuenta cuántas imágenes hay por clase en un Dataset o Subset."""
    if isinstance(subset_or_dataset, Subset):
        targets = np.array(subset_or_dataset.dataset.targets)[subset_or_dataset.indices]
    else:
        targets = np.array(subset_or_dataset.targets)
    return np.bincount(targets, minlength=len(CLASSES))


def print_split_summary(train_subset, val_subset, test_set, train_loader=None) -> None:
    """Imprime en consola las dimensiones resultantes de la partición."""
    print("=" * 64)
    print(f"Partición estratificada de CIFAR-10 (semilla = {SEED})")
    print("=" * 64)
    for name, ds in (("Entrenamiento", train_subset), ("Validación", val_subset), ("Prueba", test_set)):
        img, _ = ds[0]
        print(f"{name:<14}: {len(ds):>6} imágenes | shape por imagen: {tuple(img.shape)}")
    print("-" * 64)
    print("Distribución por clase (train / val / test):")
    dtr, dva, dte = class_distribution(train_subset), class_distribution(val_subset), class_distribution(test_set)
    for i, c in enumerate(CLASSES):
        print(f"  {i} {c:<11} {dtr[i]:>5} / {dva[i]:>4} / {dte[i]:>4}")
    if train_loader is not None:
        images, labels = next(iter(train_loader))
        print("-" * 64)
        print(f"Lote de entrenamiento (B, C, H, W): {tuple(images.shape)} | etiquetas: {tuple(labels.shape)}")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------
def denormalize(img: torch.Tensor) -> torch.Tensor:
    """Revierte la normalización para poder graficar la imagen."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def show_samples(dataset, n: int = 16, cols: int = 4, seed: int = SEED, title=None,
                 save_path=None, show: bool = True):
    """
    Grafica una cuadrícula de `n` imágenes (8 a 16 recomendadas) con su etiqueta real.
    Acepta Dataset o Subset cuyas imágenes ya sean tensores normalizados.
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5), layout="constrained")
    for ax, idx in zip(axes.flat, idxs):
        img, label = dataset[int(idx)]
        ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
        ax.set_title(f"{CLASSES[label]} ({label})", fontsize=10)
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title or f"CIFAR-10: {n} imágenes de ejemplo con su clase real (semilla {seed})")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    set_seed(SEED)
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    train_subset, val_subset, test_set = train_loader.dataset, val_loader.dataset, test_loader.dataset

    print_split_summary(train_subset, val_subset, test_set, train_loader)

    # Cuadrícula de 16 imágenes con su etiqueta real (sin aumento de datos)
    show_samples(val_subset, n=16)
