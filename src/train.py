"""
Pregunta 4: Ciclo de entrenamiento, métricas de rendimiento y curvas de aprendizaje.

- Función de pérdida: CrossEntropyLoss
- Optimizador: Adam
- Artefactos: outputs/loss_accuracy.png y outputs/confusion_matrix.png

Uso:
    python src/train.py               # 30 épocas por defecto
    python src/train.py --epochs 10
"""

import argparse
import copy
import os
import time

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: solo guarda las figuras en disco
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset

from dataset import CLASSES, SEED, get_dataloaders, print_split_summary, set_seed
from model import CIFAR10CNN, count_parameters, validate_shapes

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# ---------------------------------------------------------------------------
# Hiperparámetros
# ---------------------------------------------------------------------------
EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Una época de entrenamiento. Devuelve (loss promedio, accuracy %)."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)            # (B, 10) logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds: bool = False):
    """Evalúa el modelo. Devuelve (loss, accuracy %) y opcionalmente (y_true, y_pred)."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_true, all_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if return_preds:
            all_true.append(labels.cpu())
            all_pred.append(preds.cpu())
    loss, acc = running_loss / total, 100.0 * correct / total
    if return_preds:
        return loss, acc, torch.cat(all_true).numpy(), torch.cat(all_pred).numpy()
    return loss, acc


def plot_learning_curves(history, best_epoch, save_path):
    """Genera outputs/loss_accuracy.png con las curvas de Loss y Accuracy (train vs val)."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(epochs, history["train_loss"], "o-", ms=3, label="Entrenamiento")
    ax1.plot(epochs, history["val_loss"], "s-", ms=3, label="Validación")
    ax1.set_title("Curva de pérdida (CrossEntropyLoss)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")

    ax2.plot(epochs, history["train_acc"], "o-", ms=3, label="Entrenamiento")
    ax2.plot(epochs, history["val_acc"], "s-", ms=3, label="Validación")
    ax2.set_title("Curva de exactitud (Accuracy)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy (%)")

    for ax in (ax1, ax2):
        ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Mejor época ({best_epoch})")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(f"Curvas de aprendizaje CNN - CIFAR-10 (semilla {SEED})", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, test_acc, save_path):
    """Genera outputs/confusion_matrix.png con conteos y porcentaje por clase real."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Proporción por clase real")

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title(f"Matriz de confusión - conjunto de prueba (10 000 imágenes)\n"
                 f"Accuracy = {test_acc:.2f}% | semilla {SEED}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Dispositivo: {device} | Semilla: {SEED}")

    # ---- Datos -----------------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE, NUM_WORKERS)
    print_split_summary(train_loader.dataset, val_loader.dataset, test_loader.dataset)

    # ---- Modelo ----------------------------------------------------------------
    model = CIFAR10CNN()
    validate_shapes(model, batch_size=8)
    model = model.to(device)
    print(f"Parámetros entrenables: {count_parameters(model):,}")

    # ---- Configuración del entrenamiento ----------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = 0.0, 0, None

    print(f"\nEntrenando {EPOCHS} épocas | batch={BATCH_SIZE} | Adam lr={LEARNING_RATE} wd={WEIGHT_DECAY}")
    print(f"{'Época':>5} | {'Train Loss':>10} {'Train Acc':>9} | {'Val Loss':>8} {'Val Acc':>7} | {'LR':>8} | {'t(s)':>5}")
    print("-" * 72)

    # ---- Bucle de entrenamiento por épocas --------------------------------------
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        print(f"{epoch + 1:>5} | {train_loss:>10.4f} {train_acc:>8.2f}% | {val_loss:>8.4f} {val_acc:>6.2f}% "
              f"| {current_lr:>8.2e} | {time.time() - start:>5.1f}", flush=True)

    # ---- Evaluación final sobre el conjunto de prueba independiente -------------
    model.load_state_dict(best_state)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, return_preds=True)

    # Las métricas de entrenamiento del bucle se miden con Data Augmentation y Dropout activos.
    # Para medir la brecha de generalización real se evalúan las mismas 45 000 imágenes
    # sin aumento y en modo eval.
    train_eval_set = Subset(val_loader.dataset.dataset, train_loader.dataset.indices)
    train_eval_loader = DataLoader(train_eval_set, batch_size=BATCH_SIZE, shuffle=False)
    train_eval_loss, train_eval_acc = evaluate(model, train_eval_loader, criterion, device)

    print("-" * 72)
    print(f"Mejor época (según validación): {best_epoch} | Val Acc = {best_val_acc:.2f}%")
    print(f"Última época: Train Acc = {history['train_acc'][-1]:.2f}% | Val Acc = {history['val_acc'][-1]:.2f}% "
          f"| brecha = {history['train_acc'][-1] - history['val_acc'][-1]:+.2f} pp")
    print(f"Última época: Train Loss = {history['train_loss'][-1]:.4f} | Val Loss = {history['val_loss'][-1]:.4f}")
    print(f"ENTRENAMIENTO (modo eval, sin aumento): Loss = {train_eval_loss:.4f} | Accuracy = {train_eval_acc:.2f}%")
    print(f"PRUEBA (test): Loss = {test_loss:.4f} | Accuracy = {test_acc:.2f}%")
    print(f"Brecha de generalización (train eval - test): {train_eval_acc - test_acc:+.2f} pp\n")
    print("Reporte de clasificación (test):")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    # ---- Artefactos gráficos ----------------------------------------------------
    curves_path = os.path.join(OUTPUT_DIR, "loss_accuracy.png")
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_learning_curves(history, best_epoch, curves_path)
    cm = plot_confusion_matrix(y_true, y_pred, test_acc, cm_path)

    # Pares de clases más confundidos
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    print("Confusiones más frecuentes (real -> predicha):")
    for flat_idx in np.argsort(off_diag, axis=None)[::-1][:5]:
        i, j = np.unravel_index(flat_idx, cm.shape)
        print(f"  {CLASSES[i]:<11} -> {CLASSES[j]:<11} {cm[i, j]:>4}")

    print(f"\nGuardado: {curves_path}")
    print(f"Guardado: {cm_path}")
    return history, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento CNN en CIFAR-10")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_WORKERS = args.epochs, args.batch_size, args.lr, args.num_workers
    main()
