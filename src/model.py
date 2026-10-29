"""
Pregunta 3: Arquitectura CNN en PyTorch, flujo de tensores 4D y validación de shapes.

Uso:
    python src/model.py
"""

import torch
import torch.nn as nn

SEED = 30
NUM_CLASSES = 10
INPUT_SHAPE = (3, 32, 32)  # (C, H, W) de CIFAR-10


def conv_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    """
    Bloque convolucional: (Conv2d -> BatchNorm2d -> ReLU) x 2 -> MaxPool2d -> Dropout.
    Las convoluciones 3x3 con padding=1 conservan H y W; MaxPool2d(2) los reduce a la mitad.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(dropout),
    )


class CIFAR10CNN(nn.Module):
    """
    CNN para CIFAR-10 con 3 bloques convolucionales y un clasificador denso.

        Entrada  (B,   3, 32, 32)
        Bloque 1 (B,  32, 16, 16)
        Bloque 2 (B,  64,  8,  8)
        Bloque 3 (B, 128,  4,  4)
        Flatten  (B, 2048)
        Linear   (B, 256) -> ReLU -> Dropout
        Linear   (B, 10)   logits
    """

    def __init__(self, num_classes: int = NUM_CLASSES, input_shape=INPUT_SHAPE):
        super().__init__()
        self.block1 = conv_block(input_shape[0], 32, dropout=0.10)
        self.block2 = conv_block(32, 64, dropout=0.20)
        self.block3 = conv_block(64, 128, dropout=0.30)
        self.flatten = nn.Flatten()  # (B, C, H, W) -> (B, C*H*W)

        # Cálculo dinámico del tamaño de entrada a la primera capa densa:
        # se pasa un tensor ficticio por los bloques convolucionales y se mide la salida.
        self.flatten_dim = self._get_flatten_dim(input_shape)

        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.block3(self.block2(self.block1(x)))

    @torch.no_grad()
    def _get_flatten_dim(self, input_shape) -> int:
        dummy = torch.zeros(1, *input_shape)
        return self.flatten(self._features(dummy)).shape[1]

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """Forward pass; con verbose=True imprime el shape después de cada etapa."""
        stages = [
            ("Entrada", lambda t: t),
            ("Bloque conv 1", self.block1),
            ("Bloque conv 2", self.block2),
            ("Bloque conv 3", self.block3),
            ("Flatten", self.flatten),
            ("Linear 1 + ReLU + Dropout", lambda t: self.dropout(self.relu(self.fc1(t)))),
            ("Linear 2 (logits)", self.fc2),
        ]
        for name, layer in stages:
            x = layer(x)
            if verbose:
                print(f"  {name:<27} -> {tuple(x.shape)}")
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate_shapes(model: nn.Module, batch_size: int = 8) -> torch.Tensor:
    """
    Prueba obligatoria: pasa un lote simulado 4D (B, C, H, W) por el modelo,
    imprime los shapes intermedios y verifica con aserciones que no hay Shape Mismatch.
    """
    model.eval()
    x = torch.randn(batch_size, *INPUT_SHAPE)
    assert x.dim() == 4, "La entrada debe ser un tensor 4D (B, C, H, W)"

    print(f"Validación del forward pass con un lote simulado de shape {tuple(x.shape)}")
    with torch.no_grad():
        f1 = model.block1(x)
        f2 = model.block2(f1)
        f3 = model.block3(f2)
        flat = model.flatten(f3)
        out = model(x, verbose=True)

    # Aserciones sobre cada etapa del flujo de tensores
    assert f1.shape == (batch_size, 32, 16, 16), f"Bloque 1 inesperado: {f1.shape}"
    assert f2.shape == (batch_size, 64, 8, 8), f"Bloque 2 inesperado: {f2.shape}"
    assert f3.shape == (batch_size, 128, 4, 4), f"Bloque 3 inesperado: {f3.shape}"
    assert flat.shape == (batch_size, model.flatten_dim), f"Flatten inesperado: {flat.shape}"
    assert model.fc1.in_features == flat.shape[1], "fc1.in_features no coincide con el Flatten"
    assert out.shape == (batch_size, NUM_CLASSES), f"Salida inesperada: {out.shape}"
    assert torch.isfinite(out).all(), "La salida contiene NaN/Inf"

    print(f"OK: la arquitectura procesa tensores 4D sin errores -> salida {tuple(out.shape)}")
    return out


if __name__ == "__main__":
    torch.manual_seed(SEED)
    model = CIFAR10CNN()
    print(model)
    print(f"\nTamaño dinámico de Flatten: {model.flatten_dim} (= 128 x 4 x 4)")
    print(f"Parámetros entrenables: {count_parameters(model):,}\n")
    validate_shapes(model, batch_size=8)
