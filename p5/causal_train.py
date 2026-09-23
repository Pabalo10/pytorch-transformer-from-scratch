# Entrenamiento de LLM causal en base a un corpus
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>

import time
import sys
from pathlib import Path
import itertools
import os

# Truco para poder ejecutar directamente con 'uv run python p5/causal_train.py resources'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Ventana deslizante sobre un tensor de tokens para language modeling."""

    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def _make_dataloaders(tokens, context_size, batch_size, train_ratio=0.9):
    """Los dataloaders se encargan de ir aportando pares para el entrenamiento."""
    data = torch.tensor(tokens, dtype=torch.long)

    split = int(train_ratio * len(data))
    train_ds = TextDataset(data[:split], context_size)
    val_ds = TextDataset(data[split:], context_size)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
    )


def _run_epoch(model, dataloader, optimizer=None):
    """Ejecuta una epoch completa de entrenamiento o evaluación."""
    total_loss, n = 0, 0
    device = next(model.parameters()).device

    if optimizer:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        if optimizer:
            optimizer.zero_grad()

        _, loss = model(x, y)

        if optimizer:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / n


def train(
    model,
    tokens,
    epochs=5,
    context_size=128,
    batch_size=64,
    lr=3e-4,
    train_ratio=0.9,
    patience=0,  # MODIFICACIÓN: 0 = desactivado. Mayor que 0 = Early Stopping activado.
):
    """Entrena el modelo de lenguaje causal sobre los tokens dados."""
    train_dl, val_dl = _make_dataloaders(tokens, context_size, batch_size, train_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    t0 = time.time()
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_dl, optimizer)
        val_loss = _run_epoch(model, val_dl, None)
        elapsed = time.time() - t0

        logger.info(
            f"Epoca {epoch + 1}/{epochs} | train={train_loss:.4f} | "
            f"val={val_loss:.4f} | tiempo={elapsed:.1f}s"
        )

        # Lógica de Early Stopping (solo si patience > 0)
        if patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "temp_early_stop.pt")
            else:
                patience_counter += 1
                logger.info(
                    f"  ->  Sin mejora. Paciencia: {patience_counter}/{patience}"
                )
                if patience_counter >= patience:
                    logger.warning(
                        f" EARLY STOPPING en la época {epoch + 1}. Restaurando mejor época..."
                    )
                    model.load_state_dict(
                        torch.load("temp_early_stop.pt", weights_only=True)
                    )
                    break

    elapsed = time.time() - t0
    logger.info(f"Entrenamiento finalizado en {elapsed:.1f}s")

    # Si usamos early stopping devolvemos el mejor loss, si no, el último.
    return best_val_loss if patience > 0 and best_val_loss != float("inf") else val_loss


if __name__ == "__main__":
    from p5.causal_llm import CausalLLM
    from p5.corpus import load_corpus
    from p5.tokenizer import BPETokenizer

    corpus = sys.argv[1] if len(sys.argv) > 1 else "resources"
    text = load_corpus(corpus)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    VOCAB_SIZE = 300
    CONTEXT_SIZE = 64

    tokenizer = BPETokenizer.load_json("mejor_tokenizador.json")
    tokens = tokenizer.encode(text)

    # =================================================================
    # FASE 1: EXPLORACIÓN DE HIPERPARÁMETROS (GRID SEARCH)
    # =================================================================

    dimensiones = [64, 128, 256]
    capas = [2, 4, 6]
    learning_rates = [3e-4, 1e-3, 0.01]

    mejor_loss = float("inf")
    mejor_config = None

    logger.info("INICIANDO FASE 1: EXPLORACIÓN DE HIPERPARÁMETROS...")

    for d, l, lr in itertools.product(dimensiones, capas, learning_rates):
        logger.info("=" * 50)
        logger.info(f"PROBANDO: d_model={d} | n_layers={l} | lr={lr}")
        logger.info("=" * 50)

        model = CausalLLM(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=CONTEXT_SIZE,
            d_model=d,
            n_heads=4,
            n_layers=l,
            expansion=4,
            dropout=0.1,
        ).to(device)

        # Entrenamos solo 2 épocas, con patience=0 (sin early stopping)
        val_loss_final = train(
            model, tokens, epochs=2, context_size=CONTEXT_SIZE, lr=lr, patience=0
        )

        if val_loss_final < mejor_loss:
            mejor_loss = val_loss_final
            mejor_config = (d, l, lr)
            logger.info(
                f" ¡Nueva mejor configuración encontrada! Loss: {mejor_loss:.4f}"
            )

    logger.info("=" * 50)
    logger.info(
        f"BÚSQUEDA TERMINADA. La mejor configuración fue: d_model={mejor_config[0]} | n_layers={mejor_config[1]} | lr={mejor_config[2]}"
    )

    # =================================================================
    # FASE 2: ENTRENAMIENTO PROFUNDO DEL MEJOR MODELO
    # =================================================================

    logger.info("=" * 50)
    logger.info("INICIANDO FASE 2: ENTRENAMIENTO PROFUNDO CON EARLY STOPPING")
    logger.info("=" * 50)

    # 1. Instanciamos un modelo completamente nuevo y limpio con los parámetros ganadores
    d_opt, l_opt, lr_opt = mejor_config

    modelo_definitivo = CausalLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=CONTEXT_SIZE,
        d_model=d_opt,
        n_heads=4,
        n_layers=l_opt,
        expansion=4,
        dropout=0.1,
    ).to(device)

    # 2. Guardamos el tokenizador que usará nuestro modelo definitivo
    tokenizer.save_json("mejor_tokenizador.json")

    # 3. Entrenamos 10 épocas, pero esta vez con patience=3
    train(
        modelo_definitivo,
        tokens,
        epochs=10,
        context_size=CONTEXT_SIZE,
        lr=lr_opt,
        patience=3,  # ¡Early Stopping Activado!
    )

    # 4. Guardamos el modelo definitivo
    torch.save(modelo_definitivo.state_dict(), "mejor_modelo_definitivo.pt")
    logger.info(
        " Pipeline completado. Archivos 'mejor_modelo_definitivo.pt' y 'mejor_tokenizador.json' generados con éxito."
    )

    # Limpieza del archivo temporal
    if os.path.exists("temp_early_stop.pt"):
        os.remove("temp_early_stop.pt")
