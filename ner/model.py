import torch
import torch.nn as nn
import torch.nn.functional as F

from ner.data import PAD_LABEL
from p5.transformer import Transformer


class NerModel(nn.Module):
    """
    Modelo NER (Named Entity Recongition) para reconocer entidades nombradas en un texto
    Basado en el modelo Causal entrenado previamente y añadido como capa adicional desdpués de este
    para realizar la tarea de reconocimiento y clasificación de entidades.
    """

    def __init__(
        self,
        vocab_size,
        n_labels,
        max_seq_len,
        d_model,
        n_heads,
        n_layers,
        expansion,
        dropout,
        classifier_hidden=0,
        class_weights=None,
    ):
        super().__init__()
        self.transformer = Transformer(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            expansion=expansion,
            dropout=dropout,
        )

        if classifier_hidden > 0:
            # Red neuronal con capa oculta
            self.clasificador = nn.Sequential(
                nn.Linear(d_model, classifier_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, n_labels),
            )
        else:
            # capa lineal para casificador si no se especifica classifier_hidden220
            self.clasificador = nn.Linear(d_model, n_labels)

        if class_weights is None:
            class_weights = torch.ones(n_labels)
        self.register_buffer("class_weights", class_weights)

    def forward(self, x, y=None):
        """Feed Forward del modelo NER"""
        h = self.transformer(x, causal=False)
        logits = self.clasificador(h)

        if y is None:
            return logits, None

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            ignore_index=PAD_LABEL,
            weight=self.class_weights,
        )
        return logits, loss
