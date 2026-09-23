import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


PAD_LABEL = -100
LABELS = ["o", "pi", "pc", "li", "lc"]


def leer_datos(path):
    """
    Cargamos los datos del fichero json
    Devuelve una lista (tokens, labels)
    """
    datos = json.loads(Path(path).read_text(encoding="utf-8"))
    ejemplos = []
    for fila in datos:
        if len(fila["tokens"]) != len(fila["labels"]):
            raise ValueError("Hay una frase con tokens y labels de distinto tamano")
        ejemplos.append((fila["tokens"], fila["labels"]))
    return ejemplos


def preparar_labels(ejemplos):
    """
    Devuelve un diccionario de las etiquetas y su índice para poder convertirlas a valores numéricos
    """
    labels = LABELS[:]
    for _, etiquetas in ejemplos:
        for etiqueta in etiquetas:
            if etiqueta not in labels:
                labels.append(etiqueta)
    return {label: i for i, label in enumerate(labels)}


def etiqueta_subtokens(etiqueta, n_subtokens):
    """
    Si una palabra tiene varios subtokens, reparte la etiqueta entre ellos:
    si "pi" tiene varios subtokens pasa a ser ["pi", "pc", "pc"]
    (el modelo tendrá una etiqueta para cada subtoken además de para cada palabra original)
    """
    if n_subtokens == 0:
        return []
    if etiqueta == "pi" and n_subtokens > 1:
        return ["pi"] + ["pc"] * (n_subtokens - 1)
    if etiqueta == "li" and n_subtokens > 1:
        return ["li"] + ["lc"] * (n_subtokens - 1)
    return [etiqueta] * n_subtokens


def tokenizar_ejemplo(tokens, labels, tokenizer, label_to_id, max_seq_len):
    """
    Conevrtimos un ejemplo de tokens y labels en IDs para el modelo
    Tokenizamos, adaptamos sus etiquetas a los subtokens y divide la seq en trozos de max_seq_len
    """
    ids = []
    out_labels = []

    for token, label in zip(tokens, labels):
        sub_ids = tokenizer.encode(token)  # tokemizamos
        sub_labels = etiqueta_subtokens(
            label, len(sub_ids)
        )  # adaptamos a los subtokens
        ids.extend(sub_ids)
        out_labels.extend(label_to_id[x] for x in sub_labels)

    trozos = []
    for i in range(0, len(ids), max_seq_len):
        x = ids[i : i + max_seq_len]
        y = out_labels[i : i + max_seq_len]
        if x:
            trozos.append((x, y))
    return trozos


class NerDataset(Dataset):
    """
    Clase para adaptar los ejemplos al formato para PyTorch
    Guardamos los pares de tokens y labels en las filas (self.filas), podemos decir el nº de ejemplos
    y entrenar el modelo con ellos
    """

    def __init__(self, ejemplos, tokenizer, label_to_id, max_seq_len):
        self.filas = []
        for tokens, labels in ejemplos:
            self.filas.extend(
                tokenizar_ejemplo(tokens, labels, tokenizer, label_to_id, max_seq_len)
            )

    def __len__(self):
        return len(self.filas)

    def __getitem__(self, idx):
        return self.filas[idx]


def collate(batch):
    """
    Une varios ejemplos en un batch, rellenando tokens y etiquetas hasta la misma longitud
    Las etiquetas de relleno usan PAD_LABEL para que no cuenten en la perdida
    """
    max_len = max(len(x) for x, _ in batch)
    xs = []
    ys = []

    for x, y in batch:
        relleno = max_len - len(x)
        xs.append(x + [0] * relleno)
        ys.append(y + [PAD_LABEL] * relleno)

    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)
