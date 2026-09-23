import random
from itertools import product
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generar_texto import cargar_state_dict, inferir_config
from ner.data import (
    PAD_LABEL,
    NerDataset,
    collate,
    leer_datos,
    preparar_labels,
    tokenizar_ejemplo,
)
from ner.model import NerModel
from p5.tokenizer import BPETokenizer


# Cargamos el mejor tokenizador, guardado del entrenamiento del causalLLM
def cargar_tokenizador(path):
    return BPETokenizer.load_json(path)


# Creamos el modelo NER
def crear_modelo_ner(
    causal_state,
    n_labels,
    dropout,
    classifier_hidden=0,
    class_weights=None,
):
    """Creamos nuestro modelo NER con sus hiperparámetros y basado en el Causal"""
    config = inferir_config(
        causal_state
    )  # cargamos la configuracion del Causal recuperada anteriormente
    model = NerModel(
        vocab_size=config["vocab_size"],
        n_labels=n_labels,
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        expansion=config["expansion"],
        dropout=dropout,
        classifier_hidden=classifier_hidden,
        class_weights=class_weights,
    )  # inicializamos el modelo NER con sus hiperparametros

    pesos_transformer = {
        f"transformer.{name}": value
        for name, value in causal_state.items()
        if not name.startswith("lm_head.")
    }
    model.load_state_dict(pesos_transformer, strict=False)
    return model, config


# Calculamos pesos de las calses de los labels
def calcular_pesos_clases(ejemplos, tokenizer, label_to_id, max_seq_len, device):
    """
    Calculamos los pesos para cada una de las clases (etiquetas) segun cuanto aparezcan
    Las etiquetas mas raras reciben mas peso y las mas abundantes menos
    Asi el modelo no aprendera a predecir la calse mas dominante
    Limitamos el peso de "o" a 0.5 porque la abuancia de esta es tremendamente mayor al resto
    """
    counts = torch.ones(len(label_to_id), dtype=torch.float)

    for tokens, labels in ejemplos:
        partes = tokenizar_ejemplo(tokens, labels, tokenizer, label_to_id, max_seq_len)
        for _, y in partes:
            for label_id in y:
                counts[label_id] += 1

    pesos = counts.sum() / counts
    pesos = pesos / pesos.mean()
    pesos[label_to_id["o"]] = min(
        pesos[label_to_id["o"]], 0.5
    )  # limitamos valor de la clase "o"
    return pesos.to(device)


# Division en train val y test
def partir_datos(ejemplos, seed=13):
    """Partimos los datos en train, val y test (0.7, 0.15 y 0.15)"""
    ejemplos = ejemplos[:]
    random.Random(seed).shuffle(ejemplos)

    n = len(ejemplos)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    train = ejemplos[:n_train]
    val = ejemplos[n_train : n_train + n_val]
    test = ejemplos[n_train + n_val :]
    return train, val, test


# Creamos Dataset y Loader
def crear_loader(ejemplos, tokenizer, label_to_id, max_seq_len, batch_size, shuffle):
    """Creamos el DataLoader de Pytorhc para entrenar y evaluar el modelo"""
    dataset = NerDataset(ejemplos, tokenizer, label_to_id, max_seq_len)
    # El loader agrupa ejemplos en batches que se pueden mezclar y rellenar secuencias
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate
    )


# Entrenamos las epocas del modelo una a una
def entrenar_epochs(model, loader, optim, epochs, device):
    for epoch in range(1, epochs + 1):
        total = 0
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optim.step()

            total += loss.item()

        print(f"  epoca {epoch}/{epochs} loss={total / max(1, len(loader)):.4f}")


# Evaluamos el rendimiento
def evaluar(model, loader, device):
    """
    Evaluamos el rendimiento del modelo en validacion o test
    Calculamso el loss, el accuracy total y el accuracy de la clasificacion de clases que no son "o"
    """
    model.eval()
    total_loss = 0
    bien = 0
    total = 0
    bien_entidad = 0
    total_entidad = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits, loss = model(x, y)

            # Nos quedamos con la label mas probable para cada token
            pred = logits.argmax(dim=-1)

            # mascaras para ignorar las posiciones de padding y dejar solo tokens de entidad (!="o")
            mask = y != PAD_LABEL
            mask_entidad = mask & (y != 0)

            # aciertos en tokens reales sin el padding
            bien += ((pred == y) & mask).sum().item()
            total += mask.sum().item()  # total tokens reales

            # acierto en tokens que son entidades
            bien_entidad += ((pred == y) & mask_entidad).sum().item()
            total_entidad += mask_entidad.sum().item()

            # sumamos el loss del batch
            total_loss += loss.item()

    return {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": bien / max(1, total),
        "entity_accuracy": bien_entidad / max(1, total_entidad),
    }


# Probamos distintas configs para encontrar la mejor
def probar_config(
    nombre,
    params,
    causal_state,
    label_to_id,
    tokenizer,
    train_ejemplos,
    val_ejemplos,
    max_seq_len,
    device,
):
    """Probamos a entrenar cada una de las posibles configuraciones para encontrar el mejor modelo"""
    print(f"probando {nombre}: {params}")
    class_weights = calcular_pesos_clases(
        train_ejemplos,
        tokenizer,
        label_to_id,
        max_seq_len,
        device,
    )

    model, _ = crear_modelo_ner(
        causal_state,
        len(label_to_id),
        dropout=params["dropout"],
        classifier_hidden=params["classifier_hidden"],
        class_weights=class_weights,
    )
    model = model.to(device)

    train_loader = crear_loader(
        train_ejemplos,
        tokenizer,
        label_to_id,
        max_seq_len,
        params["batch_size"],
        True,
    )

    val_loader = crear_loader(
        val_ejemplos,
        tokenizer,
        label_to_id,
        max_seq_len,
        params["batch_size"],
        False,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    entrenar_epochs(model, train_loader, optim, params["epochs"], device)
    val = evaluar(model, val_loader, device)
    print(
        f"  validacion loss={val['loss']:.4f} "
        f"acc={val['accuracy']:.4f} ent={val['entity_accuracy']:.4f}"
    )
    return val


# Valores de parametros para entrenar el modelo
def grid_candidatos():
    """Hacemos un grid con los posibles valores de los parametros con los que entrenaremos el modelo"""
    epochs_opciones = [4, 8]
    lr_opciones = [3e-4, 1e-4]
    dropout_opciones = [0.1, 0.2]
    hidden_opciones = [0, 64]

    for epochs, lr, dropout, hidden in product(
        epochs_opciones,
        lr_opciones,
        dropout_opciones,
        hidden_opciones,
    ):
        # devuelve una configuracion cada vez
        yield {
            "epochs": epochs,
            "lr": lr,
            "dropout": dropout,
            "classifier_hidden": hidden,
            "batch_size": 16,
        }


# Entrenamos el NER
def train_ner(
    data_path,
    output_path,
    causal_weights_path="p5_causal_2603.pth",
    tokenizer_path="mejor_tokenizador.json",
    resumen_path="ner_resultados.txt",
    device=None,
):
    """
    Entrenamiento del modelo NER

    Cargamos los datos con los que entrenaremos, tokenizador y pesos de modelo causal
    Dividimos los datos y probamos las distintas configs de modelo para el entrenamiento - guardamos la mejor
    Entrenamos profundamente el modelo final con los hiperparámetros elegidos
    Evaluamos en test y guardamos la configuracion
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ejemplos = leer_datos(data_path)
    tokenizer = cargar_tokenizador(tokenizer_path)
    label_to_id = preparar_labels(ejemplos)

    causal_state = cargar_state_dict(causal_weights_path)
    config = inferir_config(causal_state)

    if len(tokenizer.vocab) != config["vocab_size"]:
        raise ValueError("El tokenizador no coincide con los pesos del modelo causal")

    train_ejemplos, val_ejemplos, test_ejemplos = partir_datos(ejemplos)
    candidatos = list(grid_candidatos())

    resultados = []
    for i, params in enumerate(candidatos, start=1):
        nombre = f"config_{i}"
        val = probar_config(
            nombre,
            params,
            causal_state,
            label_to_id,
            tokenizer,
            train_ejemplos,
            val_ejemplos,
            config["max_seq_len"],
            device,
        )
        resultados.append((nombre, params, val))

    mejor_nombre, mejor_params, _ = min(resultados, key=lambda item: item[2]["loss"])
    print(f"mejor configuracion: {mejor_nombre} {mejor_params}")

    train_val = train_ejemplos + val_ejemplos
    model, _ = crear_modelo_ner(
        causal_state,
        len(label_to_id),
        dropout=mejor_params["dropout"],
        classifier_hidden=mejor_params["classifier_hidden"],
        class_weights=calcular_pesos_clases(
            train_val,
            tokenizer,
            label_to_id,
            config["max_seq_len"],
            device,
        ),
    )
    model = model.to(device)

    train_loader = crear_loader(
        train_val,
        tokenizer,
        label_to_id,
        config["max_seq_len"],
        mejor_params["batch_size"],
        True,
    )

    test_loader = crear_loader(
        test_ejemplos,
        tokenizer,
        label_to_id,
        config["max_seq_len"],
        mejor_params["batch_size"],
        False,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=mejor_params["lr"])

    print("entrenando modelo final con train+validacion")
    entrenar_epochs(model, train_loader, optim, mejor_params["epochs"], device)
    test = evaluar(model, test_loader, device)
    print(
        f"test loss={test['loss']:.4f} "
        f"acc={test['accuracy']:.4f} ent={test['entity_accuracy']:.4f}"
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "label_to_id": label_to_id,
            "config": {
                **config,
                "classifier_hidden": mejor_params["classifier_hidden"],
            },
        },
        output_path,
    )


# Cargamos NER desde el archivo
def cargar_modelo(path, device=None):
    """Cargamos el modelo NER preentrenado desde el archivo - se usará a la hora de predecir"""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    tokenizer = BPETokenizer.from_dict(ckpt["tokenizer"])
    label_to_id = ckpt["label_to_id"]
    id_to_label = {v: k for k, v in label_to_id.items()}
    config = ckpt["config"]

    # Reconstruimos el modelo
    model = NerModel(
        vocab_size=config["vocab_size"],
        n_labels=len(label_to_id),
        max_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        expansion=config["expansion"],
        dropout=0.0,
        classifier_hidden=config.get("classifier_hidden", 0),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, tokenizer, id_to_label, config, device


# Predecimos usando los pesos
def predecir(texto, weights_path, device=None):
    """Cargamos los pesos del modelo NER preentrenado para hacer las predicciones"""
    model, tokenizer, id_to_label, config, device = cargar_modelo(weights_path, device)
    ids = tokenizer.encode(texto)
    pares = []

    with torch.no_grad():
        for i in range(0, len(ids), config["max_seq_len"]):
            trozo = ids[i : i + config["max_seq_len"]]
            x = torch.tensor([trozo], dtype=torch.long, device=device)
            logits, _ = model(x)
            pred = logits.argmax(dim=-1).squeeze(0).tolist()

            for token_id, label_id in zip(trozo, pred):
                pares.append((tokenizer.decode([token_id]), id_to_label[label_id]))

    return pares


# Juntamos partes de entidades como etiqueta inicial
def juntar_entidades(pares):
    """Agrupamos tokens consecutivos que el modelo ha detectado como parte de una entidad

    Devuelve entidades completas
    Ejemplo:
        [("Humpty", "pi"), (" ", "pc"), (" Dumpty", "pc")] -> [("Humpty Dumpty": "pi")]
    """
    entidades = []
    actual = []
    tipo = None

    for texto, label in pares:
        # ignoramos las etiquetas "o"
        if label == "o":
            if actual:
                entidades.append(("".join(actual).strip(), tipo))
                actual = []
                tipo = None
            continue

        if label in ["pi", "li"] or not actual:
            if actual:
                entidades.append(("".join(actual).strip(), tipo))
            actual = [texto]
            tipo = label
        else:
            actual.append(texto)

    if actual:
        entidades.append(("".join(actual).strip(), tipo))

    return [(texto, label) for texto, label in entidades if texto]


# Buscamos las entidades en el texto
def find_entities_file(text_path, weights_path, device=None):
    """Leemos texto del path, predecimos usando el modelo guardado en weights_path y devolvemos las entidades encontradas"""
    texto = Path(text_path).read_text(encoding="utf-8")
    return juntar_entidades(predecir(texto, weights_path, device))
