from pathlib import Path

import torch

from p5.causal_llm import CausalLLM
from p5.tokenizer import BPETokenizer


# Cargamos los pesos guardados del modelo Causal desde el path
def cargar_state_dict(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    # Si el archivo es un chekpoint con clave "model_estate" devuelvo eso,, sino devuelve lo que ha cargado
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    return checkpoint


# Recuepramos la configuración del modelo en base a las formas de los pesos
def inferir_config(state_dict, n_heads=4):
    vocab_size, d_model = state_dict["tok_emb.weight"].shape
    max_seq_len = state_dict["pos_emb.weight"].shape[0]

    capas = {
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("blocks.") and k.split(".")[1].isdigit()
    }
    n_layers = max(capas) + 1 if capas else 0

    hidden_ff = state_dict["blocks.0.ff.up.weight"].shape[0]
    expansion = hidden_ff // d_model

    # Devolvemos el diccionario con la configuración
    return {
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "expansion": expansion,
        "dropout": 0.0,
    }


def generar(
    prompt,
    modelo_path,
    tokenizador_path,
    max_tokens=100,
    temperature=0.8,
    n_heads=4,
    seed=None,
    device=None,
):
    if seed is not None:
        torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargamos el Tokenizador
    tokenizer = BPETokenizer.load_json(tokenizador_path)

    # 2. Cargamos los pesos e inferir_config
    state_dict = cargar_state_dict(modelo_path)
    config = inferir_config(state_dict, n_heads=n_heads)

    # 3. Preparamos el Modelo
    model = CausalLLM(**config).to(
        device
    )  # así cargamos todos los argumentos que espera del diccionario
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Procesamos el prompt
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("El prompt no ha producido ningun token.")

    # 5. Generamos usando el método interno de la clase
    generated_ids = model.generate(
        prompt_ids, max_tokens=max_tokens, temperature=temperature
    )

    # 6. Decodificamos
    generated_text = tokenizer.decode(generated_ids)

    return generated_text, prompt + generated_text
