from pathlib import Path

import click
import torch

from generar_texto import generar
from ner.training import find_entities_file, train_ner
from p5.causal_llm import CausalLLM
from p5.causal_train import train
from p5.corpus import load_corpus
from p5.tokenizer import BPETokenizer

# Obtenemos la ruta actual para hacer llamadas a otros archivos (ej: corpus)
BASE_DIR = Path(__file__).resolve().parent


# Resuelve la ruta al archivo que queremos recuperar buscando en la ruta actual y sino en la
# relativa a BASE_DIR
def resolver_path(path):
    path = Path(path)
    if path.exists():
        return str(path)

    path_en_p5 = BASE_DIR / path
    if path_en_p5.exists():
        return str(path_en_p5)

    return str(path)


# Definimos el grupo de comandos con click
@click.group()
def main():
    pass


# Comando para entrenar el LLM Causal
@main.command("train-tokenizer")
def train_tokenizer():
    corpus = "resources/alice_in_wonderland.txt"
    text = load_corpus(resolver_path(corpus))
    tokenizer = BPETokenizer(text, vocab_size=300)
    tokenizer.save_json("mejor_tokenizador.json")
    print("tokenizador listo: mejor_tokenizador.json")


@main.command("train-causal")
def train_causal():
    corpus = "resources/alice_in_wonderland.txt"
    context_size = 64

    text = load_corpus(resolver_path(corpus))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BPETokenizer.load_json(resolver_path("mejor_tokenizador.json"))
    tokens = tokenizer.encode(text)

    model = CausalLLM(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=context_size,
        d_model=128,
        n_heads=4,
        n_layers=4,
        expansion=4,
        dropout=0.1,
    ).to(device)

    train(
        model,
        tokens,
        epochs=10,
        context_size=context_size,
        batch_size=64,
        lr=3e-4,
        patience=3,
    )

    torch.save(model.state_dict(), "p5_causal_2603.pth")

    print("modelo causal listo: p5_causal_2603.pth")


# Comando para generar texto con los pesos del Causal entrenados
# Recibe como parámetro el prompt que le introduzcamos
@main.command("generate")
@click.argument("prompt")
@click.option("--max-tokens", default=100)
@click.option("--temperature", default=0.8)
def generate(prompt, max_tokens, temperature):
    _, full_text = generar(
        prompt=prompt,
        modelo_path=resolver_path("p5_causal_2603.pth"),
        tokenizador_path=resolver_path("mejor_tokenizador.json"),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    print(full_text)


# Comando para entrenar le modelo NER
@main.command("train-ner")
def train_ner_cmd():
    train_ner(
        data_path=resolver_path("resources/merged_2.json"),
        output_path="p5_ner_2603.pth",
        causal_weights_path=resolver_path("p5_causal_2603.pth"),
        tokenizer_path=resolver_path("mejor_tokenizador.json"),
    )
    print("modelo NER guardado en p5_ner_2603.pth")


# Comando para ejecutar el modelo NER
# Recibe como parámetro la ruta al archivo de texto del que queramos que reconozca las entidades
@main.command("ner")
@click.argument("text_file")
def ner_cmd(text_file):
    entidades = find_entities_file(
        resolver_path(text_file),
        resolver_path("p5_ner_2603.pth"),
    )
    if not entidades:
        print("No se han encontrado entidades.")
        return

    for text, label in entidades:
        print(f"{label}\t{text}")


if __name__ == "__main__":
    main()
