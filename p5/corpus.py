# p5/corpus.py

from pathlib import Path


def load_corpus(corpus_path):
    """
    Carga y concatena todos los archivos .txt de un directorio,
    o lee un archivo único si se le pasa la ruta directa.
    """
    path = Path(corpus_path)

    # Si le pasamos un archivo directo (ej: resources/alice_in_wonderland.txt)
    if path.is_file():
        return path.read_text(encoding="utf-8")

    # Si le pasamos una carpeta (ej: resources)
    elif path.is_dir():
        textos = []
        for p in path.glob("*.txt"):
            textos.append(p.read_text(encoding="utf-8"))
        return "\n\n".join(textos)

    else:
        raise ValueError(f"No se encontró la ruta o archivo: {corpus_path}")
