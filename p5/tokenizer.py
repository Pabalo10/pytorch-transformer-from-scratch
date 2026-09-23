# Tokenizador BPE (Byte Pair Encoding) mínimo, entrenado sobre el texto."""
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>


import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    """Byte Pair Encoding entrenado sobre un texto.

    Vocabulario inicial: caracteres unicos del texto. Durante el
    entrenamiento se buscan los pares adyacentes mas frecuentes y se
    fusionan en nuevos tokens, hasta alcanzar `vocab_size` tokens.

    NOTA: para ser BPE de verdad, tendríamos que hacerlo sobre bytes, no sobre
    caracteres, pero para la práctica funciona bien.
    """

    def __init__(self, text, vocab_size=300):
        self.vocab_size = vocab_size
        # Inicializamos con caracteres encontrados en el texto
        self.vocab = sorted(set(text))  # vocab[id] -> token string.
        self.tok2id = {
            tok: i for i, tok in enumerate(self.vocab)
        }  # diccionario {'a': 0; 'b':1} -> según aparecen en vocab

        tokens = [
            self.tok2id[c] for c in text
        ]  # array de los ids de los tokens/caracteres según aparecen en el texto de entrada
        self.merges = []  # lista de ((id_a, id_b), nuevo_id), para encode()

        for new_id in range(
            len(self.vocab), vocab_size
        ):  # nuevos ids desde len(vocab) hasta vocab_size
            pairs = Counter(
                zip(tokens, tokens[1:])
            )  # cuenta el número de veces que aparecen seguidos los pares de tokens
            best = pairs.most_common(1)[0][
                0
            ]  # devuelve el par de tokens más frecuente -> (id_a, id_b)

            new_tok = (
                self.vocab[best[0]] + self.vocab[best[1]]
            )  # une los dos tokens correspondientes a los id_a y id_b = "ab"
            self.tok2id[new_tok] = (
                new_id  # mete en tok2id el nuevo token y le asigna el id de new_id
            )
            self.vocab.append(new_tok)  # añade el token nuevo al vocabulario
            self.merges.append(
                (best, new_id)
            )  # ((id_a, id_b), new_id) -> la nueva combinacion de tokens ("ab") tiene como id: new_id

            tokens = self._apply_merge(tokens, best[0], best[1], new_id)

    @staticmethod
    def _apply_merge(tokens, a, b, new_id):
        """Reemplaza todas las ocurrencias del par (a, b) por new_id."""
        # TAREA: HACER

        result = []  # nuevo array de tokens
        i = 0
        while i < len(tokens):  # desde el principio hasta el final de tokens
            # si aún no hemos llegado al final del array y el token que leemos es "a" y el siguiente es "b" (combinación esperada)
            # añadimos ese token nuevo con ese nuevo id al array e incrementamos 2
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                result.append(new_id)
                i += 2
            else:  # sino, se añade el token en el que estemos
                result.append(tokens[i])
                i += 1

        return result

    def encode(self, text):
        """Codifica un texto aplicando los merges aprendidos."""
        tokens = [self.tok2id.get(c, 0) for c in text]
        for (a, b), new_id in self.merges:
            tokens = self._apply_merge(tokens, a, b, new_id)
        return tokens

    def decode(self, ids):
        """Decodifica una lista de ids a texto."""
        # TAREA: HACER
        return "".join(self.vocab[i] for i in ids)

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": [
                {"pair": [a, b], "new_id": new_id} for (a, b), new_id in self.merges
            ],
        }

    @classmethod
    def from_dict(cls, data):
        tokenizer = cls.__new__(cls)
        tokenizer.vocab_size = data["vocab_size"]
        tokenizer.vocab = data["vocab"]
        tokenizer.tok2id = {tok: i for i, tok in enumerate(tokenizer.vocab)}
        tokenizer.merges = [
            ((merge["pair"][0], merge["pair"][1]), merge["new_id"])
            for merge in data["merges"]
        ]
        return tokenizer

    def save_json(self, path):
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def __repr__(self):
        pretty = [t.replace("\n", "\\n").replace(" ", "▁") for t in self.vocab]
        return f"{len(self.vocab)} tokens: ['{"', '".join(pretty)}']"


# Si ejecutamos este módulo directamente, probamos el tokenizador
if __name__ == "__main__":
    import sys
    from pathlib import Path

    files_path = Path(sys.argv[1] if len(sys.argv) > 1 else "resources")
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    textos = "\n\n".join(open(p).read() for p in files_path.glob("*.txt"))
    tokenizer = BPETokenizer(textos, vocab_size=vocab_size)
    print(tokenizer)
