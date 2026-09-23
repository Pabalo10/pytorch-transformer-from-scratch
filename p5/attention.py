"""
import torch.nn
import torch.nn.functional as F
import math

class Attention(nn.Module): # el modelo es la clase

    def __init__(self, d_model, n_tokens, n_heads):                                      # los hiperparametros son lo del parentesis de la funcion init, n_heads para multiples cabezas (sino quitar y los parametros quitar el for y los [])
        self.W_qs = [ nn.Linear(d_model, d_model) for _ in range(n_heads)]               # los párametros es lo que declaramos en la funcion init
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

        # self.qkv = nn.Linear(d_model, 3*d_model) equivalente a las 3 primeras clineas de init (sin tener en cuenta n_heads)

    def forward(self, x): # x es el tensor de los embbedings
        Q = self.W_q @ x
        K = self.W_k @ x
        V = self.W_v @ x
        A = Q @ K.transpose()
        A /= math.sqrt(self.d_model)
        A = F.softmak(A)
        A = self.dropout(A)
        Z = A @ V
        return Z
"""

# falta la funcion de activacion para la no linealidad (ej: relu, gelu) (un perceptron)


# Mecanismo de atención comentado
#
# PLN 2025/2026 (FDI UCM)
# Antonio F. G. Sevilla <afgs@ucm.es>

import math

import torch
import torch.nn as nn
from torch.nn.functional import softmax


class Attention(nn.Module):
    """Auto-atención multi-cabezal con escala (scaled multi-head self-attention)

    Si `causal=True` en el forward, cada posicion solo atiende a las
    anteriores (util para generacion). Si `causal=False`, cada posicion
    atiende a toda la secuencia (TAREA: para qué querríamos esto?).

    dropout es el porcentaje de dropout a usar.
    """

    def __init__(self, d_model, n_heads, max_seq_len, dropout):
        super().__init__()
        self.n_heads = n_heads
        # Distribuimos la dimensión del modelo entre el numero de cabezas
        self.head_dim = d_model // n_heads
        # Una única matriz para QKV, luego separaremos
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Capa lineal para permitir al modelo reproyectar los vectores contexto
        # TAREA: necesaria? y si la quito?
        self.out = nn.Linear(d_model, d_model)
        # El dropout se activa en train y desactiva en test gracias a pytorch
        self.dropout = nn.Dropout(dropout)
        # La máscara causal pone a -inf las posiciones correspondientes a tokens
        # "futuros" (triangular superior)
        mask = torch.triu(
            torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1
        )
        # Registramos la máscara causal como tensor (no entrenable)
        self.register_buffer("mask", mask)

    def forward(self, x, causal=True):
        # Los tensores de pytorch tienen primero una dimensión batch
        # (entrenamiento más eficiente si hacemos varios a la vez)
        # luego tokens y luego ya la dimensión de los embeddings
        batch_size, n_tokens, d_model = x.shape

        # multiplicamos x por QKV (todo junto), pero separamos a lo largo de la
        # última dimensión para tener las matrices de queries, keys y values
        q, k, v = self.qkv(x).tensor_split(3, dim=-1)
        # separamos en cabezales (ver función más abajo)
        q = self.split_heads(q)  # ¿que estoy buscando?
        k = self.split_heads(k)  # ¿que info tengo?
        v = self.split_heads(v)  # ¿que info puedo aportar?
        # Se comparan Q y K para ver cuánto debe atender un token a otros y se aplican los pesos a V para mezclar info relevante

        # TAREA: Implementar
        # Nota: para escalar, dividir por raíz de head_dim (para que los logits
        # no crezcan sin control)
        # z = ...?
        # TAREA: Implementar

        # 1. Transponemos solo las dos últimas dimensiones de k (-2, -1)
        a = q @ k.transpose(-2, -1)

        # 2. Escalar por la raíz cuadrada de la dimensión del cabezal (head_dim)
        a = a / math.sqrt(self.head_dim)

        # 3. Aplicar la máscara causal (¡CRÍTICO para generar texto!)
        if causal:
            a = a + self.mask[:n_tokens, :n_tokens]

        # 4. Aplicar softmax para obtener probabilidades que sumen 1
        a = torch.nn.functional.softmax(a, dim=-1)

        # 5. Aplicar dropout (opcional pero recomendado)
        a = self.dropout(a)

        # 6. Multiplicar por los values (v)
        z = a @ v

        # "deshacemos" la partición en cabezales
        # (batch_size, n_heads, n_tokens, head_dim) -> (batch_size, n_tokens, d_model)
        z = z.transpose(1, 2).flatten(-2)

        # re-proyectamos con la última transformación
        return self.out(z)

    def split_heads(self, x):
        # (batch_size, n_tokens, d_model) -> (batch_size, n_tokens, n_heads, head_dim)
        # "partimos" la última dimensión (-1, d_model) en n_heads x head_dim
        x = x.unflatten(-1, (self.n_heads, self.head_dim))
        # (batch_size, n_tokens, n_heads, head_dim) -> (batch_size, n_heads, n_tokens, head_dim)
        #  transponemos n_tokens y n_heads para que cada cabezal de atención se
        # "multiplique por separado", haciéndolos independientes
        # TAREA: hacer el álgebra a mano para ver como funciona
        return x.transpose(1, 2)
