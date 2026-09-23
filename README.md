# P5: Generación de Texto y Reconocimiento de Entidades con Transformer

## 👥 Equipo de Desarrollo (Grupo 03)
* **Pablo Alonso Romero**
* **Rodrigo Jesús-Portanet Martínez**

---

## 📝 Descripción del Proyecto

Esta práctica implementa un sistema de PLN de nivel superior que incluye:

- **Entrenamiento de un modelo de lenguaje causal (LLM)** basado en un Transformer pequeño (probando varias configuraciones).
- **Generación de texto** a partir de un *prompt* usando el modelo entrenado.
- **Entrenamiento de un modelo NER (Named Entity Recognition)** sobre un corpus anotado.
- **Inferencia NER** en archivos de texto para extraer entidades reconocidas.

El proyecto está diseñado para funcionar localmente con Python 3.12+, usando `uv` como gestor de dependencias y `torch` para el backend de redes neuronales.

---

## ✨ Características principales

1. **Modelo causal de generación de texto**
   - Implementación de un Transformer.
   - Generación autoregresiva con sampling y temperatura.
   - Weight tying entre embeddings de entrada y cabeza de salida.

2. **Tokenización BPE propia**
   - Tokenizador Byte Pair Encoding entrenado sobre el corpus de entrenamiento.
   - Encoding y decoding adaptados a vocabulario reducido.

3. **Entrenamiento de NER**
   - El modelo NER hereda la arquitectura base del Transformer causal.
   - Entrenamiento con divisiones de datos / train-val-test.
   - Grid search sobre hiperparámetros y cálculo de pesos de clase para manejo de etiquetas desbalanceadas.

4. **Inferencia de entidades**
   - Lectura de texto, tokenización y agrupación de tokens consecutivos de una misma entidad.
   - Etiquetas principales: `o` (fuera de entidad), `pi`, `pc`, `li`, `lc`.

5. **Pipeline modular**
   - Código separado en módulos claros: CLI (`main.py`), generación (`generar_texto.py`), entrenamiento LLM (`p5/causal_train.py`), NER (`ner/training.py`), tokenización (`p5/tokenizer.py`) y carga de corpus (`p5/corpus.py`).

---

## 📁 Estructura del proyecto

```text
p5/
├── README.md
├── pyproject.toml
├── uv.lock
├── main.py
├── generar_texto.py
├── p5_causal_2603.pth
├── p5_ner_2603.pth
├── mejor_tokenizador.json
├── resources/
│   ├── alice_in_wonderland.txt
│   └── merged_2.json
├── ner/
│   ├── __init__.py
│   ├── data.py
|   ├── model.py
│   └── training.py
└── p5/
    ├── __init__.py
    ├── attention.py
    ├── causal_llm.py
    ├── causal_train.py
    ├── corpus.py
    ├── tokenizer.py
    ├── transformer.py
```

---

## ⚙️ Requisitos e instalación

### Requisitos mínimos

- Python 3.12+
- `uv` instalado globalmente
- `torch` configurado en `pyproject.toml`

### Instalar dependencias

Desde la carpeta `p5` (superior):

```bash
uv sync
```

Si el comando anterior da problemas con `torch`, asegúrate de que `uv` respete la fuente configurada en el `pyproject.toml` y de tener acceso a la red para descargar los paquetes.

---

## 🚀 Uso principal

Todos los comandos se invocan mediante el script instalado `fdi-pln-2603-p5`.

### 1. Entrenar el tokenizador

```bash
uv run fdi-pln-2603-p5 train-tokenizer
```

Este comando entrena el tokenizador BPE con el corpus `resources/alice_in_wonderland.txt` y lo guarda en `mejor_tokenizador.json`.

### 2. Entrenar el modelo causal

```bash
uv run fdi-pln-2603-p5 train-causal
```

Este comando entrena el modelo de lenguaje con el corpus `resources/alice_in_wonderland.txt`, usando el tokenizador guardado en `mejor_tokenizador.json`, y guarda los pesos en `p5_causal_2603.pth`.

### 3. Generar texto

```bash
uv run fdi-pln-2603-p5 generate "Tu prompt aquí"
```

Opciones disponibles:

```bash
uv run fdi-pln-2603-p5 generate "Tu prompt" --max-tokens 100 --temperature 0.8
```

### 4. Entrenar el modelo NER

```bash
uv run fdi-pln-2603-p5 train-ner
```

Este comando usa `resources/merged_2.json` como conjunto de entrenamiento y valida el modelo NER. Guarda el resultado en `p5_ner_2603.pth`.

### 4. Ejecutar el NER en un archivo de texto

```bash
uv run fdi-pln-2603-p5 ner resources/alice_in_wonderland.txt
```

Devuelve una lista de entidades detectadas y su etiqueta.
Nosotros lo hemos probado con alice_in_wonderland.txt, se podría probar con uno nuevo con frases inventadas.

---

## 🧠 Descripción técnica de los módulos

### `main.py`

- Define la CLI con `click`.
- Expone los comandos: `train-tokenizer`, `train-causal`, `generate`, `train-ner` y `ner`.
- Gestiona las rutas relativas y absolutas de los recursos.

### `generar_texto.py`

- Carga el tokenizador guardado y los pesos del modelo causal.
- Infiere la configuración del modelo a partir del `state_dict`.
- Ejecuta la generación autoregresiva y decodifica tokens a texto.

### `p5/causal_train.py`

- Construye un dataset deslizante (`TextDataset`) para language modeling.
- Entrena con `DataLoader` y `AdamW`.
- Implementa validación y early stopping opcional.
- Código del profesor excepto última sección

### `p5/corpus.py`

- Carga un archivo `.txt` o concatena todos los `.txt` de un directorio.
- Permite usar un corpus único o un conjunto de documentos como entrada.
- Código del profesor (entendido)

### `p5/tokenizer.py`

- Implementa un tokenizador BPE sencillo entrenado sobre el texto.
- Genera un vocabulario a partir de caracteres y aprende merges frecuentes.
- Codifica y decodifica texto en tokens.
- Código del profesor (entendido)


### `p5/causal_llm.py`

- Implementa un modelo de lenguaje causal basado en Transformer.
- Añade una cabeza linear de salida con `weight tying`.
- Genera texto token por token usando sampling con temperatura.
- Código del profesor (entendido)

### `p5/transformer.py` y `p5/attention.py`

- Implementan un transformer básico con bloques de atención multi-cabeza.
- Soportan atención causal para generación y atención completa para tareas de clasificación.
- Códigos del profesor (entendidos)

### `ner/training.py`

- Crea un modelo NER que reutiliza la arquitectura del transformer causal.
- Aplica un clasificador adicional sobre los hidden states.
- Hace grid search de hiperparámetros y entrena el modelo final.
- Guarda el modelo NER con tokenizer, etiquetas y configuración.

### `ner/model.py`

- Construye el modelo NER con un transformer base y una cabeza clasificadora.
- Soporta una capa oculta adicional cuando `classifier_hidden > 0`.
- Calcula la pérdida con `cross_entropy` y pesos de clase para balancing.

### `ner/data.py`

- Define la carga de datos anotados desde JSON.
- Normaliza etiquetas y convierte tokens a IDs.
- Maneja subtokens para que los labels se extiendan correctamente.
- Implementa el `collate` para batches con padding.

---

## 📌 Datos y modelos

- `resources/alice_in_wonderland.txt`: corpus de entrada para entrenamiento del LLM.
- `resources/merged_2.json`: dataset anotado para NER.
- `p5_causal_2603.pth`: pesos del modelo causal entrenado.
- `mejor_tokenizador.json`: tokenizador BPE entrenado.
- `p5_ner_2603.pth`: checkpoint del modelo NER final.

---

## 💡 Notas importantes

- El proyecto funciona con GPU si está disponible, pero también puede correr en CPU.
- El tokenizador y los pesos del modelo deben corresponder entre sí.
- El entrenamiento de NER asume que `mejor_tokenizador.json` y `p5_causal_2603.pth` existen y son compatibles.
- Para mejorar el sistema se puede:
  - aumentar el tamaño del vocabulario BPE,
  - mejorar el corpus de NER,
  - añadir top-k en la generación del LLM.

---

## 🧾 Referencia rápida de comandos

```bash
uv run fdi-pln-2603-p5 train-tokenizer
uv run fdi-pln-2603-p5 train-causal
uv run fdi-pln-2603-p5 generate "Frase inicial" --max-tokens 120 --temperature 0.7
uv run fdi-pln-2603-p5 train-ner
uv run fdi-pln-2603-p5 ner resources/alice_in_wonderland.txt
```

--- 

## 🛠️ Calidad del Código 

El código cumple con los más altos estándares de estilo. Para verificar o aplicar el formato requerido por la práctica, utiliza:
```bash
uv format
uv format --check
```

---
