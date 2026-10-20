# GLiNER2.5, por dentro

Visualización 3D interactiva y didáctica de GLiNER2.5 para gente no técnica.
Escribes una frase y las etiquetas que buscas. El modelo real se ejecuta y la página dibuja lo que produce cada etapa, de las piezas del texto a las entidades finales.

## Arrancar

```bash
uv run server.py            # modelo base fastino/gliner2.5-multi-v1
open http://localhost:8765
```

`uv` instala las dependencias declaradas en la cabecera de `server.py`.
La primera vez descarga el modelo de Hugging Face (~1,1 GB).

| Opción | Uso |
|---|---|
| `--model <repo>` | Otro checkpoint GLiNER2.5 *boundary*, p. ej. `fastino/gliner2.5-base-v1`. |
| `--port <n>` | Puerto (por defecto 8765). |
| `--check` | Pasa una frase, verifica la salida y termina. |

## Qué se ve

Diez etapas apiladas en una torre, de abajo arriba: texto y etiquetas, tokenizador, fichas (embeddings), 12 capas Transformer, representantes, huecos entre palabras, «¿empieza? ¿acaba?», candidatos, puntuación y decisión.
Cada etapa tiene tres pestañas: «Qué hace» (en lenguaje llano), «Con tu frase» (los valores reales) y «Por dentro» (formas, parámetros y fichero del código).
La vista «Mapa de significados» muestra cómo se mueve cada pieza en el espacio del modelo, capa a capa.

## Ficheros

- `server.py`: carga el modelo, captura con *hooks* las salidas intermedias (estados ocultos y atención del encoder, huecos, logits de inicio/fin/dentro, bolsa de candidatos, notas, abstención) y las sirve en `/api/analyze`.
- `index.html`: la página (Three.js desde CDN, sin compilación).

## Simplificaciones

- La atención se muestra como la media de las 12 cabezas.
- El mapa es una proyección PCA a 3D: las distancias son aproximadas. Las cifras de parecido usan los 768 números reales.
- Solo se recorre la extracción de entidades. El modelo también clasifica y extrae relaciones.
- Límites: 60 palabras y 8 etiquetas por frase, para que el dibujo siga legible.
