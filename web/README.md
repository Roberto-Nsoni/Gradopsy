# web — los demos

Vite + React. El motor **no se reimplementa en JavaScript**: el paquete `visualml` corre dentro
del navegador con Pyodide (D11), así que lo que ves calculado es el mismo Python que corre en
los tests. `src/engine.py` es el único puente, y no tiene matemáticas: solo mueve datos.

```bash
npm install
npm run dev          # compila el wheel, lo copia a public/ y abre localhost:5173
```

`npm run dev` ejecuta antes `npm run wheel`, que necesita `uv` en el PATH. Si cambias algo del
motor en `src/ml/`, vuelve a lanzar `npm run wheel` para que la web coja los cambios.

## Cómo está montado

- **La carrera se calcula entera de golpe** y la página reproduce la grabación. Por eso mover la
  línea de tiempo es instantáneo aunque calcular tarde ~1 s: es el modo "grabado" de D13.
- **El contorno se pinta por percentiles**, no por min/max. Las superficies van de un rango de 11
  (doble pozo) a uno de 10⁶ (Goldstein-Price); con un mapeo fijo unas salen planas y otras
  quemadas. Por percentil todas usan el rango entero, y las isolíneas quedan de área igual: donde
  se apiñan, la pendiente es fuerte.
- **Pyodide entra por `<script>` en `index.html`**, no por el bundler. El paquete de npm arrastra
  ramas de Node que Vite tiene que apuntalar y que luego revientan en el navegador.
- **El nombre del wheel importa**: micropip rechaza cualquier cosa que no cumpla PEP 427, así que
  se conserva `visualml-0.1.0-py3-none-any.whl` y el nombre se lee de `public/wheel.txt`.

## `descent`

Los siete optimizadores sobre la misma superficie desde el mismo punto. Clic en la superficie
para moverlo. El panel de abajo enseña **lo que cada método lleva encima** en ese instante
—velocidad, escalado por coordenada, `H`, `λ`, `α`—, que es lo que separa "bolas rodando" de
entender por qué se comportan distinto. SGD dice "memoria: ninguna", y ese es el contraste.

En `double_well` está la carga del demo: todos escapan de la silla menos Newton, que se queda
clavado en el origen porque resuelve `g = 0` y una silla también es un punto estacionario.

## Verificado en navegador

`spike/` guarda la comprobación de que todo esto es posible (mismos números que Python nativo,
2.5–2.9× más lento). El chasis se ha probado con Playwright: carga, cambio de superficie, clic
para mover el punto de partida y cajón de código, sin errores de consola.
