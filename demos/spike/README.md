# Spike de Pyodide

Comprueba la premisa de D11: que el motor corre **dentro del navegador**, sin backend de Python.
Pyodide es CPython + NumPy compilados a WebAssembly. Aquí se ejecuta en Node, que usa el mismo
runtime que el navegador.

```bash
cd ../..  && uv build --wheel     # genera dist/visualml-0.1.0-py3-none-any.whl
cd demos/spike && npm i && node spike.mjs
```

El script instala el wheel con micropip, corre la carrera de `descent` y un entrenamiento de MLP,
y **compara los resultados contra Python nativo** (tienen que salir idénticos, es determinista)
midiendo cuánto tarda cada uno.

## Resultado medido (2026-09-02, pyodide 314.0.6, numpy 2.4.6)

| | nativo | pyodide | |
|---|---|---|---|
| 7 optimizadores × 200 pasos sobre `double_well` | 0.51 s | 1.26 s | **2.5×** |
| MLP [2,16,16,3] × 300 pasos de Adam sobre espirales | 0.26 s (0.9 ms/paso) | 0.73 s (**2.4 ms/paso**) | 2.9× |

**Los números salen idénticos bit a bit** en los siete optimizadores y en la loss final del MLP.

Arranque: **3.4 s**. Descarga: ~14.7 MB sin comprimir (9.2 wasm + 2.5 stdlib + 2.9 numpy +
0.03 del propio visualml).

## Qué implica

- **D11 confirmada.** El sitio puede ser estático. Sin FastAPI, sin WebSockets, sin sesiones.
- **D13 se puede relajar.** A 2.4 ms por paso salen ~400 pasos/segundo, muy por encima de los
  60 fps que necesita una animación: `descent` y `boundary` entrenan en vivo con holgura, y
  probablemente `predict` también.
- El coste real no es la velocidad, es **la descarga inicial y los 3.4 s de arranque**. Eso es
  diseño de estado de carga, no un problema de arquitectura.
