import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";

const py = await loadPyodide();
await py.loadPackage(["numpy", "micropip"]);
const micropip = py.pyimport("micropip");
await micropip.install("http://localhost:5199/gradopsy-0.1.0-py3-none-any.whl");   // igual que en el navegador
console.log("wheel instalado desde URL");

py.runPython(readFileSync("../src/engine.py", "utf8"));
console.log("engine.py cargado");

const si = py.globals.get("surface_info");
const info = si("double_well", 240).toJs({ dict_converter: Object.fromEntries });
console.log("surface_info:", { n: info.n, x_range: info.x_range, minima: info.minima, z_min: info.z_min });

const sz = py.globals.get("surface_z");
const z = new Float32Array(sz("double_well", 240).toJs().buffer);
console.log("mapa de alturas:", z.length, "valores, primero", z[0].toFixed(3));

const race = py.globals.get("race");
const r = race("double_well", py.toPy([{ key: "adam", lr: 0.05 }, { key: "newton" }, { key: "sgd", lr: 0.002 }]), 200, null)
             .toJs({ dict_converter: Object.fromEntries });
for (const [k, v] of Object.entries(r)) {
  const fin = v.path[v.path.length - 1];
  console.log(`  ${k.padEnd(8)} ${v.steps} pasos -> (${fin[0].toFixed(4)}, ${fin[1].toFixed(4)})  claves: ${Object.keys(v).join(",")}`);
}
