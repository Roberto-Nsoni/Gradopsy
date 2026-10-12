// Does gradopsy run inside the browser? Pyodide is CPython + NumPy compiled to WebAssembly,
// which is the whole premise of D11. This checks two things: that the numbers come out
// identical to native Python, and how much slower it is.
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";

const RAIZ = new URL("../../", import.meta.url).pathname;
const WHEEL = "gradopsy-0.1.0-py3-none-any.whl";

const t0 = performance.now();
const pyodide = await loadPyodide();
await pyodide.loadPackage(["numpy", "micropip"]);
console.log(`pyodide + numpy arrancados en ${((performance.now() - t0) / 1000).toFixed(1)} s`);

// El wheel se copia al sistema de ficheros virtual y micropip lo instala desde ahi
pyodide.FS.writeFile(`/${WHEEL}`, readFileSync(`${RAIZ}dist/${WHEEL}`));
const micropip = pyodide.pyimport("micropip");
await micropip.install(`emfs:/${WHEEL}`);
console.log("gradopsy instalado desde el wheel");

pyodide.FS.writeFile("/carrera.py", readFileSync(new URL("./carrera.py", import.meta.url).pathname));
pyodide.runPython(`import sys; sys.path.insert(0, "/")`);

const enPyodide = pyodide.runPython(`
import json
from carrera import carrera
json.dumps(carrera("double_well", 200))
`);

const nativo = execFileSync(`${RAIZ}.venv/bin/python`,
  ["-c", `import sys, json; sys.path.insert(0, "${new URL(".", import.meta.url).pathname}"); from carrera import carrera; print(json.dumps(carrera("double_well", 200)))`],
  { encoding: "utf8" });

const a = JSON.parse(enPyodide), b = JSON.parse(nativo);
const segPyodide = a.__segundos__, segNativo = b.__segundos__;
delete a.__segundos__; delete b.__segundos__;

console.log("\noptimizador   pyodide                     nativo                      iguales");
let todosIguales = true;
for (const nombre of Object.keys(a)) {
  const iguales = JSON.stringify(a[nombre]) === JSON.stringify(b[nombre]);
  todosIguales &&= iguales;
  console.log(`${nombre.padEnd(13)} ${JSON.stringify(a[nombre]).padEnd(27)} ${JSON.stringify(b[nombre]).padEnd(27)} ${iguales ? "si" : "NO"}`);
}

console.log(`\n200 pasos x 7 optimizadores`);
console.log(`  nativo  : ${segNativo.toFixed(3)} s`);
console.log(`  pyodide : ${segPyodide.toFixed(3)} s   (${(segPyodide / segNativo).toFixed(1)}x mas lento)`);
// El caso que de verdad decide D13: un MLP entrenando en vivo, no dos parametros
const mlpPyodide = JSON.parse(pyodide.runPython(`
import json
from carrera import entrenamiento
json.dumps(entrenamiento(300))
`));
const mlpNativo = JSON.parse(execFileSync(`${RAIZ}.venv/bin/python`,
  ["-c", `import sys, json; sys.path.insert(0, "${new URL(".", import.meta.url).pathname}"); from carrera import entrenamiento; print(json.dumps(entrenamiento(300)))`],
  { encoding: "utf8" }));

const mismaLoss = mlpPyodide.loss === mlpNativo.loss;
console.log(`\nMLP [2,16,16,3] sobre espirales, 300 pasos de Adam`);
console.log(`  nativo  : ${mlpNativo.__segundos__.toFixed(3)} s   (${(1000 * mlpNativo.__segundos__ / 300).toFixed(1)} ms/paso)`);
console.log(`  pyodide : ${mlpPyodide.__segundos__.toFixed(3)} s   (${(1000 * mlpPyodide.__segundos__ / 300).toFixed(1)} ms/paso, ${(mlpPyodide.__segundos__ / mlpNativo.__segundos__).toFixed(1)}x)`);
console.log(`  misma loss final: ${mismaLoss ? "SI" : "NO"}  (${mlpPyodide.loss})`);

console.log(`\nmismos numeros: ${todosIguales && mismaLoss ? "SI" : "NO"}`);
process.exit(todosIguales && mismaLoss ? 0 : 1);
