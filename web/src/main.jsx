import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import "./world/world.css";
import "./world/parts.css";
import "./world/layout.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
