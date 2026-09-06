import { useEffect, useState } from "react";

import { boot } from "./engine";
import { Masthead, Waiting, Foot } from "./components/Chrome";
import Home from "./Home";
import Boundary from "./demos/Boundary";
import Overfit from "./demos/Overfit";
import Lenses from "./demos/Lenses";
import Descent from "./demos/Descent";
import Look, { LooksIndex } from "./looks/Look";

const ROUTES = { boundary: Boundary, overfit: Overfit, lenses: Lenses, descent: Descent };

const currentRoute = () => window.location.hash.replace(/^#\/?/, "").split("?")[0];

export default function App() {
  const [progress, setProgress] = useState("starting up");
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const [route, setRoute] = useState(currentRoute);

  useEffect(() => {
    boot(setProgress)
      .then(() => setReady(true))
      .catch((e) => { setProgress(e.message); setFailed(true); });
  }, []);

  useEffect(() => {
    const onHash = () => { setRoute(currentRoute()); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", onHash);

    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  if (!ready) return <Waiting progress={progress} failed={failed} />;

  // Seven quick studies of a replacement world for the home page, off the main navigation
  if (route === "looks") return <LooksIndex />;
  if (route.startsWith("look/")) return <Look id={route.slice(5)} />;

  const Plate = ROUTES[route];

  return (
    <>
      <Masthead route={route} />
      {Plate ? (
        <main className="page">
          <Plate />
          <Foot />
        </main>
      ) : (
        <main className="page">
          <Home />
        </main>
      )}
    </>
  );
}
