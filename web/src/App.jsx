import { Component, useEffect, useState } from "react";

import { boot } from "./engine";
import Waiting from "./looks/Waiting";
import Look, { LooksIndex, TypeIndex } from "./looks/Look";
import Playground from "./playground/Playground";

// The four plates of the old world (demos/, components/, world/) are off the routes, not deleted:
// the playground replaces all of them and they stay in the repository as reference.
const ROUTES = { playground: Playground };

/**
 * If a page ever throws while rendering, say so on the tube instead of leaving it black. The
 * error still goes to the console, which is where it is useful.
 */
class Guard extends Component {
  constructor(props) { super(props); this.state = { failed: false }; }
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error, info) { console.error("the page broke", error, info?.componentStack); }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="look" style={{ display: "grid", placeItems: "center", padding: 24 }}>
        <div className="look-column look-column-mid" role="alert">
          <h1 className="look-column-mark">visualML</h1>
          <p className="look-column-say">Something on this page broke. Reload to start it again; the details are in the console.</p>
          <button type="button" className="look-act" onClick={() => window.location.reload()}>Reload</button>
        </div>
      </div>
    );
  }
}

// Real paths — `/playground`, not `#/playground`. An old hash link is rewritten on arrival, and
// internal links are caught below so moving between pages never reloads Python.
function currentRoute() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  if (hash) window.history.replaceState(null, "", "/" + hash);
  return window.location.pathname.replace(/^\/+|\/+$/g, "");
}

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
    const onPop = () => { setRoute(currentRoute()); window.scrollTo(0, 0); };
    const onClick = (e) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = e.target.closest?.("a[href]");
      if (!a || a.target || a.origin !== window.location.origin) return;
      e.preventDefault();
      if (a.pathname + a.search !== window.location.pathname + window.location.search) {
        window.history.pushState(null, "", a.pathname + a.search);
      }
      onPop();
    };
    window.addEventListener("popstate", onPop);
    document.addEventListener("click", onClick);

    return () => { window.removeEventListener("popstate", onPop); document.removeEventListener("click", onClick); };
  }, []);

  if (!ready) return <Waiting progress={progress} failed={failed} />;

  // Seven quick studies of a replacement world for the home page, off the main navigation
  if (route === "looks") return <LooksIndex />;
  if (route.startsWith("look/")) return <Look id={route.slice(5)} />;

  // Nine studies of what the cover should say, off the main navigation
  if (route === "type") return <TypeIndex />;
  if (route.startsWith("type/")) return <Look chrome={route.slice(5)} />;

  // A study of the network panel, on the same live playground, off the main navigation
  if (route === "playground/linked") return <Guard key="playground"><Playground study="linked" /></Guard>;

  // The cover is the front door: the phosphor screen, a choreography picked at random
  if (!ROUTES[route]) return <Look />;

  const Plate = ROUTES[route];

  return <Guard key={route}><Plate /></Guard>;
}
