// Shared furniture: the masthead, the measured band at the foot of every section, the plate
// wrapper the demos live inside, and the wait.

const REPO = "https://github.com/Roberto-Nsoni/visualML";

/** Drawn, like every other mark here. A text arrow is a glyph standing in for an icon set. */
export const Arrow = ({ size = 11 }) => (
  <svg viewBox="0 0 16 16" width={size} height={size} fill="none" stroke="currentColor"
       strokeWidth="1.6" aria-hidden="true">
    <path d="M2 8h11M9 4l4 4-4 4" />
  </svg>
);

export function Masthead({ route }) {
  const links = [
    ["#/boundary", "boundary"],
    ["#/overfit", "capacity"],
    ["#/lenses", "two lenses"],
    ["#/descent", "descent"],
  ];

  return (
    <header className="masthead">
      <a className="masthead-mark" href="#/">
        <span className="masthead-name">visualML</span>
        <span className="masthead-sub overline">a machine-learning engine, cut open</span>
      </a>

      <nav className="masthead-nav">
        {links.map(([href, label]) => (
          <a key={href} href={href} className={"masthead-link" + (route === href.slice(2) ? " is-here" : "")}>
            {label}
          </a>
        ))}
        <a className="btn masthead-repo" href={REPO} target="_blank" rel="noreferrer">
          source <Arrow />
        </a>
      </nav>
    </header>
  );
}

/** Measured values sit in a band at the foot of the section, like a drawing's data strip. */
export function MetricBand({ items }) {
  return (
    <div className="band">
      {items.map(({ label, value, unit, lit }) => (
        <div className="band-cell" key={label}>
          <span className="overline">{label}</span>
          <span className={"band-value measured" + (lit ? " is-lit" : "")}>
            {value}
            {unit && <i className="band-unit">{unit}</i>}
          </span>
        </div>
      ))}
    </div>
  );
}

/** A readout tied to the thing it describes, on a leader line. */
export function Leader({ label, value, note }) {
  return (
    <div className="leader">
      <span className="leader-line" aria-hidden="true" />
      <span>
        <span className="overline">{label}</span>
        {note && <span className="leader-note prose">{note}</span>}
      </span>
      <span className="leader-value measured">{value}</span>
    </div>
  );
}

export function Plate({ title, note, children }) {
  return (
    <section className="plate">
      <div className="plate-head">
        <h2 className="headline">{title}</h2>
        {note && <p className="prose plate-note">{note}</p>}
      </div>
      {children}
    </section>
  );
}

export function Waiting({ progress, failed }) {
  return (
    <div className="waiting">
      <div className="waiting-sheet">
        <h1 className="display waiting-title">
          visual<br />ML
        </h1>
        <p className={"prose waiting-note" + (failed ? " is-failed" : "")}>
          {failed ? `Could not start: ${progress}` : progress}
          {!failed && <span className="waiting-dots" aria-hidden="true" />}
        </p>
        <p className="waiting-small prose">
          Python and NumPy are loading into this page. About 15&nbsp;MB, once.
        </p>
      </div>
      <div className="waiting-ground" aria-hidden="true" />
    </div>
  );
}

export function Foot() {
  return (
    <footer className="foot">
      <hr className="rule" />
      <div className="foot-row">
        <p className="prose foot-note">
          The engine, its tests and this page are one repository. Nothing here is precomputed on a
          server: the mathematics runs in your browser.
        </p>
        <a className="btn" href={REPO} target="_blank" rel="noreferrer">
          read the source <Arrow />
        </a>
      </div>
    </footer>
  );
}
