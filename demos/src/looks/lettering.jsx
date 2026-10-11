/*
 * What the page says, and how quietly it says it.
 *
 * The drawing is settled; the lettering is not. Seven studies of the same screen, each a
 * different answer to one question: how much type does a live instrument need before it stops
 * being an instrument?
 *
 * The incumbent (`panel`) is here as the anti-reference. It is a display headline, a lede and
 * two buttons in a box — the headline-over-a-button landing that this world's own Overview says
 * it refuses. It is kept so the alternatives can be judged against it rather than in the air.
 *
 * Three things every study must carry, whatever else it drops:
 *   · the wordmark, because nothing else names the product;
 *   · the four plates, because this surface exists to send people to them;
 *   · the repository, because it is the only bridge to who made this.
 */

const REPO = "https://github.com/Roberto-Nsoni/visualML";

/* Each plate described in its own words — the title it already carries — rather than in a
   summary written for the cover. The cover has no business paraphrasing them. */
const PLATES = [
  ["/playground", "playground", "Build a network and watch every neuron draw its own cut"],
  ["/recognize", "recognize", "Draw a digit and watch two trained networks read it"],
];

const Nav = () => (
  <nav className="look-nav">
    {PLATES.map(([href, name]) => <a key={href} href={href}>{name}</a>)}
  </nav>
);

/** The plates handed over in the body of a study rather than filed in the masthead. */
const Plates = () => (
  <nav className="look-plates">
    {PLATES.map(([href, name]) => <a key={href} href={href}>{name}</a>)}
  </nav>
);

const Source = ({ className = "look-nav" }) => (
  <nav className={className}>
    <a href={REPO} target="_blank" rel="noreferrer">source</a>
  </nav>
);

const Band = ({ readout }) => (
  <div className="look-band">
    <span><i>loss</i>{readout ? readout.loss.toFixed(4) : "—"}</span>
    <span><i>accuracy</i>{readout ? (readout.accuracy * 100).toFixed(1) + "%" : "—"}</span>
  </div>
);

const Mark = () => <a className="look-mark" href="/">visualML</a>;

/* ------------------------------------------------------------------ the studies */


/**
 * The cover. One right-hand column: the name, what it is, and where to go.
 *
 * Everything the page says is stacked against a single right edge, so the drawing keeps the
 * whole left of the screen and the type reads as one object rather than three. The wordmark is
 * the only display type; the sentence beneath it states the mechanism and makes no promise; the
 * four plates carry their own titles and are the only thing on the page you are meant to press.
 *
 * The two readings and the repository take the right, the column takes the left, and nothing is
 * written across the middle where the drawing is. What the cover no longer carries is the name
 * of the choreography and its description: the shape on screen explains itself, and naming it
 * was the page talking about its own wallpaper.
 */
function Cover({ readout }) {
  return (
    <>
      <header className="look-head look-head-end"><Source className="look-source" /></header>

      <div className="look-column">
        <h1 className="look-column-mark"><a href="/">visualML</a></h1>
        <p className="look-column-say">
          An autograd engine written from scratch, training in this browser while you read this
        </p>
        <ol className="look-column-plates">
          {PLATES.map(([href, name, what]) => (
            <li key={href}>
              <a href={href}><span>{name}</span><em>{what}</em></a>
            </li>
          ))}
        </ol>
      </div>

      <footer className="look-foot">
        <Band readout={readout} />
      </footer>
    </>
  );
}

/** 00 · The incumbent. A claim in a box, kept as the thing to beat. */
function Panel({ motion, readout }) {
  return (
    <>
      <header className="look-head"><Mark /><Nav /></header>
      <div className="look-say">
        <h1 className="look-title">Machine learning<br />with nothing<br />underneath it</h1>
        <p className="look-lede">
          An autograd engine, seven optimizers and four plates that run the mathematics
          in front of you. No PyTorch. No server.
        </p>
        <div className="look-acts">
          <a className="look-act" href="/boundary">Open the plates</a>
          <a className="look-act look-act-quiet" href={REPO} target="_blank" rel="noreferrer">
            Read the source
          </a>
        </div>
      </div>
      <footer className="look-foot">
        <Band readout={readout} />
        <p className="look-name"><a href="/looks">{motion.name}</a> · {motion.note}</p>
      </footer>
    </>
  );
}

/** 01 · Bare. No claim anywhere. The drawing is the whole argument. */
function Bare({ motion, readout }) {
  return (
    <>
      <header className="look-head">
        <Mark />
        <div className="look-ends"><Nav /><Source /></div>
      </header>
      <footer className="look-foot">
        <Band readout={readout} />
        <p className="look-name"><a href="/looks">{motion.name}</a> · {motion.note}</p>
      </footer>
    </>
  );
}

/** 02 · Legend. The type is silk-screened onto the panel: what this is, in the measured register. */
function Legend({ motion, readout }) {
  return (
    <>
      <header className="look-head">
        <Mark />
        <div className="look-ends"><Nav /><Source /></div>
      </header>
      <div className="look-legend">
        <p><i>on screen</i>{motion.name}</p>
        <p><i>engine</i>autograd, from scratch</p>
        <p><i>running in</i>this browser</p>
        <p><i>plates</i><Plates /></p>
      </div>
      <footer className="look-foot">
        <Band readout={readout} />
        <p className="look-name">{motion.note} · <a href="/looks">all eight</a></p>
      </footer>
    </>
  );
}

/** 03 · Wordmark. The name is the only display type on the page. */
function Wordmark({ motion, readout }) {
  return (
    <>
      <header className="look-head">
        <span />
        <div className="look-ends"><Nav /><Source /></div>
      </header>
      <div className="look-big">
        <h1><a href="/">visualML</a></h1>
        <Plates />
      </div>
      <footer className="look-foot">
        <Band readout={readout} />
        <p className="look-name"><a href="/looks">{motion.name}</a> · {motion.note}</p>
      </footer>
    </>
  );
}

/** 04 · Caption. No display type at all; the claim becomes a caption beside the readout. */
function Caption({ motion, readout }) {
  return (
    <>
      <header className="look-head">
        <Mark />
        <div className="look-ends"><Nav /><Source /></div>
      </header>
      <footer className="look-foot look-foot-wide">
        <Band readout={readout} />
        <p className="look-caption">
          An autograd engine written from scratch, training in this browser while you read this.
          The network is finding the boundary between the two colours.
        </p>
        <Plates />
        <p className="look-name"><a href="/looks">{motion.name}</a> · {motion.note}</p>
      </footer>
    </>
  );
}

/** 05 · Contents. The four plates stop being navigation and become the page. */
function Contents({ motion, readout }) {
  return (
    <>
      <header className="look-head"><Mark /><Source /></header>
      <ol className="look-contents">
        {PLATES.map(([href, name, what], i) => (
          <li key={href}>
            <a href={href}><b>{String(i + 1).padStart(2, "0")}</b><span>{name}</span><em>{what}</em></a>
          </li>
        ))}
      </ol>
      <footer className="look-foot">
        <Band readout={readout} />
        <p className="look-name"><a href="/looks">{motion.name}</a> · {motion.note}</p>
      </footer>
    </>
  );
}

/** 06 · Etched. The label a maker stamps on the back of an instrument, and nothing more. */
function Etched({ motion, readout }) {
  return (
    <>
      <header className="look-head">
        <div className="look-etch">
          <Mark />
          <p>autograd engine · seven optimizers · no server</p>
          <Plates />
        </div>
        <Source />
      </header>
      <footer className="look-foot">
        <Band readout={readout} />
        <p className="look-name"><a href="/looks">{motion.name}</a> · {motion.note}</p>
      </footer>
    </>
  );
}

/** 07 · Corner. Everything the page says is stacked into one corner, out of the drawing's way. */
function Corner({ motion, readout }) {
  return (
    <>
      <header className="look-head"><Mark /><Source /></header>
      <div className="look-corner">
        <p className="look-corner-say">
          An autograd engine, seven optimizers, four plates.<br />
          Running the mathematics in this browser.
        </p>
        <Nav />
      </div>
      <footer className="look-foot">
        <Band readout={readout} />
        <p className="look-name"><a href="/looks">{motion.name}</a> · {motion.note}</p>
      </footer>
    </>
  );
}

export const LETTERINGS = [
  { id: "cover", name: "Cover", note: "The one the site wears: wordmark, one sentence and the four plates, all against a single right edge.", Chrome: Cover },
  { id: "panel", name: "Panel", note: "The incumbent: a claim, a lede and two buttons in a box. Here to be beaten.", Chrome: Panel },
  { id: "bare", name: "Bare", note: "No claim anywhere. Masthead, drawing, readout. The drawing is the whole argument.", Chrome: Bare },
  { id: "legend", name: "Legend", note: "Silk-screened onto the panel, in the same register as the numbers.", Chrome: Legend },
  { id: "wordmark", name: "Wordmark", note: "The name is the only display type on the page.", Chrome: Wordmark },
  { id: "caption", name: "Caption", note: "No display type at all. The claim becomes a caption beside the readout.", Chrome: Caption },
  { id: "contents", name: "Contents", note: "The four plates stop being navigation and become the page.", Chrome: Contents },
  { id: "etched", name: "Etched", note: "The label stamped on the back of an instrument, and nothing more.", Chrome: Etched },
  { id: "corner", name: "Corner", note: "Everything the page says, stacked into one corner, out of the drawing's way.", Chrome: Corner },
];

export const letteringById = (id) => LETTERINGS.find((l) => l.id === id);
