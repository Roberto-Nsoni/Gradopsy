// The signature. In the world this is a band of light travelling across a section to say what
// hour it is; here it says which training step you are looking at. Everything behind the head
// has happened, the head itself is the strike, and nothing ahead is drawn yet.

const PLAY = "M4 2l9 6-9 6z";
const PAUSE = "M4 2h3v12H4zm5 0h3v12H9z";
const RESET = "M3 2h2v12H3zm10 0v12L5 8z";

const Icon = ({ d }) => (
  <svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor" aria-hidden="true">
    <path d={d} />
  </svg>
);

export default function Scrubber({ step, setStep, total, playing, setPlaying, label = "step",
                                  shown, shownTotal, ready = 1 }) {
  const at = total > 1 ? step / (total - 1) : 0;

  return (
    <div className="scrub">
      <div className="scrub-controls">
        <button className="scrub-btn" onClick={() => { setStep(0); setPlaying(false); }}
                title="Back to the first step" aria-label="Back to the first step">
          <Icon d={RESET} />
        </button>
        <button className="scrub-btn scrub-btn-sun" onClick={() => setPlaying(!playing)}
                title={playing ? "Hold" : "Run"} aria-label={playing ? "Hold" : "Run"}>
          <Icon d={playing ? PAUSE : PLAY} />
          <span>{playing ? "hold" : "run"}</span>
        </button>
      </div>

      <div className="scrub-band">
        <div className="scrub-track">
          {/* Three states on one band: bare track is a step that does not exist yet, low phosphor is
              computed and waiting, gold is the part you have watched. While a run streams the
              track darkens ahead of the playhead, which is the honest picture of what the tab
              is doing — no spinner needed, because the progress is the drawing. */}
          <div className="scrub-ready" style={{ width: `${Math.min(1, ready) * 100}%` }} />
          <div className="scrub-lit" style={{ width: `${at * 100}%` }} />
          <div className="scrub-head" style={{ left: `${at * 100}%` }} />
          {/* The counter rides the head all the way to both ends: it sits on the playhead and
              swings from left-aligned to right-aligned as the head crosses the band, instead of
              being parked at a fixed inset where it came unstuck from the mark it labels — and
              hung four pixels off the right of a phone screen. */}
          <span className="scrub-now measured"
                style={{ left: `${at * 100}%`, transform: `translateX(${-at * 100}%)` }}>
            {String(shown ?? step).padStart(3, "0")} / {String(shownTotal ?? total - 1).padStart(3, "0")}
          </span>
        </div>

        <input
          type="range" min="0" max={Math.max(total - 1, 0)} value={step}
          onChange={(e) => { setStep(+e.target.value); setPlaying(false); }}
          aria-label={`Scrub through the run by ${label}`} className="scrub-input"
        />

      </div>

      <span className="scrub-label overline">{label}</span>
    </div>
  );
}
