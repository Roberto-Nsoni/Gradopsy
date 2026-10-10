// Drawn, like every other mark here: one 16-unit grid, one 1.5 stroke, square ends.

const PATHS = {
  play: <path d="M4.5 3v10l8-5z" fill="currentColor" stroke="none" />,
  pause: <path d="M5 3v10M11 3v10" strokeWidth="2.2" />,
  step: <><path d="M3.5 3.5v9l6.5-4.5z" fill="currentColor" stroke="none" /><path d="M12.5 3.5v9" strokeWidth="1.8" /></>,
  back: <><path d="M12.5 3.5v9L6 8z" fill="currentColor" stroke="none" /><path d="M3.5 3.5v9" strokeWidth="1.8" /></>,
  reset: <><path d="M2.5 3.5v9" strokeWidth="1.8" /><path d="M9 3.5v9L3.5 8z M14 3.5v9L8.5 8z" fill="currentColor" stroke="none" /></>,
  brush: <><circle cx="8" cy="8" r="5.5" /><circle cx="8" cy="8" r="1.4" fill="currentColor" stroke="none" /><circle cx="5.2" cy="6.2" r="0.9" fill="currentColor" stroke="none" /><circle cx="10.6" cy="9.8" r="0.9" fill="currentColor" stroke="none" /><circle cx="9.9" cy="5.4" r="0.9" fill="currentColor" stroke="none" /></>,
  point: <><path d="M8 1.5v3.5M8 11v3.5M1.5 8H5M11 8h3.5" /><circle cx="8" cy="8" r="1.6" fill="currentColor" stroke="none" /></>,
  erase: <><circle cx="8" cy="8" r="5.5" strokeDasharray="2 1.6" /><path d="M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8" /></>,
  move: <path d="M8 1.5v13M1.5 8h13M8 1.5L6 3.5M8 1.5l2 2M8 14.5l-2-2M8 14.5l2-2M1.5 8l2-2M1.5 8l2 2M14.5 8l-2-2M14.5 8l-2 2" />,
  undo: <><path d="M5.5 3.5L2.5 6.5l3 3" /><path d="M2.5 6.5H10a3.5 3.5 0 010 7H6" /></>,
  redo: <><path d="M10.5 3.5l3 3-3 3" /><path d="M13.5 6.5H6a3.5 3.5 0 000 7h4" /></>,
  clear: <path d="M3 4.5h10M6 4.5V2.5h4v2M4.5 4.5l.8 9h5.4l.8-9" />,
  dice: <><rect x="2.5" y="2.5" width="11" height="11" /><circle cx="5.5" cy="5.5" r="1" fill="currentColor" stroke="none" /><circle cx="10.5" cy="10.5" r="1" fill="currentColor" stroke="none" /><circle cx="8" cy="8" r="1" fill="currentColor" stroke="none" /></>,
  plus: <path d="M8 3v10M3 8h10" />,
  minus: <path d="M3 8h10" />,
  copy: <><rect x="5.5" y="5.5" width="8" height="8" /><path d="M10.5 5.5v-3h-8v8h3" /></>,
  pulse: <path d="M1.5 8h3l1.5-4 3 8 1.5-4h4" />,
  live: <><path d="M3 3.5v9l6-4.5z" fill="currentColor" stroke="none" /><path d="M10 3.5v9M13 3.5v9" strokeWidth="1.6" /></>,
};

export default function Icon({ name, size = 14 }) {
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} fill="none" stroke="currentColor"
         strokeWidth="1.5" strokeLinecap="square" aria-hidden="true" focusable="false">
      {PATHS[name]}
    </svg>
  );
}

/** A class's own mark, the same shape the screen draws it in. */
export function ClassMark({ shape, colour, size = 12, hollow = false }) {
  const fill = hollow ? "none" : colour;
  const common = { fill, stroke: colour, strokeWidth: hollow ? 1.6 : 0 };
  return (
    <svg viewBox="0 0 16 16" width={size} height={size} aria-hidden="true" focusable="false">
      {shape === "disc" && <circle cx="8" cy="8" r="5.5" {...common} />}
      {shape === "square" && <rect x="3" y="3" width="10" height="10" {...common} />}
      {shape === "triangle" && <path d="M8 1.8l6.4 11H1.6z" {...common} />}
      {shape === "diamond" && <path d="M8 1.2L14.8 8 8 14.8 1.2 8z" {...common} />}
    </svg>
  );
}
