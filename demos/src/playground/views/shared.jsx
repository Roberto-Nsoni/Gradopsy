import { useLayoutEffect, useState } from "react";

import { ACTIVATIONS, MAX_UNITS } from "../config";
import Icon from "../icons";

/* What the parked warp study needs from outside itself: its own size, and a layer's controls. */

export function useBox(ref) {
  const [size, setSize] = useState({ w: 400, h: 400 });
  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return undefined;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(node);
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

/** The two controls a hidden layer carries: how many units, and which activation. */
export function LayerControls({ index, layer, onUnits, onAct, compact = false }) {
  return (
    <span className={"pgv-ctl" + (compact ? " is-compact" : "")}>
      <span className="pg-net-units">
        <button type="button" className="pg-tiny" aria-label={`Fewer units in layer ${index + 1}`}
                disabled={layer.units <= 1} onClick={() => onUnits(index, -1)}><Icon name="minus" size={10} /></button>
        <b>{layer.units}</b>
        <button type="button" className="pg-tiny" aria-label={`More units in layer ${index + 1}`}
                disabled={layer.units >= MAX_UNITS} onClick={() => onUnits(index, 1)}><Icon name="plus" size={10} /></button>
      </span>
      <select className="pg-net-act" value={layer.act} aria-label={`Activation of layer ${index + 1}`}
              onChange={(e) => onAct(index, e.target.value)}>
        {ACTIVATIONS.map((a) => <option key={a} value={a}>{a}</option>)}
      </select>
    </span>
  );
}
