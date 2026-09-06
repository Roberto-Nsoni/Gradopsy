/*
 * The arrangement the live section trains on: six lobes sixty degrees apart with the classes
 * alternating, turning slowly. Opposite lobes disagree, so no straight line and no single
 * stripe separates them — only a real rosette, which is the point.
 */

export const LOBES = 6;
export const PER_LOBE = 3;
export const CLASSES = 2;

const ORBIT = 0.56;
const SPREAD = 0.15;
const SPIN = 0.1;

export const lobeAt = (lobe, t) => {
  const a = t * SPIN + (lobe * 2 * Math.PI) / LOBES;
  return [Math.cos(a) * ORBIT, Math.sin(a) * ORBIT];
};

export function seedPoints() {
  const rng = (i) => Math.sin(i * 12.9898) * 43758.5453;
  const frac = (v) => v - Math.floor(v);

  return Array.from({ length: LOBES * PER_LOBE }, (_, i) => {
    const lobe = i % LOBES;
    const a = frac(rng(i + 1)) * Math.PI * 2;
    const r = Math.sqrt(frac(rng(i + 97))) * SPREAD;

    return { lobe, cls: lobe % 2, ox: Math.cos(a) * r, oy: Math.sin(a) * r,
             wobble: frac(rng(i + 41)) * Math.PI * 2 };
  });
}

export const clampUnit = (v) => Math.max(-0.94, Math.min(0.94, v));

export function placeAt(seeds, t) {
  return seeds.map(({ lobe, ox, oy, wobble }) => {
    const [cx, cy] = lobeAt(lobe, t);
    const w = t * 0.6 + wobble;
    return [clampUnit(cx + ox + Math.cos(w) * 0.02), clampUnit(cy + oy + Math.sin(w * 1.3) * 0.02)];
  });
}

/**
 * Marching squares over the class indicator, already in pixel space and the right way up.
 * Row 0 of a field is y = -1, the bottom of the plane; a canvas draws row 0 at the top.
 */
export function boundarySegments(field, confidence, n, size) {
  const at = (r, c) => (field[r * n + c] === 0 ? 1 : -1) * (confidence[r * n + c] / 255);
  const cross = (a, b) => a / (a - b);
  const out = [];

  for (let r = 0; r < n - 1; r++) {
    for (let c = 0; c < n - 1; c++) {
      const tl = at(r, c), tr = at(r, c + 1), br = at(r + 1, c + 1), bl = at(r + 1, c);
      const mask = (tl > 0 ? 8 : 0) | (tr > 0 ? 4 : 0) | (br > 0 ? 2 : 0) | (bl > 0 ? 1 : 0);
      if (mask === 0 || mask === 15) continue;

      const top = [c + cross(tl, tr), r], right = [c + 1, r + cross(tr, br)];
      const bottom = [c + cross(bl, br), r + 1], left = [c, r + cross(tl, bl)];
      const pairs = {
        1: [left, bottom], 2: [bottom, right], 3: [left, right], 4: [top, right],
        5: [left, top, bottom, right], 6: [top, bottom], 7: [left, top], 8: [left, top],
        9: [top, bottom], 10: [left, bottom, top, right], 11: [top, right], 12: [left, right],
        13: [bottom, right], 14: [left, bottom],
      }[mask];

      const px = (v) => (v / (n - 1)) * size;
      const py = (v) => size - (v / (n - 1)) * size;
      for (let k = 0; k < pairs.length; k += 2) {
        out.push([px(pairs[k][0]), py(pairs[k][1]), px(pairs[k + 1][0]), py(pairs[k + 1][1])]);
      }
    }
  }

  return out;
}

/** Where a class sits at a cell, 0..1 by confidence, already the right way up. */
export function sampler(field, confidence, n) {
  return (col, rowFromTop) => {
    const r = n - 1 - rowFromTop;
    const i = r * n + col;
    const sure = Math.max(0, (confidence[i] / 255 - 0.5) / 0.5);
    return { cls: field[i], sure };
  };
}
