/*
 * How the observations move.
 *
 * Every observation is a body with a position and a velocity, and every one of them collides
 * with the others and with the walls. What a choreography supplies is either a *slot* to be
 * sprung towards, or a *force* to be pushed by — never a position to be teleported to. That one
 * decision is what makes the whole thing feel like one physical world: a slot can be shoved out
 * of, a thrown observation has to barge its way back through the crowd, and dragging one across
 * the screen ploughs a furrow through the rest.
 *
 * It also removed a seam. A dropped observation used to be handed to a scripted walk home,
 * separate from the physics and tuned by hand; now the throw and the return are the same spring
 * and the same collisions, so the transition between them is not animated, it just happens.
 *
 * ENERGY. Every choreography has a *pace*: the speed its bodies move at when nobody is touching
 * them. Collisions are elastic and walls give everything back, so the only thing that ever takes
 * speed away is the distance from that pace. Above it, the excess decays exponentially and the
 * decay goes to nothing exactly as the pace is reached — so a hard throw is kept in full, spends
 * itself over several seconds, and lands back in the choreography without ever being stopped.
 * Below the pace nothing is taken, so a table of elastic balls keeps rolling indefinitely.
 *
 * HEAT. Every body also carries a heat: how much of a throw it is still living through. Heat
 * rises the instant something fast happens to a body and then fades on a clock of its own, and
 * the slot's authority is the cube of what is left of it. That is deliberately *not* a function
 * of the body's current speed. Reading the speed made the handover a threshold: the body ran on
 * at very nearly its thrown speed until it crossed a number, and then braked and steered at once
 * — exactly the seam the physics exists to avoid. A clock cannot do that.
 *
 * Splitting the two is what makes it read. Speed is spent by the pace, briskly and visibly, over
 * about six seconds; the slot's authority comes back on the heat's own clock over about twelve.
 * So a thrown observation is unmistakably slowing down from the first frame and is still nowhere
 * near home when it has finished doing so, and it drifts the rest of the way on a curve that
 * tightens the whole time. Tying both to one number gave a two-and-a-half second event that
 * looked like a throw followed by a snap.
 *
 * BOUNDS. Positions live in a square [-1, 1] world mapped into the part of that square the
 * viewport can actually show — the drawing is a square cropped to cover, so a wide screen throws
 * the top and bottom away. The mapping follows the viewport's shape and never its size, so
 * zooming the page leaves the animation exactly where it was.
 */

const N = 18;
const HALF = N / 2;
const EDGE = 0.94;               // how close to the visible edge anything is allowed

const R = 0.034;                 // the space a ball keeps to itself
const BOUNCE = 0.995;            // how much of a collision comes back out of it
const SPRING = 10;               // how hard a slot pulls its observation
const DAMP = 4.6;                // and how much of the swing it takes out
const TOP = 4;                   // the only cap left, and it is there to stop tunnelling
const SUBSTEPS = 4;              // so nothing tunnels through anything at that speed

const PACE = 0.12;               // the speed a slot motion settles back to
const BLEED = 0.55;              // how fast whatever is carried above the pace is shed

const COOL = 5;                  // seconds for a thrown body to lose a third of its heat
const GRIP = 3;                  // and how late in the cooling its slot really takes hold
const HOT_AT = 1.4;              // the speed above which something has clearly been thrown
const HOT_SPAN = 2.2;            // and the span over which it counts as fully thrown

const frac = (v) => v - Math.floor(v);
const rand = (i) => frac(Math.sin(i * 12.9898) * 43758.5453);
const clamp01 = (u) => (u < 0 ? 0 : u > 1 ? 1 : u);

/* ------------------------------------------------------------------ written patterns */

/** Lobes evenly spaced round a circle, classes alternating, the whole thing turning. */
const lobes = (count, spin = 0.1, orbit = 0.56, spread = 0.15) => ({
  labels: Array.from({ length: N }, (_, i) => (i % count) % 2),
  at(i, t) {
    const a = t * spin + ((i % count) * 2 * Math.PI) / count;
    const off = rand(i + 1) * Math.PI * 2;
    const r = Math.sqrt(rand(i + 97)) * spread;
    const w = t * 0.6 + rand(i + 41) * Math.PI * 2;
    return [Math.cos(a) * orbit + Math.cos(off) * r + Math.cos(w) * 0.02,
            Math.sin(a) * orbit + Math.sin(off) * r + Math.sin(w * 1.3) * 0.02];
  },
});

/**
 * One class inside the other, breathing.
 *
 * Exactly two quantities move, and both are plain sinusoids a quarter of a cycle apart: the
 * middle radius the pair sits at, and the gap between them. Each ring's radius is a different
 * mixture of the two, so the two rings are visibly linked and never in step.
 *
 * The sinusoid is the point. An eased breath — anything that lingers at its extremes and hurries
 * between them — reads as a phase that starts and finishes, and with two of them the ring looked
 * like a sequence of fades rather than one continuous motion. A sine has no such moment: its
 * speed changes everywhere and stops nowhere, so there is no phase to see.
 *
 * Driving the gap instead of the radii is also what makes crossing impossible: the inner ring
 * cannot pass the outer one, because the outer one is defined as the inner one plus a positive
 * number.
 */
const BREATH = 16;               // seconds; the cycle it closes on

const annulus = {
  labels: Array.from({ length: N }, (_, i) => (i < HALF ? 0 : 1)),
  at(i, t) {
    const u = (2 * Math.PI * t) / BREATH;
    const gap = 0.34 + Math.sin(u) * 0.18;              // 0.16 out to 0.52
    const mid = 0.53 + Math.cos(u) * 0.13;              // 0.40 out to 0.66, a quarter behind

    const inner = i < HALF;
    const k = inner ? i : i - HALF;
    const a = (k / HALF) * Math.PI * 2 + t * (inner ? 0.16 : -0.09) + rand(i) * 0.4;

    const r = mid + (inner ? -gap : gap) / 2;
    return [Math.cos(a) * r, Math.sin(a) * r];
  },
};

/** Two waves in step and close together, so the cut has to become the wave. */
const wave = {
  labels: Array.from({ length: N }, (_, i) => i % 2),
  at(i, t) {
    const s = i % 2;
    const k = Math.floor(i / 2) / (HALF - 1);
    const x = -0.88 + k * 1.76;
    return [x, Math.sin(x * 2.3 + t * 0.42) * 0.4 + (s ? 0.17 : -0.17)];
  },
};

/**
 * Two arms, and differential rotation.
 *
 * Every observation turns at its own constant rate, the outermost at exactly twice the pace of
 * the innermost, and that is the entire mechanism. Nine radii that begin on one line wind
 * themselves into a coil at one extra turn a minute, for as long as the page is open. There is
 * no ramp in it, no easing, no reversal and no second motion: the rates are the same in the ten
 * thousandth frame as in the first. Every version that varied them read as a figure running
 * down — which is what a rate that falls is, however smooth the curve that falls it.
 *
 * The radii are evenly spaced, which is what an Archimedean spiral is, and which is also the
 * only spacing eighteen observations can hold while they are still on one line. That constraint
 * is tighter than it looks: a straight row of eighteen does not fit the height of a wide screen
 * at all, so the coil has to have wound measurably before the row first turns through the
 * vertical, sixteen seconds in. Twice the rate at the rim is enough — by then the outer end
 * leads the inner by a radian, and no two observations are on the same ray again.
 *
 * The one thing it does not take from the viewport is its shape. The figure is drawn three parts
 * tall to four parts wide rather than to the screen's own squash, because a spiral reads as a
 * spiral only once it is round enough to have a centre, and nine sixteenths is not. The scale is
 * capped so the outermost ring never leaves the screen, which on a viewport too letterboxed to
 * hold a 3:4 figure quietly gives back as much of it as fits.
 */
const TURN_CORE = 0.1;           // the innermost rate — the pace the rosette turns at
const TURN_RIM = 0.2;            // and the outermost, exactly twice it
const ROUND = 0.75;              // three parts tall to four parts wide
const SPAN = [0.05, 0.7];        // innermost and outermost radius

const spiral = {
  labels: Array.from({ length: N }, (_, i) => i % 2),
  at(i, t, box) {
    const u = Math.floor(i / 2) / (HALF - 1);
    const r = SPAN[0] + u * (SPAN[1] - SPAN[0]);
    const a = (TURN_CORE + u * (TURN_RIM - TURN_CORE)) * t + (i % 2) * Math.PI;
    const tall = Math.min((ROUND * box.ex) / box.ey, EDGE / SPAN[1]);
    return [Math.cos(a) * r, Math.sin(a) * r * tall];
  },
  // A winding cut is a harder thing to find than a straight one, so this one arrives at nearly
  // three times the rate the rest of the set uses. It needs no longer to do it in.
  warm: 0.003,
};

/* ------------------------------------------------------------------ forces */

/**
 * A break, and then the table.
 *
 * No slot and no home: eighteen balls rolling, striking each other and coming off the cushions.
 * The only thing keeping it from being a mess is a slight distaste between the two colours,
 * enough to let clusters form and not enough to stop anything crossing the table.
 *
 * It breaks fully mixed, and that is the piece. A table that opens already sorted hands the
 * network a straight line on the first frame and there is nothing to watch; opening shuffled
 * means the cut starts as a tangle and is dragged into shape by the balls themselves over the
 * first half-minute.
 *
 * Nothing here damps. The cushions and the balls are elastic, so a ball thrown hard keeps its
 * speed and its line through every bounce, and the only thing that brings it down is the pace.
 * Below the pace nothing is taken at all, which is why the table is still rolling at the same
 * speed five minutes in — an elastic table does not need winding up, and winding it up was
 * measurably worse: it kept the two colours mixed faster than their mutual distaste could sort
 * them, and the screen fell from a hundred per cent to sixty.
 */
const billiards = {
  labels: Array.from({ length: N }, (_, i) => (i < HALF ? 0 : 1)),
  start: (i, box) => ({
    x: (rand(i + 5) - 0.5) * 1.7 * box.ex,
    y: (rand(i + 31) - 0.5) * 1.7 * box.ey,
    vx: (rand(i + 61) - 0.5) * 0.34, vy: (rand(i + 83) - 0.5) * 0.34,
  }),
  force(b, i, t, labels) {
    let fx = 0, fy = 0;
    for (let j = 0; j < N; j++) {
      if (j === i || labels[j] === labels[i]) continue;
      const dx = b[i].x - b[j].x, dy = b[i].y - b[j].y;
      const d = Math.hypot(dx, dy);
      if (d > 0.55 || d < 1e-4) continue;
      const f = 0.07 * (1 - d / 0.55);
      fx += (dx / d) * f; fy += (dy / d) * f;
    }
    return [fx, fy];
  },
  pace: 0.26,
};

/**
 * One floor, and everything bounces on it.
 *
 * The sheet is perfectly elastic at the height the choreography is meant to run at: a ball
 * arriving at its own pace leaves at exactly that pace, so it bounces to three quarters of the
 * screen and keeps doing it for as long as the page is open. Anything thrown in on top of that
 * is handed back a little short each time it lands, and anything a collision has taken out is
 * handed back a little long, so the whole rack converges on one height from either side.
 *
 * The two classes end up separated across the screen rather than stacked on decks of their own.
 * That matters: with one floor and the classes interleaved along it, the label is the parity of
 * a ball's column and no line can find it — the screen reads sixty-six per cent, which is what
 * put two decks there in the first place. Two columns with a corridor between them is a
 * vertical cut, and it reads a hundred.
 *
 * But the columns are not where it starts. Everything opens shuffled across the whole floor, and
 * the only thing that sorts it is a horizontal distaste between the colours — each ball shoved
 * sideways by every ball of the other colour near it. So the cut opens as a tangle it cannot
 * win, and the balls spend the first half-minute arranging the problem into one it can. Watching
 * that happen is the piece; a screen that opens already sorted has nothing in it.
 */
const G = 0.24;                  // the pull

/**
 * Which lane each observation opens in: evenly spaced across the floor, shuffled, and even.
 *
 * Even spacing so eighteen balls all have room on the first frame. Shuffled so the two colours
 * open interleaved — seeding the lanes from the world's own hash did that job worse, leaving
 * gaps a ball wide and stretches of one colour four long, which is already half-sorted. And
 * shuffled *afresh on every load*, because one fixed seed is not an unbiased arrangement, it is
 * one arrangement: that seed happened to stack the greens on the right and the blues on the
 * left, and it did so every single time the page opened.
 *
 * A fair shuffle alone still lands lopsided about a third of the time, and lopsided is the one
 * thing this opening cannot be, so the draw is rejected until each half of the floor holds four
 * or five of each colour.
 *
 * It is also rejected for being *too* well stirred. With nothing pushing the colours apart any
 * more, the shuffle is the whole problem: the cut has to be a comb with one tooth per run of
 * colour, and eighteen lanes that change colour ten times need a comb no smooth boundary will
 * draw — measured across eight loads, that reads between 57 and 83 per cent. Five to seven
 * changes is three or four runs of each colour, which is unmistakably interleaved and still
 * something a boundary can hold: the same eight loads read between 89 and 100.
 */
function lanes() {
  const order = Array.from({ length: N }, (_, i) => i);
  for (let attempt = 0; attempt < 64; attempt++) {
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    const left = order.slice(0, HALF).filter((i) => i % 2 === 0).length;
    let swaps = 0;
    for (let k = 1; k < N; k++) if (order[k] % 2 !== order[k - 1] % 2) swaps++;
    if (left >= 4 && left <= 5 && swaps >= 5 && swaps <= 7) break;
  }
  const lane = new Array(N);
  order.forEach((i, slot) => { lane[i] = slot; });
  return lane;
}

const gravity = {
  labels: Array.from({ length: N }, (_, i) => i % 2),
  /** Every body at once, because they have to agree on a shuffle before any of them has a lane. */
  open(box) {
    const lane = lanes();
    const ey = box.ey * EDGE;
    const v0 = Math.sqrt(3 * G * ey);          // the speed that reaches three quarters of the way up
    const rise = (v0 * v0) / (2 * G);

    return Array.from({ length: N }, (_, i) => {
      // already mid-flight, each at its own point on the same arc, so nothing falls into place
      const y = -ey + rand(i + 29) * rise * 0.96;
      const vy = Math.sqrt(Math.max(0, v0 * v0 - 2 * G * (y + ey))) * (rand(i + 47) < 0.5 ? -1 : 1);
      return { x: (-0.9 + (lane[i] / (N - 1)) * 1.8) * box.ex, y, vx: 0, vy };
    });
  },
  /**
   * Fall, and nothing else.
   *
   * The colours have no opinion about each other here, which is the one thing that separates
   * this choreography from every other pair of classes in the set. An earlier version filed
   * them apart with a sideways shove and it worked — twenty-five seconds and the greens were on
   * one side and the blues on the other — but a screen that has resolved into two blocks has
   * stopped being about anything. The horizontal drag only keeps each ball in the lane it
   * opened in, so the arrangement the shuffle deals is the arrangement for as long as the page
   * is open, and the cut has to be the comb that fits it.
   */
  force: (b, i) => [-b[i].vx * 0.35, -G],
  pace: (box) => Math.sqrt(3 * G * box.ey * EDGE),
  restore: 0.62,                 // how much of the difference the sheet keeps on each landing
};

/**
 * Two clusters that detonate.
 *
 * They gather, and every twenty seconds they blow apart along whatever direction each one
 * happens to be facing, ricochet off each other and off the walls, and are drawn back in. The
 * first version pushed everything away from the centre of the *screen* rather than out of its
 * own cluster, which is why it drifted outwards and never came home.
 *
 * The detonation is the one event in this world that is allowed to be an event: a quarter of a
 * second of hard shove, and then nothing gated on the clock ever again. Everything after it is
 * the same two clocks a thrown observation gets. The blast puts the whole cluster over the heat
 * threshold, so the pace spends its speed briskly and visibly while the gather and the ring-down
 * come back as the cube of what the heat has left — over about twelve seconds, which is why the
 * cycle had to grow from twelve to twenty to hold it.
 *
 * The version this replaces switched the ring-down on at six tenths of a second and the gather
 * at one and six, both as steps from nothing to full. That is the same seam as a throw that
 * runs at speed and then brakes and steers at once, and it looked like it.
 */
const CYCLE = 20;                // seconds between detonations
const BLAST = 4.2;                 // how hard the charge shoves
const GATHER = 1.7;              // how hard a cluster draws itself back together
const CALM = 2.1;                // and the ring-down that stops it oscillating when it gets there
const DRIFT = 0.18;              // the pace a swarm halfway home already carries
const shoal = {
  labels: Array.from({ length: N }, (_, i) => i % 2),
  /**
   * Scattered across the whole field, and already drifting home.
   *
   * Every body at once, because the drift is towards its own colour's centre of mass and no
   * single body knows where that is. The drift is the point: opening from rest meant the charge
   * that fires on the first frame had nothing to cancel, and the page began a quarter faster
   * than any detonation after it — 1.35 against 1.06. Started at the pace a swarm halfway home
   * already carries, the first charge lands on the same state as the twentieth: 0.90 against
   * 0.96. The drift is retuned whenever the charge is, because the two are read against each
   * other and only their agreement matters.
   */
  open(box) {
    const at = Array.from({ length: N }, (_, i) => [
      (rand(i + 3) - 0.5) * 1.7 * box.ex,
      (rand(i + 23) - 0.5) * 1.7 * box.ey,
    ]);
    const sum = [[0, 0], [0, 0]], n = [0, 0];
    at.forEach(([x, y], i) => { const c = i % 2; sum[c][0] += x; sum[c][1] += y; n[c]++; });
    const home = sum.map(([x, y], k) => [x / (n[k] || 1), y / (n[k] || 1)]);

    return at.map(([x, y], i) => {
      const [hx, hy] = home[i % 2];
      const dx = hx - x, dy = hy - y, d = Math.hypot(dx, dy) || 1;
      return { x, y, vx: (dx / d) * DRIFT, vy: (dy / d) * DRIFT };
    });
  },
  force(b, i, t, labels, grip) {
    const m = [[0, 0, 0], [0, 0, 0]];
    for (let j = 0; j < N; j++) { const c = labels[j]; m[c][0] += b[j].x; m[c][1] += b[j].y; m[c][2]++; }
    const home = m.map(([hx, hy, n]) => [hx / (n || 1), hy / (n || 1)]);

    const c = labels[i];
    const [hx, hy] = home[c];
    const dx = b[i].x - hx, dy = b[i].y - hy;
    const d = Math.hypot(dx, dy) || 1;

    /*
     * The detonation: a shove outward from its own cluster, for a quarter of a second.
     *
     * The charge is soft and the cooling is very long, and those are now independent numbers —
     * see `ignite` below. Together they give a recovery that takes half again as long as the
     * cycle it lives in, so the charge always lands on a swarm that is still drawing itself
     * together and never on one that has finished. Measured as the mean distance from each
     * observation to its own cluster's centre: 0.54 at the top of the scatter, 0.32 when the
     * next charge goes off, and 0.07 if it were ever left alone long enough to close. The
     * cluster is a thing this piece is always halfway towards and never arrives at.
     */
    const blast = frac(t / CYCLE) * CYCLE < 0.25 ? BLAST : 0;

    // the pull home and the ring-down, both on the heat's clock rather than on the wall's
    const gather = GATHER * grip;
    const calm = CALM * grip;

    return [(dx / d) * blast - dx * gather - b[i].vx * calm,
            (dy / d) * blast - dy * gather - b[i].vy * calm];
  },
  /**
   * The charge lights every observation's heat itself rather than earning it through speed.
   *
   * That is what lets a soft charge have a long recovery. Read off the speed, the two are one
   * number: half the charge is half the heat, the gather comes straight back, and a gentler
   * detonation is simply over sooner. Lit directly, the charge sets how far the swarm goes and
   * the cooling sets how long it takes to come back, and neither touches the other.
   */
  ignite: (t) => (frac(t / CYCLE) * CYCLE < 0.25 ? 1 : 0),
  pace: 0.35,
  cool: 30,
  bleed: 0.09,
};

/* ------------------------------------------------------------------ the set */

export const MOTIONS = [
  { id: "rosette", name: "Roseta", note: "Seis grupos alternos. Ni una recta ni una banda los separan.", ...lobes(6) },
  { id: "quartet", name: "Cuarteto", note: "Cuatro grupos. Una sola franja basta, y se ve por qué.", ...lobes(4, 0.13) },
  { id: "annulus", name: "Anillo", note: "Uno dentro del otro. El par sube y baja mientras el hueco se abre y se cierra.", ...annulus },
  { id: "wave", name: "Ola", note: "Dos olas en fase y muy juntas. El corte tiene que ser la ola.", ...wave },
  { id: "spiral", name: "Espiral", note: "Dos brazos salidos del centro, apretados dentro y abiertos fuera.", ...spiral },
  { id: "billiards", name: "Billar", note: "Bandas, carambolas, y un ligero desagrado entre colores.", ...billiards },
  { id: "gravity", name: "Cama Elástica", note: "Un suelo para todos. Cada bola vuelve a su altura por fuerte que la lances.", ...gravity },
  { id: "shoal", name: "Detonación", note: "Dos grupos que estallan cada doce segundos y vuelven a juntarse.", ...shoal },
];

export const motionById = (id) => MOTIONS.find((m) => m.id === id);
export const randomMotion = () => MOTIONS[Math.floor(Math.random() * MOTIONS.length)];

/**
 * One physical world, running whichever choreography it was given.
 *
 * A held observation is kinematic: it goes exactly where the pointer says and is treated as
 * immovable by everything it runs into, so dragging one shoulders the others out of the way.
 */
export function system(motion, bounds) {
  const labels = motion.labels.slice();
  const positions = Array.from({ length: N }, () => [0, 0]);
  const heat = new Float64Array(N);          // how much of a throw each body is still living through

  // Everything opens where the choreography wants it, not at the origin waiting to be sprung
  // there. A page that arrives already in formation is the whole difference between a drawing
  // and a drawing that has just been switched on.
  const b = motion.open ? motion.open(bounds) : Array.from({ length: N }, (_, i) => {
    if (motion.start) return motion.start(i, bounds);
    const [ux, uy] = motion.at.call(motion, i, 0, bounds);
    return { x: ux * bounds.ex, y: uy * bounds.ey, vx: 0, vy: 0 };
  });

  let box = bounds;
  let held = null;

  const wall = () => [box.ex * EDGE, box.ey * EDGE];
  const paceOf = () =>
    typeof motion.pace === "function" ? motion.pace(box) : motion.pace ?? PACE;

  /** Push any two that overlap apart, and trade the speed along the line between them. */
  function collide() {
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = b[j].x - b[i].x, dy = b[j].y - b[i].y;
        const d = Math.hypot(dx, dy);
        if (d >= R * 2 || d < 1e-6) continue;

        const nx = dx / d, ny = dy / d;
        const overlap = R * 2 - d;
        const iHeld = held?.index === i, jHeld = held?.index === j;
        // whoever is under the pointer does not budge; the other one gives all the ground
        const wi = iHeld ? 0 : jHeld ? 1 : 0.5;
        const wj = 1 - wi;

        b[i].x -= nx * overlap * wi; b[i].y -= ny * overlap * wi;
        b[j].x += nx * overlap * wj; b[j].y += ny * overlap * wj;

        const rel = (b[j].vx - b[i].vx) * nx + (b[j].vy - b[i].vy) * ny;
        if (rel > 0) continue;
        const imp = -(1 + BOUNCE) * rel;
        if (iHeld) { b[j].vx += imp * nx; b[j].vy += imp * ny; }
        else if (jHeld) { b[i].vx -= imp * nx; b[i].vy -= imp * ny; }
        else {
          b[i].vx -= (imp / 2) * nx; b[i].vy -= (imp / 2) * ny;
          b[j].vx += (imp / 2) * nx; b[j].vy += (imp / 2) * ny;
        }
      }
    }
  }

  /** Elastic on every side. The floor is the one that can also be a trampoline. */
  function walls(pace) {
    const [ex, ey] = wall();
    for (let i = 0; i < N; i++) {
      const p = b[i];
      if (p.x < -ex) { p.x = -ex; p.vx = Math.abs(p.vx); }
      if (p.x > ex) { p.x = ex; p.vx = -Math.abs(p.vx); }
      if (p.y > ey) { p.y = ey; p.vy = -Math.abs(p.vy); }
      if (p.y < -ey) {
        p.y = -ey;
        const up = Math.abs(p.vy);
        p.vy = motion.restore != null ? pace + (up - pace) * motion.restore : up;
      }
    }
  }

  const api = {
    labels,
    positions,
    heldIndex: null,
    resize(next) { box = next; },

    advance(t, dt) {
      const h = Math.min(dt, 0.05) / SUBSTEPS;
      const pace = paceOf();
      const bleed = motion.bleed ?? BLEED;
      // a choreography may hold its heat longer than the default; the detonation does, because
      // its whole cycle is one recovery and the recovery has to fill it
      const cool = Math.exp(-h / (motion.cool ?? COOL));

      /*
       * And a choreography may light its own heat rather than earning it through speed.
       *
       * That separation is the whole reason a charge can be softened without the recovery
       * shortening to match. Read off the speed, the two are one number: half the charge is half
       * the heat, the gather comes straight back, and a gentler detonation is over sooner rather
       * than being gentler for longer. Lit directly, the charge sets how far the swarm goes and
       * the cooling sets how long it takes to come back, and neither touches the other.
       */
      const spark = motion.ignite ? motion.ignite(t) : 0;

      for (let s = 0; s < SUBSTEPS; s++) {
        for (let i = 0; i < N; i++) {
          if (held?.index === i) continue;
          const p = b[i];

          // heat rises the moment something fast happens and then fades on its own clock, and
          // the slot takes hold as the cube of what is left — so the first several seconds after
          // a throw are almost free, and the authority arrives without ever arriving at once
          const speed = Math.hypot(p.vx, p.vy);
          heat[i] = Math.max(heat[i] * cool, clamp01((speed - HOT_AT) / HOT_SPAN), spark);
          const rest = 1 - heat[i];
          const grip = rest * rest * rest;

          if (motion.at) {
            // The slot pulls and brakes with whatever authority the heat has left it. Both fade
            // in together from the first frame after a throw, so there is no moment at which the
            // body starts braking and no separate moment at which it starts steering.
            const [ux, uy] = motion.at.call(motion, i, t, box);
            const tx = ux * box.ex, ty = uy * box.ey;
            p.vx += ((tx - p.x) * SPRING - p.vx * DAMP) * grip * h;
            p.vy += ((ty - p.y) * SPRING - p.vy * DAMP) * grip * h;
          } else {
            const [fx, fy] = motion.force.call(motion, b, i, t, labels, grip);
            p.vx += fx * h;
            p.vy += fy * h;
          }

          // The pace is the floor speed decay runs down to, and the decay vanishes exactly as
          // it is reached, so there is no edge to feel. Below the pace nothing is taken at all.
          const v = Math.hypot(p.vx, p.vy);
          if (bleed > 0 && v > pace) {
            const k = (pace + (v - pace) * Math.exp(-bleed * h)) / v;
            p.vx *= k; p.vy *= k;
          }

          const w = Math.hypot(p.vx, p.vy);
          if (w > TOP) { p.vx = (p.vx / w) * TOP; p.vy = (p.vy / w) * TOP; }

          p.x += p.vx * h;
          p.y += p.vy * h;
        }

        if (held) { b[held.index].x = held.at[0]; b[held.index].y = held.at[1]; }
        collide();
        collide();
        walls(pace);
      }

      const [ex, ey] = wall();
      for (let i = 0; i < N; i++) {
        positions[i] = [Math.max(-ex, Math.min(ex, b[i].x)), Math.max(-ey, Math.min(ey, b[i].y))];
      }

      return positions;
    },

    hold(index, at) {
      const [ex, ey] = wall();
      held = { index, at: [Math.max(-ex, Math.min(ex, at[0])), Math.max(-ey, Math.min(ey, at[1]))] };
      b[index].vx = 0; b[index].vy = 0;
      heat[index] = 0;
      api.heldIndex = index;
    },

    moveTo(at) {
      if (!held) return;
      const [ex, ey] = wall();
      held.at = [Math.max(-ex, Math.min(ex, at[0])), Math.max(-ey, Math.min(ey, at[1]))];
    },

    /** Let go with the speed the pointer had, in full. The pace spends it, nothing clips it. */
    release(vx = 0, vy = 0) {
      if (!held) return;
      const p = b[held.index];
      const mag = Math.hypot(vx, vy);
      const speed = Math.min(TOP, mag);
      const dir = mag || 1;
      p.vx = (vx / dir) * speed;
      p.vy = (vy / dir) * speed;
      // a firm throw buys the full ten seconds of freedom; a gentle nudge buys almost none
      heat[held.index] = clamp01(speed / 1.8);
      held = null;
      api.heldIndex = null;
    },

    pick(x, y, within = 0.09) {
      let best = -1, near = within;
      positions.forEach(([px, py], i) => {
        const d = Math.hypot(px - x, py - y);
        if (d < near) { near = d; best = i; }
      });
      return best;
    },
  };

  return api;
}
