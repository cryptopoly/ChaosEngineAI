/**
 * Curated library of single-page HTML Challenge prompts, grouped into
 * four categories. Feeds the Option-C prompt picker (tabbed card grid +
 * search) in the HTML Challenge tab.
 *
 * Each entry carries a short ``summary`` for the card face and a full
 * ``prompt`` that lands in the challenge textarea on selection. Keep the
 * library balanced — every category should hold the same count so the
 * UI tab badges stay even.
 */

export type ChallengePromptCategoryId =
  | "games"
  | "simulations"
  | "tech-demos"
  | "creative-tools";

export interface ChallengePromptCategory {
  id: ChallengePromptCategoryId;
  /** Short tab label. */
  label: string;
  /** One-line description for the tab tooltip / header. */
  blurb: string;
}

export interface ChallengePrompt {
  id: string;
  title: string;
  category: ChallengePromptCategoryId;
  /** Trimmed one-liner shown on the card face. */
  summary: string;
  /** Full prompt text inserted into the challenge textarea. */
  prompt: string;
}

export const CHALLENGE_PROMPT_CATEGORIES: ChallengePromptCategory[] = [
  { id: "games", label: "Games", blurb: "Interactive games with a win/lose state." },
  { id: "simulations", label: "Simulations", blurb: "Emergent systems and physical models." },
  { id: "tech-demos", label: "Tech Demos", blurb: "Algorithm, graphics and audio showcases." },
  { id: "creative-tools", label: "Creative Tools", blurb: "Tools that produce art, audio or output." },
];

export const CHALLENGE_PROMPTS: ChallengePrompt[] = [
  // ---- Games -------------------------------------------------------------
  {
    id: "snake",
    title: "Snake Game",
    category: "games",
    summary: "Grid snake. Arrow keys + WASD, growing tail, high-score persistence.",
    prompt:
      "Grid-based snake on a single HTML page. 20x20 or 30x30 grid. Arrow keys + WASD to move. Snake grows when eating food (random cell, never on the snake body). Game over on wall hit or self-collision. Speed increases every 5 foods eaten. Display current score + high score (persisted via localStorage). Pause on spacebar. Start/restart button. Snake one colour, food contrasting, background dark.",
  },
  {
    id: "tetris",
    title: "Tetris",
    category: "games",
    summary: "Full tetromino game. 10x20 well, SRS rotation, next-piece, levels.",
    prompt:
      "Full tetromino game on a 10x20 playfield. All 7 pieces (I, O, T, S, Z, J, L) with SRS rotation rules. Arrow keys: left/right move, down soft drop, up rotate, space hard drop. Line clear on a full row, multi-line clears award bonus (Tetris = 4 lines). Next-piece preview box. Level + lines-cleared + score display. Speed increases every 10 lines. Optional hold-piece slot (C key). Game over when the stack reaches the top.",
  },
  {
    id: "pong",
    title: "Pong (2-player + AI)",
    category: "games",
    summary: "Two-paddle Pong with AI toggle, spin physics, first to 10.",
    prompt:
      "Classic two-paddle Pong. Left paddle = W/S, right paddle = arrow up/down. A toggle button switches the right paddle to AI. Ball physics: rebound angle depends on where it hits the paddle, ball speeds up after each paddle hit, resets to centre on a score. First to 10 wins. Score display top centre. Ball trail or particle effect on paddle hit. AI tracks the ball with a slight lag for fairness.",
  },
  {
    id: "flappy",
    title: "Flappy Bird Clone",
    category: "games",
    summary: "Tap to flap, gravity, scrolling pipes, death animation, high score.",
    prompt:
      "Side-scrolling bird game. Spacebar or click to flap (an upward impulse), gravity pulls the bird down. Pipes spawn from the right edge with random gap heights and scroll left at a constant speed. Score increments per pipe passed. Collision with a pipe or the ground = game over with a brief death animation (bird tumbles down). Current score + high score via localStorage. Restart button or click-to-restart. Optional day/night background cycle every 10 points.",
  },
  {
    id: "breakout",
    title: "Breakout / Arkanoid",
    category: "games",
    summary: "Paddle + ball, brick field, power-ups, 3 lives, particle bursts.",
    prompt:
      "Brick-breaker game. Paddle at the bottom controlled by mouse or left/right arrows. Ball bounces off the paddle, walls and bricks. Brick grid at the top (8 rows x 14 cols), different colours mapped to different point values. Ball angle on the paddle depends on hit position. 3 lives. Power-up drops (multi-ball, wider paddle, slow ball, sticky paddle) from roughly 10% of bricks. Win condition: all bricks cleared. Particle burst on brick break.",
  },
  {
    id: "2048",
    title: "2048 Game",
    category: "games",
    summary: "4x4 tile merge. Arrow/swipe, animated slides, undo, high score.",
    prompt:
      "Tile-merging puzzle. 4x4 grid. Arrow keys or swipe (touch + mouse drag) to slide tiles in one direction. Tiles with the same value merge into the doubled value (2+2=4, 4+4=8, etc.). A new random tile (2 or 4) spawns after each move. Win toast at the 2048 tile (game continues for higher tiles). Lose when the board is full and no moves are possible. Score = running sum of merged values. Animated slide + merge transitions. 1-deep undo button. High score in localStorage. Restart button.",
  },
  {
    id: "starfield",
    title: "Interactive Starfield with Spaceship",
    category: "games",
    summary: "Parallax stars, controllable ship, asteroid dodging, survival score.",
    prompt:
      "Side-on space scroller with parallax. 3 star layers move at different speeds. Spaceship at the left edge controlled with arrow keys (up/down for vertical, left/right for slight horizontal). Thrust effect when space is held (small flame + screen shake). Asteroids spawn from the right at random Y + size and scroll left. Collision = game over with explosion particles. Score = time survived. Speed ramps over time. Choose between a 3-hit shield bar or one-hit-kill mode. Ship rendered as simple triangle geometry.",
  },
  {
    id: "platformer",
    title: "Platformer Level (Mario-style)",
    category: "games",
    summary: "Jump physics, moving platforms, coins, enemy AI, goal flag.",
    prompt:
      "Single-screen 2D platformer. Player character with jump physics (variable height by jump-button-hold duration), gravity, ground friction and coyote-time forgiveness. Arrow keys + spacebar to jump. 5-8 static platforms + 1-2 moving platforms (horizontal or vertical patrol). Coins to collect with a counter + collect sound. 1-2 patrolling enemies that reverse at platform edges; jump-on-head kills an enemy + small bounce, side-touch kills the player. Goal flag at the right edge triggers a win. Lives counter. Death respawns at the start.",
  },

  // ---- Simulations -------------------------------------------------------
  {
    id: "game-of-life",
    title: "Conway's Game of Life",
    category: "simulations",
    summary: "Cellular automata sandbox. Draw cells, presets, speed control.",
    prompt:
      "Cellular automata sandbox. Grid roughly 80x60 rendered on canvas. Click/drag to toggle cells alive/dead. Play/pause button. Step-once button. Speed slider (1-60 generations/sec). Clear-all + random-fill buttons. Preset pattern dropdown: glider, glider gun, pulsar, R-pentomino, lightweight spaceship. Standard B3/S23 rules. Generation counter + live-cell counter. Toggleable wrap-around vs. dead-edge boundaries.",
  },
  {
    id: "boids",
    title: "Boids Flocking Simulation",
    category: "simulations",
    summary: "Separation/alignment/cohesion flock, mouse predator, tunable rules.",
    prompt:
      "Emergent flocking on canvas. 200-500 boids. Three rules per boid: separation (avoid close neighbours), alignment (match neighbour heading), cohesion (steer toward neighbour centroid). Sliders for each rule weight + neighbour radius + max speed. The mouse pointer acts as a predator (boids flee within a radius). Edge behaviour toggle: wrap-around vs. steer-back. Boids rendered as small triangles oriented to the velocity vector. Optional motion trails with alpha fade.",
  },
  {
    id: "physics-sandbox",
    title: "Physics Sandbox",
    category: "simulations",
    summary: "Spawn balls/boxes/springs, gravity + bounciness sliders, drag objects.",
    prompt:
      "Click-to-spawn 2D physics playground. Toolbar buttons: ball, box, static block, soft spring/rope. Click-drag to spawn with an initial velocity. Gravity on/off + magnitude slider. Bounciness + friction sliders. Click-and-drag existing objects with the mouse (rubber-band attach). Clear-all button. Verlet or simple Euler integrator with circle/AABB collision response. Object count + FPS counter. Right-click to delete an object.",
  },
  {
    id: "ecosystem",
    title: "Ecosystem / Predator-Prey Simulation",
    category: "simulations",
    summary: "Grass/rabbits/foxes, energy + reproduction, live population graph.",
    prompt:
      "Lotka-Volterra style simulation on a grid. Three entities: grass (regrows on empty tiles), rabbits (eat grass, reproduce when an energy threshold is reached), foxes (eat rabbits, reproduce when an energy threshold is reached). Each animal has energy, age and a vision radius. Sliders: grass regrow rate, rabbit reproduction threshold, fox reproduction threshold, rabbit vision, fox vision. Live population sparkline graph over time. Reset button. Display per-species population counts. Oscillations should emerge naturally.",
  },
  {
    id: "solar-system",
    title: "Solar System / N-body Gravity",
    category: "simulations",
    summary: "N-body orbits, click-drag to add bodies, trails, time-warp.",
    prompt:
      "2D orbital simulator on canvas. A central star + 3-5 starting planets at random distances and tangential velocities for roughly stable orbits. Universal gravitation between all bodies (Newton's law, configurable G). Click empty space, then drag, to spawn a new body with the drag vector as the initial velocity. Each body draws a fading trail of past N positions (slider). Pause/play + time-warp slider (0.1x to 10x). Display body count + total kinetic + potential energy. Reset-to-default button. Optional collision merging (larger absorbs smaller).",
  },
  {
    id: "fluid-sim",
    title: "2D Fluid Simulation",
    category: "simulations",
    summary: "Real-time fluid, drag to inject dye + velocity, viscosity sliders.",
    prompt:
      "Real-time fluid on canvas. Particle-based (SPH) or grid-based (Stam stable fluids). Click-drag injects velocity and coloured dye into the fluid. Right-click adds static obstacles. Viscosity + diffusion sliders. Toggle between a velocity-field arrow view and a dye-density view. Resolution slider (32x32 up to 128x128 cells). Clear-canvas button. Display FPS + active grid size. Boundary mode: closed walls or open / wrap-around.",
  },
  {
    id: "ant-colony",
    title: "Ant Colony Pheromone Foraging",
    category: "simulations",
    summary: "Ants lay pheromone trails to food, evaporation, shortest path emerges.",
    prompt:
      "Emergent shortest-path via pheromone trails. Roughly 200 ants emerge from a central nest. Multiple food sources placed by click. Ants random-walk until they hit food, then return to the nest depositing 'to-food' pheromone. Returning ants reverse and deposit 'to-nest' pheromone. Ants probabilistically bias their next step toward the relevant gradient. Pheromone evaporates over time. Trails drawn as fading colour overlays. Sliders: ant count, deposit rate, evaporation rate, random-turn probability. The shortest path should emerge.",
  },
  {
    id: "double-pendulum",
    title: "Double Pendulum Chaos",
    category: "simulations",
    summary: "Accurate two-link pendulum, fading tip trail, chaos swarm toggle.",
    prompt:
      "Two-link rigid pendulum suspended from a fixed pivot. Accurate equations of motion via a Lagrangian + RK4 integrator. Render the rods + bobs and draw a fading trail of the lower bob tip (last N positions). Sliders: rod 1 length, rod 2 length, mass 1, mass 2, gravity. Click-and-drag to set the initial angle of each bob. Play/pause/reset. Time-step slider. Trail coloured by velocity magnitude. Optional 'swarm' toggle: spawn 20 pendulums with starting angles offset by 0.001 rad so diverging trails show sensitive dependence on initial conditions.",
  },

  // ---- Tech Demos --------------------------------------------------------
  {
    id: "mandelbrot",
    title: "Mandelbrot Set Explorer",
    category: "tech-demos",
    summary: "Zoom/pan fractal, iteration slider, colour schemes, progressive render.",
    prompt:
      "Interactive fractal viewer. Render the Mandelbrot set on canvas with a smooth coloured escape-iteration map. Click-and-drag to pan, scroll wheel to zoom (centred on the cursor). Iteration limit slider (50-2000). Colour scheme selector: fire, ocean, grayscale, rainbow. Reset-view button. Display the current zoom magnitude + centre coordinates. Progressive render (low-res first, refine in passes) for responsiveness.",
  },
  {
    id: "fireworks",
    title: "Particle Fireworks / Explosion System",
    category: "tech-demos",
    summary: "Click to launch rockets, gravity particles, trails, auto-launch.",
    prompt:
      "Click-to-launch firework display. Click anywhere and a rocket trail ascends from the bottom, exploding at the target with 100-200 particles in random colour palettes. Particles are affected by gravity, fade alpha over their lifetime and leave optional motion trails. Multiple fireworks active simultaneously. Auto-launch toggle (random launches every 1-2 sec). Particle count slider. Dark night-sky background. Optional pop sound on explosion via a Web Audio noise burst.",
  },
  {
    id: "sorting-visualizer",
    title: "Sorting Algorithm Visualizer",
    category: "tech-demos",
    summary: "Animated bars, 6 algorithms, comparison/swap colours, counters.",
    prompt:
      "Animated sort comparison. Bar chart of roughly 80 random-height bars. Algorithm dropdown: bubble, selection, insertion, quick, merge, heap. Speed slider. Shuffle button to randomise. Start/pause. Highlight comparisons and swaps with colour (e.g. red = comparing, green = just-swapped, blue = sorted-final). Live counters: comparisons, swaps, elapsed steps. Optional audio: pitch maps to bar height on each swap.",
  },
  {
    id: "maze-pathfinder",
    title: "Procedural Maze Generator + Pathfinder",
    category: "tech-demos",
    summary: "Animated maze gen + A*/BFS solve, pick start/end, counters.",
    prompt:
      "Two-stage maze app. Stage 1: generate a maze on a grid using a selectable algorithm (recursive backtracker, Prim's, or Kruskal's). Animate generation step-by-step. Stage 2: click to pick start + end cells. A Solve button runs A* or BFS, animates explored cells then highlights the final path. Reset + regenerate buttons. Maze size slider (10x10 to 80x60). Animation speed slider. Counters: cells visited, path length, time elapsed.",
  },
  {
    id: "raycaster",
    title: "Raycaster Pseudo-3D Engine",
    category: "tech-demos",
    summary: "Wolfenstein-style FPS, per-column raycasting, WASD + mini-map.",
    prompt:
      "Wolfenstein-style first-person renderer. An internal top-down grid map (e.g. 16x16) with walls. Player position + heading angle. Cast one ray per screen column, compute the wall distance, and render a vertical strip with height inversely proportional to distance. Distance-based shading. WASD to move (forward/back/strafe), mouse-look or left/right arrows to rotate the view. Mini-map in a corner showing the map + player position + facing arrow. Different wall colours per side (N/S vs E/W). Optional textured walls.",
  },
  {
    id: "generative-art",
    title: "Generative Art Studio",
    category: "tech-demos",
    summary: "Flow-field / recursive trees / kaleidoscope modes, live sliders, export.",
    prompt:
      "Live-tweak generative visuals on canvas. Three selectable modes: flow-field (Perlin noise vectors with particle trails), recursive trees (L-system or fractal branch with branching angle + depth), kaleidoscope (radial-symmetry mirror of mouse-drawn strokes). Per-mode sliders: noise scale, particle count, branch angle, recursion depth, symmetry segments, colour palette presets. Randomise button. Export-to-PNG button. Auto-evolve toggle that slowly drifts parameters over time.",
  },
  {
    id: "wireframe-3d",
    title: "3D Wireframe Engine",
    category: "tech-demos",
    summary: "No-WebGL 3D: rotation matrices, perspective projection, shape picker.",
    prompt:
      "Real-time 3D rotating wireframe on canvas, no WebGL. Implement: a 3D vertex array, X/Y/Z rotation matrices, and perspective projection to 2D screen space. Render edges as lines. Shape selector: cube, tetrahedron, octahedron, torus, wireframe sphere (latitude/longitude bands). Auto-rotate with per-axis speed sliders. Mouse drag for manual rotation. FOV + camera-distance sliders. Vertex/edge counter. Optional hidden-line removal or simple flat shading toggle.",
  },
  {
    id: "spectrum-analyzer",
    title: "Audio Spectrum Analyzer",
    category: "tech-demos",
    summary: "Web Audio FFT, mic or file, bars/scope/radial/spectrogram modes.",
    prompt:
      "Real-time audio visualizer using a Web Audio AnalyserNode. Source switch: microphone (with a permission prompt) or an uploaded audio file (with playback controls). Display modes: frequency-bar spectrum, waveform oscilloscope, circular radial spectrum, spectrogram waterfall (history scroll). FFT size selector (256 to 4096). Smoothing slider. Colour scheme presets. Sensitivity/gain slider. Optional peak-hold indicator on bars. Pause + clear buttons.",
  },

  // ---- Creative Tools ----------------------------------------------------
  {
    id: "drum-machine",
    title: "Drum Machine / 16-step Sequencer",
    category: "creative-tools",
    summary: "8-track step grid, synthesized sounds, BPM, pattern save/load.",
    prompt:
      "Step sequencer using the Web Audio API. An 8-track grid: kick, snare, closed hi-hat, open hi-hat, clap, low tom, high tom, cymbal. 16 columns = 16 steps. Click cells to toggle. Play/stop button. BPM input (60-200). Per-track volume sliders. Save/load patterns to localStorage with named slots. Every sound is synthesized from oscillators + noise + envelopes - no sample files. The active step is highlighted during playback.",
  },
  {
    id: "pixel-art-editor",
    title: "Pixel Art Editor",
    category: "creative-tools",
    summary: "Grid canvas, brush/fill/eyedropper tools, undo/redo, PNG export.",
    prompt:
      "Drawing tool on a grid canvas. Canvas size selector (16x16, 32x32, 64x64, 128x128). Colour picker (HTML5 native input + recent-colours palette + a preset 16-colour palette). Brush sizes 1x1 to 4x4. Tools: pencil, eraser, fill bucket, eyedropper, line, rectangle outline + fill. Undo/redo stack (at least 20 steps). Export-to-PNG button (canvas toBlob, scaled up nearest-neighbour). Save/load named slots to localStorage. Grid lines toggle. Zoom controls.",
  },
  {
    id: "synth-keyboard",
    title: "Synthesizer Keyboard",
    category: "creative-tools",
    summary: "Playable QWERTY synth, waveform + ADSR + filter, polyphonic.",
    prompt:
      "Playable polyphonic synth via the Web Audio API. QWERTY keys mapped to chromatic notes (A-S-D-F-G-H-J = white keys, W-E-T-Y-U = sharps); Z/X shift the octave. On-screen piano keys are clickable and visually highlighted when held. Waveform selector (sine, square, sawtooth, triangle). ADSR envelope sliders (attack, decay, sustain, release). Low-pass filter cutoff + resonance sliders. LFO toggle with rate + depth + target (pitch or filter). Master volume. Optional delay/reverb send.",
  },
  {
    id: "whiteboard",
    title: "Whiteboard Sketch App",
    category: "creative-tools",
    summary: "Smoothed freehand pen, shapes, multi-page, undo/redo, PNG export.",
    prompt:
      "Freehand drawing tool. A pen tool with smoothing (Bezier interpolation across recent mouse points). Tools: pen, eraser, rectangle, ellipse, line, arrow, text. Colour picker + brush-size slider. Undo/redo stack. Multi-page support (next/prev buttons, pages stored in localStorage). Export the current page to PNG. Clear-page + delete-page buttons. Pan + zoom (mouse wheel + middle-drag or two-finger gesture). Optional layer system (background + drawing).",
  },
  {
    id: "markdown-editor",
    title: "Markdown Editor with Live Preview",
    category: "creative-tools",
    summary: "Split-pane editor, live HTML render, toolbar, export .md/.html.",
    prompt:
      "Split-pane editor. Left = a raw markdown textarea, right = a rendered HTML preview updating on keystroke. Support: headings, bold/italic, links, images, ordered + unordered lists, fenced code blocks with monospace styling, blockquotes, horizontal rules, tables. Toolbar buttons inserting common syntax at the cursor. Sync-scroll between panes. Save the document to localStorage with named slots. Export rendered HTML or a raw .md file via a download blob. Live word + character count.",
  },
  {
    id: "palette-generator",
    title: "Color Palette Generator",
    category: "creative-tools",
    summary: "Harmony schemes from a base colour, lockable swatches, CSS/JSON export.",
    prompt:
      "A harmonious colour scheme generator. Base colour picker (HTML5 native input + HSL sliders). Scheme dropdown: complementary, analogous, triadic, tetradic, monochromatic, split-complementary. Display 5-6 swatches showing hex + RGB + HSL values, each click-to-copy. Lock individual swatches and regenerate the rest. Random-palette button. Save a palette to localStorage with a name. Export as CSS variables, JSON, or a Tailwind config snippet. Optional gradient preview ribbon between swatches.",
  },
  {
    id: "mind-map",
    title: "Mind Map / Node Editor",
    category: "creative-tools",
    summary: "Create + connect nodes on canvas, edit labels, save/load, export.",
    prompt:
      "Visual graph editor on canvas. Double-click empty space to create a node with an editable text label. Click-drag from a node's edge to another node to create a connecting edge. Drag nodes to reposition. Single-click a node to edit its label, colour, or shape (rect/ellipse/diamond). Right-click for delete. Pan + zoom the canvas. Save/load named mind maps to localStorage. Export as PNG or JSON. Auto-layout button (simple force-directed or radial tree).",
  },
  {
    id: "ascii-art",
    title: "ASCII Art Converter",
    category: "creative-tools",
    summary: "Image to ASCII, brightness ramp, width + charset controls, copy/export.",
    prompt:
      "An image-to-ASCII tool. File drop or upload input (PNG/JPG). Convert the image to grayscale, sample blocks of pixels, and map each block to an ASCII character by brightness using a configurable ramp (default ' .:-=+*#%@'). Output width slider (40-200 chars). Character set selector (default ramp, block chars, custom user input). Invert toggle. Display the result in a monospace pre element. Copy-to-clipboard + download-as-.txt buttons. Optional colour mode (per-char colour sampled from the original pixel).",
  },
];

/**
 * Filter the library by category (``"all"`` for no category filter) and a
 * free-text query. The query matches title, summary and the full prompt
 * body so a user can search by mechanic ("pheromone", "FFT") not just name.
 */
export function filterChallengePrompts(
  category: ChallengePromptCategoryId | "all",
  query: string,
): ChallengePrompt[] {
  const normalized = query.trim().toLowerCase();
  return CHALLENGE_PROMPTS.filter((entry) => {
    if (category !== "all" && entry.category !== category) {
      return false;
    }
    if (!normalized) {
      return true;
    }
    return (
      entry.title.toLowerCase().includes(normalized) ||
      entry.summary.toLowerCase().includes(normalized) ||
      entry.prompt.toLowerCase().includes(normalized)
    );
  });
}

/** Count of prompts per category, used for the tab badges. */
export function challengePromptCountByCategory(): Record<ChallengePromptCategoryId, number> {
  const counts = { games: 0, simulations: 0, "tech-demos": 0, "creative-tools": 0 } as Record<
    ChallengePromptCategoryId,
    number
  >;
  for (const entry of CHALLENGE_PROMPTS) {
    counts[entry.category] += 1;
  }
  return counts;
}
