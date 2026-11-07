# Studio layout

Image Studio and Video Studio show what a first-time user needs and fold the rest away. This page explains
the layout and the sidebar.

## The default view

Both Studios open on the controls you use every time:

- **Model** and **Prompt**.
- An input-image picker where the model takes one (a reference image in Image Studio, a start image in Video
  Studio).
- The quality preset and the standard size buttons. Image Studio has an aspect row (1:1, 3:2, 2:3, 4:3, 3:4,
  16:9, 9:16, 21:9) and a size row (0.5, 1, 2, 4 and 8 MP). Video Studio has an aspect row and named
  resolutions (SD 480p up to 8K). A button the model cannot take is greyed out with the reason.
- The run estimate strip and **Generate**.

## Options

**Options** is a fold below the action. Its title line shows the current values, so you can see them without
opening it. It holds the things you change now and then:

- Image Studio: negative prompt, **Width** and **Height**, number of images and the seed.
- Video Studio: negative prompt, **Width** and **Height**, **Frames**, **FPS** and the seed.

## Expert

**Expert** holds the engine settings. Which ones you see depends on the model and your machine.

- Image Studio: **Steps**, **Guidance**, **Sampler**, **Diffusion cache**, **CFG decay**, **Preview VAE** and
  **FP8 layerwise** on supported NVIDIA cards.
- Video Studio: **Fast preview**, **Steps**, **Guidance**, **4-bit (NF4)**, the LTX two-stage upscale, **CFG
  decay**, **Preview VAE**, **FP8 layerwise**, **Diffusion cache** and **STG scale**.

Options and Expert remember whether you left them open, separately for each Studio.

## The run estimate strip

The strip sits directly above **Generate** and shows three things:

- **Time**, for example "About 40 s", or "Time: unknown until the first run".
- **Memory**, for example "Peak 9 GB of 64 GB", or "Memory: not estimated". The colour follows how close the
  peak is to your total memory.
- **Disk**, for example "Writes 12 MB".

It updates as you change a control. A chip says where the numbers come from:

- **Measured:** timed on this machine for this model and preset.
- **Estimated:** taken from the model card or a similar machine, not timed here yet.

When the memory check says a run is likely to crash the backend, the strip asks you to tick "Run anyway. I
accept that the backend may crash." If another job is running, the strip says so and when yours starts.

Music Studio and the Storyboard Mixer use the same strip.

## The Models button and More actions

The header has one **Models** button, which opens the Models page for that area. Image Studio also keeps
**Gallery**.

**More actions** holds the less common model actions: **Model card**, and loading or unloading the model
(**Preload Model** and **Unload Model** in Image Studio; **Load into memory** and **Unload** in Video
Studio), plus an unload entry for another loaded model.

## The sidebar

The sidebar groups the tabs under two small labels.

- **Make:** Images, Video, Voice, Music and Storyboard.
- **Run:** Chat, Benchmarks and Tools.

Dashboard stays first. Server, Logs and Settings stay last. Music and Storyboard appear once those areas are
switched on.

The sidebar is an accordion. Only the group you are working in stays open, so the list fits a small window.
To keep every group open, go to **Settings > Appearance** and turn on **Keep all sidebar groups open**. The
switch applies to the **Collapsible** sidebar style.
