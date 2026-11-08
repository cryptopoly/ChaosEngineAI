# Klingon (tlhIngan Hol)

ChaosEngineAI ships a Klingon interface, the first local AI app with one. It
is a fan translation: machine-assisted, written in Marc Okrand's Latin
romanisation, and open to correction. English stays for model names, file
formats, hardware terms (GPU, MLX, GGUF, CUDA), units, keyboard keys and
anything that is code.

## Pick it

**Settings > Language**, choose **tlhIngan Hol**. The app switches at once and
remembers the choice. A line under the dropdown says the translation is a
community one. Anything that has no Klingon string yet falls back to English.

Numbers, dates and file sizes keep English formatting: no platform ships
Klingon locale data, so the runtime uses the English rules.

Spelling is case-sensitive in Klingon (`q` and `Q` are different sounds;
`I`, `D`, `H`, `S` are always capital), so the interface turns off the
small-caps styling it uses elsewhere and never starts a sentence with a
capital.

## What it covers

Every screen: the sidebar, Chat, Models, Image, Video, Voice, Music, Projects,
Settings, the install and diagnostics panels, the native menu and tray, and
the backend's user-facing error messages. Voice stays English: there is no
Klingon speech model.

Technical terms were built from Okrand's vocabulary with Klingon grammar
where no word exists (a model is `yab`, a mind; a download is `lI'`, to
transmit data here; a project is `Qu'`, a mission). Every coined term is
listed in `docs/plans/klingon/glossary.md` with its source and a confidence
mark.

## Report a correction

- Open **Settings > Language** and use **Open tlh on GitHub**: it lands on
  `src/locales/tlh/`, one JSON file per area. Edit the string and open a pull
  request, or open an issue quoting the key (for example
  `common.actions.save`) and the better wording.
- For a term used everywhere (model, download, project), say so in the issue;
  one glossary change updates every string that uses it.
- Keep `{placeholders}` and `#` as they are. Inside a string that contains
  `{`, an apostrophe directly before `{`, `}`, `#` or `|` has to be written
  `''` (an ICU rule, not a Klingon one).

Qapla'.
