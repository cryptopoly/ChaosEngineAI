/**
 * A short, built-in list of well-known artist and band names, used only to show a non-blocking hint when
 * the "Describe the music" text names one ("Naming an artist can produce a copy of their style").
 *
 * This is a HEURISTIC on the client, not a rights check: it misses most artists, it can never be
 * complete, and it must never block a run. The plan asked for a backend check; there is none yet, so
 * this list stands in. Names that are also everyday words (Prince, Queen, Sting, Drake, Seal, Cher, Muse,
 * Genesis, Oasis) are left out on purpose, because a style prompt such as "queen of the night" must not trigger.
 * Keep entries lower case; matching is whole-word and case-insensitive.
 */
export const KNOWN_ARTIST_NAMES: readonly string[] = [
  "taylor swift",
  "beyonce",
  "beyoncé",
  "ed sheeran",
  "adele",
  "rihanna",
  "madonna",
  "eminem",
  "kanye west",
  "jay-z",
  "kendrick lamar",
  "billie eilish",
  "ariana grande",
  "dua lipa",
  "the weeknd",
  "bruno mars",
  "lady gaga",
  "katy perry",
  "justin bieber",
  "post malone",
  "travis scott",
  "bad bunny",
  "harry styles",
  "olivia rodrigo",
  "lana del rey",
  "coldplay",
  "radiohead",
  "the beatles",
  "beatles",
  "rolling stones",
  "led zeppelin",
  "pink floyd",
  "david bowie",
  "elvis presley",
  "michael jackson",
  "frank sinatra",
  "bob dylan",
  "bob marley",
  "johnny cash",
  "metallica",
  "nirvana",
  "foo fighters",
  "green day",
  "linkin park",
  "daft punk",
  "avicii",
  "skrillex",
  "calvin harris",
  "david guetta",
  "deadmau5",
  "hans zimmer",
  "john williams",
  "ennio morricone",
  "ludovico einaudi",
  "billy joel",
  "elton john",
  "stevie wonder",
  "whitney houston",
  "mariah carey",
  "celine dion",
  "shakira",
  "nicki minaj",
  "cardi b",
  "snoop dogg",
  "dr. dre",
  "tupac",
  "notorious b.i.g.",
  "fleetwood mac",
  "abba",
  "bee gees",
  "u2",
  "ac/dc",
  "guns n' roses",
  "red hot chili peppers",
  "arctic monkeys",
  "tame impala",
  "the strokes",
  "kate bush",
];

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const MATCHERS: Array<{ name: string; pattern: RegExp }> = KNOWN_ARTIST_NAMES.map((name) => ({
  name,
  // Whole-word match: not preceded or followed by a letter or digit.
  pattern: new RegExp(`(?<![\\p{L}\\p{N}])${escapeRegExp(name)}(?![\\p{L}\\p{N}])`, "iu"),
}));

/** The first well-known artist name found in `text`, or null. */
export function findNamedArtist(text: string): string | null {
  if (!text) return null;
  for (const { name, pattern } of MATCHERS) {
    if (pattern.test(text)) return name;
  }
  return null;
}
