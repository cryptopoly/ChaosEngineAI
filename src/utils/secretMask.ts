/**
 * Keys on screen are masked by default (overnight review, stream F): code samples, test commands and the key
 * fields show dots and at most the last 4 characters until the user presses Reveal. Copy always copies the real
 * text, so only what is drawn goes through these helpers.
 */

const DOTS = "•".repeat(8);

/** Display form of a secret: a fixed run of dots plus its last 4 characters (only dots for a short secret). */
export function maskSecret(secret: string): string {
  return secret.length > 8 ? `${DOTS}${secret.slice(-4)}` : DOTS;
}

/** ``text`` with every occurrence of each given secret replaced by its masked form. Empty secrets are skipped. */
export function maskSecretsIn(text: string, secrets: ReadonlyArray<string | null | undefined>): string {
  let out = text;
  // Longest first, so a secret that contains another is masked whole.
  const ordered = secrets.filter((s): s is string => typeof s === "string" && s.length > 0).sort((a, b) => b.length - a.length);
  for (const secret of ordered) out = out.split(secret).join(maskSecret(secret));
  return out;
}
