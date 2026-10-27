/**
 * Cuts a reply that is still streaming into pieces that can be spoken now.
 *
 * A piece ends at the last sentence end or paragraph break seen so far that
 * is not inside an open code fence or thinking block, so speech can start
 * after the first sentence. The backend cleans each piece for speech
 * (markdown, links, code blocks) and splits it into sentences again.
 */

// Sentence end followed by a space and something that starts a sentence,
// CJK sentence punctuation, or a blank line.
const BOUNDARY_RE = /[.!?…]["')\]]*\s+(?=["'(\[]?[\p{Lu}\p{N}])|[。！？]["')\]」』]*|\n\s*\n/gu;

function openBlock(text: string): boolean {
  const fences = text.match(/```/g)?.length ?? 0;
  if (fences % 2 === 1) return true;
  const opens = text.match(/<think>/g)?.length ?? 0;
  const closes = text.match(/<\/think>/g)?.length ?? 0;
  return opens > closes;
}

export class SpeechTextTracker {
  private consumed = 0;

  /** New speakable pieces of ``text`` (the whole reply so far). */
  next(text: string, final = false): string[] {
    const pending = text.slice(this.consumed);
    if (final) {
      this.consumed = text.length;
      return pending.trim() ? [pending] : [];
    }
    let cut = -1;
    for (const match of pending.matchAll(BOUNDARY_RE)) {
      const end = (match.index ?? 0) + match[0].length;
      if (!openBlock(pending.slice(0, end))) cut = end;
    }
    if (cut <= 0) return [];
    this.consumed += cut;
    const piece = pending.slice(0, cut);
    return piece.trim() ? [piece] : [];
  }

  reset(): void {
    this.consumed = 0;
  }
}
