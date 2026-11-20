// CLIP byte-pair tokenizer, written to match the tokenizer that ships with the
// SD-Turbo model (vocab.json + merges.txt). Output is always 77 ids:
// <start> tokens... <end>, then padded with id 0.

const TOKEN_PATTERN =
  /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;

export const MAX_TOKENS = 77;
const START_ID = 49406;
const END_ID = 49407;
const PAD_ID = 0;

// CLIP runs byte-pair merges over UTF-8 bytes, with each byte shown as one
// printable character. This rebuilds that byte -> character table.
function buildByteEncoder() {
  const bytes = [];
  for (let b = 33; b <= 126; b++) bytes.push(b);
  for (let b = 161; b <= 172; b++) bytes.push(b);
  for (let b = 174; b <= 255; b++) bytes.push(b);
  const chars = bytes.slice();
  let extra = 0;
  for (let b = 0; b < 256; b++) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      chars.push(256 + extra);
      extra += 1;
    }
  }
  const table = new Array(256);
  bytes.forEach((b, i) => { table[b] = String.fromCodePoint(chars[i]); });
  return table;
}
const BYTE_ENCODER = buildByteEncoder();
const UTF8 = new TextEncoder();

export class ClipTokenizer {
  /**
   * @param {Record<string, number>} vocab  contents of vocab.json
   * @param {string} mergesText             contents of merges.txt
   */
  constructor(vocab, mergesText) {
    this.vocab = vocab;
    this.ranks = new Map();
    // First line is a version header; the model uses the first 48,894 merges.
    const lines = mergesText.trim().split('\n').slice(1, 49152 - 256 - 2 + 1);
    for (let i = 0; i < lines.length; i++) this.ranks.set(lines[i].trim(), i);
    this.cache = new Map();
  }

  bpe(token) {
    const hit = this.cache.get(token);
    if (hit) return hit;
    const chars = Array.from(UTF8.encode(token), (b) => BYTE_ENCODER[b]);
    let word = chars.slice(0, -1).concat(chars[chars.length - 1] + '</w>');

    while (word.length > 1) {
      let bestRank = Infinity;
      let bestIndex = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.ranks.get(word[i] + ' ' + word[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex < 0) break;
      const first = word[bestIndex];
      const second = word[bestIndex + 1];
      const merged = [];
      for (let i = 0; i < word.length; ) {
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
          merged.push(first + second);
          i += 2;
        } else {
          merged.push(word[i]);
          i += 1;
        }
      }
      word = merged;
    }
    this.cache.set(token, word);
    return word;
  }

  /** Token ids without start/end/padding. */
  encodeText(text) {
    const ids = [];
    // This model's tokenizer config registers "!" as its padding token, so the
    // reference pipeline maps every "!" in a prompt straight to id 0. Match it.
    const segments = String(text).normalize('NFC').split('!');
    segments.forEach((segment, index) => {
      if (index > 0) ids.push(PAD_ID);
      const clean = segment.replace(/\s+/g, ' ').trim().toLowerCase();
      for (const match of clean.matchAll(TOKEN_PATTERN)) {
        const piece = match[0];
        if (piece === '<|startoftext|>') { ids.push(START_ID); continue; }
        if (piece === '<|endoftext|>') { ids.push(END_ID); continue; }
        for (const part of this.bpe(piece)) {
          const id = this.vocab[part];
          ids.push(id === undefined ? END_ID : id);
        }
      }
    });
    return ids;
  }

  /**
   * @returns {{ ids: Int32Array, used: number, truncated: boolean }}
   * `used` counts prompt tokens, out of a possible 75.
   */
  encode(text) {
    const body = this.encodeText(text);
    const room = MAX_TOKENS - 2;
    const truncated = body.length > room;
    const kept = truncated ? body.slice(0, room) : body;
    const ids = new Int32Array(MAX_TOKENS).fill(PAD_ID);
    ids[0] = START_ID;
    ids.set(kept, 1);
    ids[kept.length + 1] = END_ID;
    return { ids, used: kept.length, truncated };
  }
}
