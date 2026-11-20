// Rewrites a single-file ONNX model as a small "graph only" model whose large
// weights are references (offset + length) back into the original file.
//
// Why: when a model with its weights inline is handed to the engine, the whole
// file is copied into the engine's 4 GB working memory several times over and
// a 1.7 GB model no longer fits. With weights as references, the engine pulls
// each one straight from the original file to the graphics chip instead.
//
// Only the protobuf structure is touched; no weight bytes are read or changed.

const WINDOW = 16 * 1024;

// Field numbers from the ONNX protobuf schema.
const MODEL_GRAPH = 7;
const GRAPH_INITIALIZER = 5;
const TENSOR_RAW_DATA = 9;
const TENSOR_EXTERNAL_DATA = 13;
const TENSOR_DATA_LOCATION = 14;
const LOCATION_EXTERNAL = 1;

class Cursor {
  /** @param {(at: number, length: number) => Uint8Array} readAt */
  constructor(readAt, size) {
    this.readAt = readAt;
    this.size = size;
    this.windowStart = 0;
    this.window = new Uint8Array(0);
  }

  byte(at) {
    if (at < this.windowStart || at >= this.windowStart + this.window.length) {
      this.windowStart = at;
      this.window = this.readAt(at, Math.min(WINDOW, this.size - at));
      if (this.window.length === 0) throw new Error('Model file ended unexpectedly.');
    }
    return this.window[at - this.windowStart];
  }

  /** @returns {[value: number, next: number]} */
  varint(at) {
    let value = 0;
    let scale = 1;
    for (let i = 0; i < 10; i++) {
      const b = this.byte(at + i);
      value += (b & 0x7f) * scale;
      if (b < 0x80) return [value, at + i + 1];
      scale *= 128;
    }
    throw new Error('Model file is not valid ONNX.');
  }

  bytes(at, length) {
    if (at >= this.windowStart && at + length <= this.windowStart + this.window.length) {
      return this.window.slice(at - this.windowStart, at - this.windowStart + length);
    }
    return this.readAt(at, length);
  }
}

function encodeVarint(value) {
  const out = [];
  let rest = value;
  while (rest >= 128) {
    out.push((rest % 128) | 0x80);
    rest = Math.floor(rest / 128);
  }
  out.push(rest);
  return Uint8Array.from(out);
}

function concat(chunks) {
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

const text = new TextEncoder();

function lengthDelimited(field, payload) {
  return concat([encodeVarint(field * 8 + 2), encodeVarint(payload.length), payload]);
}

function externalEntry(key, value) {
  const entry = concat([lengthDelimited(1, text.encode(key)), lengthDelimited(2, text.encode(String(value)))]);
  return lengthDelimited(TENSOR_EXTERNAL_DATA, entry);
}

// Walks the fields of one protobuf message in [start, end) and calls
// visit(field, wireType, fieldStart, payloadStart, payloadLength, next).
function walk(cursor, start, end, visit) {
  let at = start;
  while (at < end) {
    const fieldStart = at;
    let tag;
    [tag, at] = cursor.varint(at);
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    let payloadStart = at;
    let payloadLength = 0;
    if (wire === 0) {
      [, at] = cursor.varint(at);
    } else if (wire === 1) {
      at += 8;
    } else if (wire === 5) {
      at += 4;
    } else if (wire === 2) {
      [payloadLength, at] = cursor.varint(at);
      payloadStart = at;
      at += payloadLength;
    } else {
      throw new Error('Model file is not valid ONNX.');
    }
    if (at > end) throw new Error('Model file is not valid ONNX.');
    visit(field, wire, fieldStart, payloadStart, payloadLength, at);
  }
}

function rewriteTensor(cursor, start, end, location, threshold) {
  const parts = [];
  let moved = null;
  walk(cursor, start, end, (field, wire, fieldStart, payloadStart, payloadLength, next) => {
    if (field === TENSOR_RAW_DATA && wire === 2 && payloadLength >= threshold) {
      moved = { offset: payloadStart, length: payloadLength };
      return; // leave the bytes where they are
    }
    parts.push(cursor.bytes(fieldStart, next - fieldStart));
  });
  if (moved) {
    parts.push(externalEntry('location', location));
    parts.push(externalEntry('offset', moved.offset));
    parts.push(externalEntry('length', moved.length));
    parts.push(concat([encodeVarint(TENSOR_DATA_LOCATION * 8), encodeVarint(LOCATION_EXTERNAL)]));
  }
  return { bytes: concat(parts), moved: moved ? moved.length : 0 };
}

/**
 * @param {(at: number, length: number) => Uint8Array} readAt  random access to the model file
 * @param {number} size       model file size in bytes
 * @param {string} location   name the weights file will be mounted under
 * @param {number} [threshold] smallest weight (bytes) worth leaving in the file
 * @returns {{ graph: Uint8Array, tensors: number, weightBytes: number }}
 */
export function externalizeWeights(readAt, size, location, threshold = 1024) {
  const cursor = new Cursor(readAt, size);
  const modelParts = [];
  let tensors = 0;
  let weightBytes = 0;

  walk(cursor, 0, size, (field, wire, fieldStart, payloadStart, payloadLength, next) => {
    if (field !== MODEL_GRAPH || wire !== 2) {
      modelParts.push(cursor.bytes(fieldStart, next - fieldStart));
      return;
    }
    const graphParts = [];
    walk(cursor, payloadStart, payloadStart + payloadLength, (gField, gWire, gStart, gPayload, gLength, gNext) => {
      if (gField !== GRAPH_INITIALIZER || gWire !== 2) {
        graphParts.push(cursor.bytes(gStart, gNext - gStart));
        return;
      }
      const tensor = rewriteTensor(cursor, gPayload, gPayload + gLength, location, threshold);
      if (tensor.moved) {
        tensors += 1;
        weightBytes += tensor.moved;
      }
      graphParts.push(lengthDelimited(GRAPH_INITIALIZER, tensor.bytes));
    });
    modelParts.push(lengthDelimited(MODEL_GRAPH, concat(graphParts)));
  });

  return { graph: concat(modelParts), tensors, weightBytes };
}
