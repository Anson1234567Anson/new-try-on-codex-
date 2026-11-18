// Helpers for the "Compute and Mixed Precision" lesson: real rounding to
// 32-bit, 16-bit (half) and bfloat16 floating point, and their bit patterns.

const buf = new DataView(new ArrayBuffer(4));

/** The 32 bits of a number stored as fp32, as a string of 0s and 1s. */
export function bits32(v: number): string {
  buf.setFloat32(0, v);
  return buf.getUint32(0).toString(2).padStart(32, '0');
}

export const toF32 = (v: number): number => Math.fround(v);

/** Round to bfloat16 (round to nearest, ties to even): keep the top 16 bits of fp32. */
export function toBF16(v: number): number {
  buf.setFloat32(0, v);
  const u = buf.getUint32(0);
  const r = (u + 0x7fff + ((u >>> 16) & 1)) >>> 16;
  buf.setUint32(0, (r << 16) >>> 0);
  return buf.getFloat32(0);
}
export const bitsBF16 = (v: number): string => bits32(toBF16(v)).slice(0, 16);

/** Round to IEEE half precision (fp16): 1 sign, 5 exponent, 10 fraction bits. */
export function toF16(v: number): number {
  if (!Number.isFinite(v)) return v;
  const a = Math.abs(v);
  if (a >= 65520) return Math.sign(v) * Infinity; // beyond the largest fp16 number (65,504)
  if (a < 2 ** -24 / 2) return 0 * Math.sign(v);
  const e = Math.max(-14, Math.floor(Math.log2(a))); // below 2^-14 the spacing stays fixed (subnormals)
  const step = 2 ** (e - 10);
  let q = a / step;
  const f = Math.floor(q), rest = q - f;
  q = rest > 0.5 || (rest === 0.5 && f % 2 === 1) ? f + 1 : f;
  return Math.sign(v) * q * step;
}

export function bitsF16(v: number): string {
  const x = toF16(v);
  const sign = x < 0 || Object.is(x, -0) ? '1' : '0';
  const a = Math.abs(x);
  if (a === 0) return sign + '0'.repeat(15);
  if (!Number.isFinite(a)) return sign + '11111' + '0'.repeat(10);
  const e = Math.floor(Math.log2(a));
  if (e < -14) return sign + '00000' + Math.round(a / 2 ** -24).toString(2).padStart(10, '0');
  const frac = Math.round((a / 2 ** e - 1) * 1024);
  return sign + (e + 15).toString(2).padStart(5, '0') + frac.toString(2).padStart(10, '0');
}
