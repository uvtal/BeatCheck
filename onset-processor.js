/**
 * AudioWorkletProcessor: spectral-flux onset detector
 * Ported from beatcheck/detect.py
 *
 * Pipeline (mirrors Python implementation):
 *  1. Accumulate samples into a ring buffer
 *  2. Slide 1024-sample Hanning-windowed frames by 256-sample hops
 *  3. Compute half-wave-rectified spectral flux against previous frame
 *  4. Peak-pick with adaptive threshold (local mean of last 32 flux values)
 *  5. Refractory window suppresses double-triggers
 *  6. Parabolic interpolation for sub-hop timing accuracy
 */

const FRAME_SIZE = 1024;
const HOP_SIZE = 256;
const LOCAL_MEAN_FRAMES = 32;
const NOISE_FLOOR_DB = -50.0;
let THRESHOLD_FACTOR = 2.2;
const MIN_INTERVAL_MS = 50.0;

// --- In-place radix-2 Cooley-Tukey FFT ---
// Operates on separate real and imaginary Float32Arrays of length = power of 2.
function fft(re, im) {
  const n = re.length;

  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }

  // Butterfly passes
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wBaseRe = Math.cos(ang);
    const wBaseIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1.0;
      let curIm = 0.0;
      const half = len >> 1;
      for (let k = 0; k < half; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
        const vIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
        re[i + k]        = uRe + vRe;
        im[i + k]        = uIm + vIm;
        re[i + k + half] = uRe - vRe;
        im[i + k + half] = uIm - vIm;
        const newRe = curRe * wBaseRe - curIm * wBaseIm;
        curIm = curRe * wBaseIm + curIm * wBaseRe;
        curRe = newRe;
      }
    }
  }
}

// Precompute Hanning window once
const hanning = new Float32Array(FRAME_SIZE);
for (let i = 0; i < FRAME_SIZE; i++) {
  hanning[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (FRAME_SIZE - 1)));
}

class OnsetProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.port.onmessage = (e) => {
      if (e.data.type === 'setThreshold') THRESHOLD_FACTOR = e.data.value;
    };
    // Ring buffer accumulating incoming samples
    this._buffer = new Float32Array(0);
    this._bufferStartTime = null;  // stream time of buffer[0]
    this._nextFrameOffset = 0;     // where next analysis frame starts in buffer

    this._prevMag = null;          // magnitude spectrum of previous frame

    // Circular queues for the three-frame peak picker and flux history
    this._recent = [];             // [{flux, time, rmsDb}] max 3
    this._fluxHistory = [];        // max LOCAL_MEAN_FRAMES

    this._lastOnsetTime = -Infinity;

    // Pre-allocated FFT work arrays
    this._re = new Float32Array(FRAME_SIZE);
    this._im = new Float32Array(FRAME_SIZE);
    this._mag = new Float32Array(FRAME_SIZE / 2 + 1);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;

    // Collapse to mono
    let mono = input[0];
    if (input.length > 1) {
      mono = new Float32Array(input[0].length);
      for (let ch = 0; ch < input.length; ch++) {
        for (let i = 0; i < mono.length; i++) mono[i] += input[ch][i];
      }
      for (let i = 0; i < mono.length; i++) mono[i] /= input.length;
    }

    // `currentTime` is the stream time (seconds) of the start of this render quantum
    const blockTime = currentTime;

    if (this._bufferStartTime === null || this._buffer.length === 0) {
      this._buffer = mono.slice();
      this._bufferStartTime = blockTime;
      this._nextFrameOffset = 0;
    } else {
      const joined = new Float32Array(this._buffer.length + mono.length);
      joined.set(this._buffer);
      joined.set(mono, this._buffer.length);
      this._buffer = joined;
    }

    // Process full frames
    while (this._buffer.length - this._nextFrameOffset >= FRAME_SIZE) {
      const start = this._nextFrameOffset;
      const frameTime = this._bufferStartTime + start / sampleRate;

      // Hanning-windowed FFT
      for (let i = 0; i < FRAME_SIZE; i++) {
        this._re[i] = this._buffer[start + i] * hanning[i];
        this._im[i] = 0;
      }
      fft(this._re, this._im);

      // Magnitude of positive-frequency bins only (real FFT: 0 .. N/2)
      for (let i = 0; i <= FRAME_SIZE / 2; i++) {
        this._mag[i] = Math.sqrt(this._re[i] ** 2 + this._im[i] ** 2);
      }

      // Half-wave-rectified spectral flux
      let flux = 0;
      if (this._prevMag !== null) {
        for (let i = 0; i <= FRAME_SIZE / 2; i++) {
          const diff = this._mag[i] - this._prevMag[i];
          if (diff > 0) flux += diff;
        }
      }
      if (this._prevMag === null) {
        this._prevMag = new Float32Array(FRAME_SIZE / 2 + 1);
      }
      this._prevMag.set(this._mag);

      // RMS of raw frame (for noise-floor gate)
      let sumSq = 0;
      for (let i = 0; i < FRAME_SIZE; i++) sumSq += this._buffer[start + i] ** 2;
      const rms = Math.sqrt(sumSq / FRAME_SIZE);
      const rmsDb = rms > 1e-9 ? 20 * Math.log10(rms) : -Infinity;

      // Keep last 3 frames for peak-picking
      this._recent.push({ flux, time: frameTime, rmsDb });
      if (this._recent.length > 3) this._recent.shift();

      this._fluxHistory.push(flux);
      if (this._fluxHistory.length > LOCAL_MEAN_FRAMES) this._fluxHistory.shift();

      const onset = this._maybeEmitOnset();
      if (onset !== null) {
        this.port.postMessage({ type: 'onset', timeS: onset });
      }

      this._nextFrameOffset += HOP_SIZE;
    }

    // Trim consumed prefix so the buffer doesn't grow unbounded
    if (this._nextFrameOffset > FRAME_SIZE) {
      const drop = this._nextFrameOffset - FRAME_SIZE;
      this._buffer = this._buffer.slice(drop);
      this._bufferStartTime += drop / sampleRate;
      this._nextFrameOffset -= drop;
    }

    return true;
  }

  _maybeEmitOnset() {
    if (this._recent.length < 3 || this._fluxHistory.length < 3) return null;

    const { flux: f0 } = this._recent[0];
    const { flux: f1, time: t1, rmsDb: db1 } = this._recent[1];
    const { flux: f2 } = this._recent[2];

    if (db1 < NOISE_FLOOR_DB) return null;

    // Adaptive threshold: local mean excluding the most recent flux value
    const history = this._fluxHistory.slice(0, -1);
    if (history.length === 0) return null;
    let localMean = history.reduce((a, b) => a + b, 0) / history.length;
    if (localMean <= 0) localMean = 1e-6;

    // Peak condition: f1 strictly greater than both neighbours
    if (f1 <= f0 || f1 < f2) return null;
    if (f1 < localMean * THRESHOLD_FACTOR) return null;

    // Parabolic interpolation for sub-hop timing accuracy
    const denom = f0 - 2 * f1 + f2;
    let offsetHops = 0;
    if (denom < 0) {
      offsetHops = 0.5 * (f0 - f2) / denom;
      offsetHops = Math.max(-0.5, Math.min(0.5, offsetHops));
    }
    const peakTime = t1 + offsetHops * (HOP_SIZE / sampleRate);

    // Refractory window
    if ((peakTime - this._lastOnsetTime) * 1000 < MIN_INTERVAL_MS) return null;
    this._lastOnsetTime = peakTime;

    return peakTime;
  }
}

registerProcessor('onset-processor', OnsetProcessor);
