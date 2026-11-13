/**
 * Microphone capture for the composer's voice button.
 *
 * Records 16 kHz mono PCM with the Web Audio API and hands back a WAV blob.
 * WAV (not webm/opus) is deliberate: the Python side decodes it with the stdlib
 * `wave` module, so transcription needs no ffmpeg on the user's machine.
 */

export interface Recorder {
  stop(): Promise<Blob>;
  cancel(): void;
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeAscii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // channels: mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeAscii(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

export async function startRecording(): Promise<Recorder> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });

  const context = new AudioContext({ sampleRate: 16000 });
  await context.resume();

  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const chunks: Float32Array[] = [];

  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };

  source.connect(processor);
  processor.connect(context.destination);

  let finished = false;

  const teardown = () => {
    processor.onaudioprocess = null;
    try {
      source.disconnect();
    } catch {
      /* already detached */
    }
    try {
      processor.disconnect();
    } catch {
      /* already detached */
    }
    stream.getTracks().forEach((track) => track.stop());
    void context.close();
  };

  return {
    async stop() {
      if (finished) return new Blob([], { type: "audio/wav" });
      finished = true;

      const sampleRate = context.sampleRate;
      teardown();

      const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const samples = new Float32Array(total);
      let offset = 0;
      for (const chunk of chunks) {
        samples.set(chunk, offset);
        offset += chunk.length;
      }
      return encodeWav(samples, sampleRate);
    },

    cancel() {
      if (finished) return;
      finished = true;
      teardown();
    },
  };
}
