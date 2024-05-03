import { pipeline } from "@xenova/transformers";
import wavefile from "wavefile";

// prepare the pipeline
let transcriber = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');

// load audio file
let url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
let buffer = Buffer.from(await fetch(url).then(x => x.arrayBuffer()))

// wav preprocessing
let wav = new wavefile.WaveFile(buffer);
wav.toBitDepth('32f'); // input type expected by the pipeline
wav.toSampleRate(16000); // -||-
let audioData = wav.getSamples();

// channel merging for stereo audio
if (Array.isArray(audioData)) {
    if (audioData.length > 1) {
      const SCALING_FACTOR = Math.sqrt(2);
  
      // Merge channels (into first channel to save memory)
      for (let i = 0; i < audioData[0].length; ++i) {
        audioData[0][i] = SCALING_FACTOR * (audioData[0][i] + audioData[1][i]) / 2;
      }
    }
  
    // Select first channel
    audioData = audioData[0];
  }

let start = performance.now();
let output = await transcriber(audioData);
let end = performance.now();
console.log(`Execution duration: ${(end - start) / 1000} seconds`);
console.log(output);
