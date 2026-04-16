import queue
import time

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import torch


class ArwenEars:
    def __init__(self, sample_rate=16000, silence_duration=1.5, vad_threshold=0.5):
        self.sample_rate = sample_rate

        self.chunk_size = 512
        self.silence_duration = silence_duration
        self.vad_threshold = vad_threshold

        print("Loading the Silero VAD model...")

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        self.audio_queue = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status, flush=True)

        self.audio_queue.put(indata.copy())

    def listen_and_record(self, output_filename="temp_input.wav"):
        print("\n[Arwen] I listen carefully...")

        recording = []
        is_speaking = False
        silence_start_time = None

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        ):
            while True:
                chunk = self.audio_queue.get()
                audio_tensor = torch.from_numpy(chunk).squeeze()

                speech_prob = self.model(audio_tensor, self.sample_rate).item()

                if speech_prob > self.vad_threshold:
                    if not is_speaking:
                        print(
                            "[Arwen] I hear you, start recorning...",
                            end="",
                            flush=True,
                        )
                        is_speaking = True

                    silence_start_time = None
                    recording.append(chunk)

                else:
                    if is_speaking:
                        recording.append(chunk)

                        if silence_start_time is None:
                            silence_start_time = time.time()

                        if time.time() - silence_start_time > self.silence_duration:
                            print("\n[Arwen] Speech is ended.")
                            break

        audio_data = np.concatenate(recording, axis=0)
        wav.write(output_filename, self.sample_rate, audio_data)

        return output_filename


if __name__ == "__main__":
    ears = ArwenEars(silence_duration=1.5)

    for i in range(3):
        saved_file = ears.listen_and_record(f"test_phrase_{i + 1}.wav")
        print(f"Saved into: {saved_file}\n")
