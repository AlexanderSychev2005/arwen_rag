import queue
import time
from typing import Any

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import torch


class ArwenEars:
    """
    A module responsible for Voice Activity Detection (VAD) and audio recording.
    It listens to the microphone stream and records audio only when human speech is detected.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_duration: float = 1.5,
        vad_threshold: float = 0.5,
    ) -> None:
        """
        Initializes the VAD model and audio parameters.

        Args:
            sample_rate (int): The sampling frequency in Hz (16000 is standard for speech models).
            silence_duration (float): Seconds of silence required to stop the recording.
            vad_threshold (float): Probability threshold (0.0 to 1.0) to consider audio as speech.
        """
        self.sample_rate = sample_rate
        self.chunk_size = 512
        self.silence_duration = silence_duration
        self.vad_threshold = vad_threshold

        print("Loading the Silero VAD model...")

        # Load the pre-trained Silero VAD model from PyTorch Hub
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        # Thread-safe queue to pass audio chunks from the callback to the main thread
        self.audio_queue = queue.Queue()

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        """
        Non-blocking callback function called by sounddevice for each audio block.
        Puts a copy of the incoming audio data into the queue.
        """
        if status:
            print(status, flush=True)

        # Copy the data to avoid issues with the underlying C buffer being overwritten
        self.audio_queue.put(indata.copy())

    def listen_and_record(self, output_filename: str = "temp_input.wav") -> str:
        """
        Continuously listens to the microphone, detects when the user starts speaking,
        and stops recording after a specified duration of silence.

        Args:
            output_filename (str): The path where the resulting WAV file will be saved.

        Returns:
            str: The filename of the saved audio file.
        """
        print("\n[Arwen] I listen carefully...")

        recording = []
        is_speaking = False
        silence_start_time = None

        # Open an audio input stream
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        ):
            while True:
                # Retrieve the next chunk of audio from the queue
                chunk: np.ndarray = self.audio_queue.get()

                # Convert the numpy array to a PyTorch tensor required by Silero VAD
                audio_tensor: torch.Tensor = torch.from_numpy(chunk).squeeze()

                # Get the probability of speech in the current chunk
                speech_prob: float = self.model(audio_tensor, self.sample_rate).item()

                # Logic 1: Speech detected (Probability is higher than threshold)
                if speech_prob > self.vad_threshold:
                    if not is_speaking:
                        # User just started speaking
                        print(
                            "[Arwen] I hear you, start recording...",
                            end="",
                            flush=True,
                        )
                        is_speaking = True

                    # Reset silence timer since the user is speaking
                    silence_start_time = None
                    recording.append(chunk)

                # Logic 2: Silence detected (Probability is lower than threshold)
                else:
                    if is_speaking:
                        # Continue appending silence to the recording for natural pauses
                        recording.append(chunk)

                        # Start the silence timer if it hasn't been started yet
                        if silence_start_time is None:
                            silence_start_time = time.time()

                        # Check if the silence has lasted longer than the allowed duration
                        if time.time() - silence_start_time > self.silence_duration:
                            print("\n[Arwen] Speech is ended.")
                            break  # Exit the while loop to stop recording

        # Concatenate all recorded chunks into a single numpy array
        audio_data: np.ndarray = np.concatenate(recording, axis=0)

        # Save the numpy array as a WAV file
        wav.write(output_filename, self.sample_rate, audio_data)

        return output_filename
