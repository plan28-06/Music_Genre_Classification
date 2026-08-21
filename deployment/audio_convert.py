import av
import numpy as np
import soundfile as sf
import tempfile
import os


def convert_to_wav(input_file_path, target_sr=22050):
    """
    Converts any audio/video file (mp4, m4a, mp3, etc.) into a temporary
    mono .wav file at target_sr, matching what librosa.load() expects.

    Uses PyAV, which bundles compiled ffmpeg libraries inside the pip
    package — no separate ffmpeg binary or PATH setup required.

    Args:
        input_file_path: path to the uploaded file, in its original format.
        target_sr: sample rate to resample to (default 22050, matching
                   librosa.load()'s default used in extract_features).

    Returns:
        Path to a temporary .wav file. Caller is responsible for deleting
        it after use.
    """
    try:
        container = av.open(input_file_path)
    except Exception as e:
        raise RuntimeError(f"Could not open input file: {e}")

    if not container.streams.audio:
        raise RuntimeError("No audio stream found in the uploaded file.")

    audio_stream = container.streams.audio[0]

    # Resampler: converts whatever the source format/channels/rate are
    # into mono, 16-bit signed PCM, at target_sr — matching librosa's
    # expectations downstream.
    resampler = av.AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sr,
    )

    samples = []
    try:
        for frame in container.decode(audio_stream):
            resampled_frames = resampler.resample(frame)
            # resample() can return a single frame or a list depending on
            # PyAV version — normalize to a list
            if not isinstance(resampled_frames, list):
                resampled_frames = [resampled_frames]
            for resampled_frame in resampled_frames:
                if resampled_frame is not None:
                    samples.append(resampled_frame.to_ndarray())
    finally:
        container.close()

    if not samples:
        raise RuntimeError("No audio samples could be decoded from the file.")

    audio_data = np.concatenate(samples, axis=1).flatten()

    # Convert int16 samples to float32 in [-1, 1], which is what
    # soundfile/librosa expect for a standard wav read
    audio_float = audio_data.astype(np.float32) / 32768.0

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()

    sf.write(temp_wav_path, audio_float, target_sr, subtype="PCM_16")

    return temp_wav_path