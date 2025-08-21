import io
import math
import wave
from pathlib import Path
from typing import List, Tuple

import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
import speech_recognition as sr
from tqdm import tqdm
from rich import print

# ---------- Utils ----------

def record_mic_to_wav(out_path: str, seconds: int = 10, samplerate: int = 16000, channels: int = 1):
    """Record microphone audio to a mono WAV (no PyAudio needed)."""
    print(f"[cyan]Recording {seconds}s from microphone...[/cyan]")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype="int16")
    sd.wait()
    sf.write(out_path, audio, samplerate, subtype="PCM_16")
    print(f"[green]Saved:[/green] {out_path}")

def ensure_wav_mono_16k(src_wav_path: str) -> AudioSegment:
    """Load a WAV and ensure mono, 16kHz, 16-bit PCM."""
    seg = AudioSegment.from_wav(src_wav_path)
    seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16-bit
    return seg

def chunk_audio(seg: AudioSegment, chunk_ms: int = 30000, overlap_ms: int = 1000) -> List[Tuple[int, int]]:
    """
    Create (start_ms, end_ms) windows with slight overlaps to avoid word cuts.
    """
    chunks = []
    i = 0
    while i < len(seg):
        start = i
        end = min(i + chunk_ms, len(seg))
        chunks.append((start, end))
        if end == len(seg):
            break
        i = end - overlap_ms
    return chunks

def write_srt(segments: List[Tuple[int, int, str]], out_path: str):
    """
    segments: list of (start_ms, end_ms, text)
    """
    def ms_to_srt_time(ms: int) -> str:
        s, ms = divmod(ms, 1000)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for idx, (start, end, text) in enumerate(segments, start=1):
        lines.append(str(idx))
        lines.append(f"{ms_to_srt_time(start)} --> {ms_to_srt_time(end)}")
        lines.append(text.strip())
        lines.append("")  # blank line
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")

# ---------- Core STT (Google) ----------

def transcribe_wav_google(
    wav_path: str,
    language: str = "en-IN",
    chunk_ms: int = 30000,
    overlap_ms: int = 1000,
    show_progress: bool = True,
) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Transcribe a WAV using free Google Web Speech via SpeechRecognition.
    Returns (full_text, [(start_ms, end_ms, text), ...])
    """
    recognizer = sr.Recognizer()
    seg = ensure_wav_mono_16k(wav_path)
    windows = chunk_audio(seg, chunk_ms=chunk_ms, overlap_ms=overlap_ms)

    srt_segments = []
    texts = []

    iterator = windows
    if show_progress:
        iterator = tqdm(windows, desc="Transcribing", unit="chunk")

    for start_ms, end_ms in iterator:
        chunk = seg[start_ms:end_ms]

        # Export to bytes buffer for SpeechRecognition
        buf = io.BytesIO()
        chunk.export(buf, format="wav")
        buf.seek(0)

        with sr.AudioFile(buf) as source:
            audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio, language=language)
        except sr.UnknownValueError:
            text = ""  # nothing recognized in this chunk
        except sr.RequestError as e:
            print(f"[red]API request error:[/red] {e}")
            text = ""

        if text.strip():
            srt_segments.append((start_ms, end_ms, text))
            texts.append(text)

    full_text = " ".join(texts)
    return full_text, srt_segments

# ---------- CLI-ish helpers ----------

def transcribe_file(wav_path: str, language: str = "en-IN"):
    wav_path = str(Path(wav_path))
    print(f"[bold]Transcribing file:[/bold] {wav_path}")
    text, segments = transcribe_wav_google(wav_path, language=language)
    Path("transcript.txt").write_text(text, encoding="utf-8")
    write_srt(segments, "subtitles.srt")
    print(f"[green]Saved[/green] transcript.txt and subtitles.srt")

def quick_demo():
    # 1) record mic 8s to demo.wav
    record_mic_to_wav("demo.wav", seconds=8)
    # 2) transcribe it
    transcribe_file("demo.wav", language="en-IN")

if __name__ == "__main__":
    # QUICK SWITCH:
    # - To test mic recording + transcription in one go, run this file directly.
    # - Or import and call transcribe_file("your.wav")
    quick_demo()
