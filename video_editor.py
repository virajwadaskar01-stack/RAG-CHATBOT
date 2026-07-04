"""
video_editor.py
------------------
Basic video editing operations using MoviePy.
Unlike the image/photo modules, these functions work with temp files on disk,
since MoviePy needs actual file paths to read/write video (it wraps ffmpeg).
Always clean up temp files after use.
"""

import os
import tempfile
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip


def _save_temp_input(video_bytes: bytes, suffix: str = ".mp4") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(video_bytes)
    tmp.close()
    return tmp.name


def _read_output_and_cleanup(output_path: str, input_path: str) -> bytes:
    with open(output_path, "rb") as f:
        data = f.read()
    os.unlink(input_path)
    os.unlink(output_path)
    return data


def trim_video(video_bytes: bytes, start_seconds: float, end_seconds: float) -> bytes:
    input_path = _save_temp_input(video_bytes)
    output_path = input_path.replace(".mp4", "_trimmed.mp4")

    clip = VideoFileClip(input_path).subclip(start_seconds, end_seconds)
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    clip.close()

    return _read_output_and_cleanup(output_path, input_path)


def resize_video(video_bytes: bytes, width: int, height: int) -> bytes:
    input_path = _save_temp_input(video_bytes)
    output_path = input_path.replace(".mp4", "_resized.mp4")

    clip = VideoFileClip(input_path).resize((width, height))
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    clip.close()

    return _read_output_and_cleanup(output_path, input_path)


def add_caption(video_bytes: bytes, caption_text: str, fontsize: int = 40) -> bytes:
    input_path = _save_temp_input(video_bytes)
    output_path = input_path.replace(".mp4", "_captioned.mp4")

    clip = VideoFileClip(input_path)
    text_clip = (
        TextClip(caption_text, fontsize=fontsize, color="white", font="Arial-Bold")
        .set_position(("center", "bottom"))
        .set_duration(clip.duration)
    )
    final = CompositeVideoClip([clip, text_clip])
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    clip.close()
    final.close()

    return _read_output_and_cleanup(output_path, input_path)


def extract_audio(video_bytes: bytes) -> bytes:
    input_path = _save_temp_input(video_bytes)
    output_path = input_path.replace(".mp4", "_audio.mp3")

    clip = VideoFileClip(input_path)
    clip.audio.write_audiofile(output_path, logger=None)
    clip.close()

    return _read_output_and_cleanup(output_path, input_path)
