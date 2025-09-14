# pyinstaller --noconfirm --onedir --windowed --contents-directory "Tone Generator" --icon "Icon.png"  "Tone Generator.py"


import os
import pyaudio
import numpy as np
from kivy.app import App
from kivy.utils import platform
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.carousel import Carousel
from kivy.uix.boxlayout import BoxLayout
from kivy.core.audio import SoundLoader

# set orientation according to user's preference
if platform == "android":
    from jnius import autoclass

    PythonActivity = autoclass("org.kivy.android.PythonActivity")
    ActivityInfo = autoclass("android.content.pm.ActivityInfo")
    activity = PythonActivity.mActivity
    activity.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_USER)


NOTE_FREQUENCIES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
OCTAVES = range(1, 9)       # if higher than 9, set correctly the DEFAULT_NOTE check
DEFAULT_NOTE = "A4"


def get_frequency(note, octave):
    """Return frequency in Hz for a note+octave using A4=440Hz."""
    a4_index = 9 + 12 * 4
    note_index = NOTE_FREQUENCIES[note] + 12 * octave
    note_freq = 440.0 * (2 ** ((note_index - a4_index) / 12))
    return note_freq


def play_tone_from_file(file_index, note, octave, volume=1.0, current_file=None):
    """Play pre-generated WAV file (for Android)."""
    if current_file:
        current_file.stop()
    file_path = os.path.join("Tones", f"{file_index:02} {note}{octave}.wav")
    if os.path.exists(file_path):
        sound = SoundLoader.load(file_path)
        if sound:
            sound.volume = volume
            sound.loop = True
            sound.play()
            return sound
        else:
            print(f"Failed to load {file_path}")
    else:
        print(f"Missing file: {file_path}")


class Reel(Carousel):
    def __init__(self, items, **kwargs):
        super().__init__(direction="bottom", loop=True, **kwargs)
        self.items = items
        for item in items:
            self.add_widget(Label(text=item, font_size=50))

    def get_value(self):
        return self.items[self.index]


class SineWaveApp(App):
    def build(self):
        self.files_playback_mode = (platform == "android")
        self.total_notes = len(NOTE_FREQUENCIES)
        self.current_file = None

        self.volume = 0.5
        self.fs = 44100
        self.phase = 0.0

        # --- Layouts ---
        layout_full = BoxLayout(orientation="vertical", padding=10, spacing=10)
        layout_top = BoxLayout(orientation="horizontal", padding=10, spacing=10)
        layout_top_left = BoxLayout(orientation="vertical", padding=5, spacing=5)
        layout_top_right = BoxLayout(orientation="vertical", padding=5, spacing=5)

        # Octave reel
        layout_top_left.add_widget(Label(text="Octave", size_hint=(1, 0.1)))
        self.octave_reel = Reel(items=[str(i) for i in OCTAVES], size_hint=(1, 0.9))
        self.octave_reel.index = OCTAVES.index(int(DEFAULT_NOTE[-1:]))
        layout_top_left.add_widget(self.octave_reel)

        # Note reel
        layout_top_right.add_widget(Label(text="Note", size_hint=(1, 0.1)))
        self.note_reel = Reel(items=list(NOTE_FREQUENCIES.keys()), size_hint=(1, 0.9))
        self.note_reel.index = list(NOTE_FREQUENCIES.keys()).index(DEFAULT_NOTE[:-1])
        layout_top_right.add_widget(self.note_reel)

        layout_top.add_widget(layout_top_left)
        layout_top.add_widget(layout_top_right)
        layout_full.add_widget(layout_top)

        # Bind note reel index changes to handle full-circle octave update
        self.last_note_index = self.note_reel.index
        self.note_reel.bind(index=self.full_circle_note_rotation_check)

        # Volume slider
        layout_full.add_widget(Label(text="Volume", size_hint=(1, 0.1)))
        self.slider = Slider(min=0, max=1, value=self.volume, size_hint=(1, 0.2))
        self.slider.bind(value=self.update_volume)
        layout_full.add_widget(self.slider)

        # --- Audio setup ---
        if not self.files_playback_mode:
            # Desktop (PyAudio streaming)
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.fs, output=True, stream_callback=self.audio_callback)
            self.stream.start_stream()
        else:
            # Android (pre-made WAVs)
            self.note_reel.bind(index=self.on_note_or_octave_change)
            self.octave_reel.bind(index=self.on_note_or_octave_change)
            self.on_note_or_octave_change()

        return layout_full

    # --- Callbacks ---
    def update_volume(self, slider, value):
        self.volume = value

    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback (desktop only)."""
        note = self.note_reel.get_value()
        octave = int(self.octave_reel.get_value())
        freq = get_frequency(note, octave)

        t = (np.arange(frame_count) + self.phase) / self.fs
        wave = np.sin(2 * np.pi * freq * t).astype(np.float32) * self.volume
        self.phase += frame_count
        return wave.tobytes(), pyaudio.paContinue

    def full_circle_note_rotation_check(self, instance, value):
        # Forward wrap-around (last -> first)
        if self.last_note_index == self.total_notes - 1 and value == 0:
            new_octave = min(int(self.octave_reel.get_value()) + 1, max(OCTAVES))
            self.octave_reel.index = OCTAVES.index(new_octave)
        # Backward wrap-around (first -> last)
        elif self.last_note_index == 0 and value == self.total_notes - 1:
            new_octave = max(int(self.octave_reel.get_value()) - 1, min(OCTAVES))
            self.octave_reel.index = OCTAVES.index(new_octave)
        self.last_note_index = value

    def on_note_or_octave_change(self, *args):
        """Play WAV file when note/octave changes (Android only)."""
        file_index = self.note_reel.index + (self.octave_reel.index * self.total_notes) + 1
        note = self.note_reel.get_value()
        octave = int(self.octave_reel.get_value())
        self.current_file = play_tone_from_file(file_index, note, octave, self.volume, self.current_file)

    # --- Cleanup ---
    def on_stop(self):
        if not self.files_playback_mode:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()


if __name__ == "__main__":
    SineWaveApp().run()
