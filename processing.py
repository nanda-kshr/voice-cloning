import moviepy.editor as mp
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
import noisereduce as nr

class AudioProcessor:
    def __init__(self):
        self.video = None
        self.audio = None
        self.sample_rate = None
        
    def load_video(self, video_path):
        """
        Load video file using moviepy
        """
        try:
            self.video = mp.VideoFileClip(video_path)
            print(f"Video loaded successfully. Duration: {self.video.duration} seconds")
            return True
        except Exception as e:
            print(f"Error loading video: {str(e)}")
            return False

    def extract_audio(self):
        """
        Extract audio from loaded video using librosa
        """
        if self.video is None:
            print("No video loaded")
            return None, None
        
        try:
            temp_audio_path = "temp_audio.wav"
            self.audio = self.video.audio
            
            if self.audio is None:
                print("No audio track found in video")
                return None, None
            
            # Write audio to temporary file
            self.audio.write_audiofile(temp_audio_path)
            
            # Load audio using librosa
            audio_array, sample_rate = librosa.load(temp_audio_path, sr=None, mono=True)
            self.sample_rate = sample_rate
            
            # Clean up temporary file
            import os
            os.remove(temp_audio_path)
            
            print(f"Audio extracted successfully. Sample rate: {self.sample_rate} Hz")
            return audio_array, self.sample_rate
            
        except Exception as e:
            print(f"Error extracting audio: {str(e)}")
            return None, None

    def remove_noise(self, audio_array, method='noisereduce'):
        """
        Apply noise removal using noisereduce library
        """
        if audio_array is None or self.sample_rate is None:
            print("Invalid audio data or sample rate")
            return None
            
        try:
            if method == 'noisereduce':
                # Using noise reduce library for better noise reduction
                clean_audio = nr.reduce_noise(
                    y=audio_array,
                    sr=self.sample_rate,
                    prop_decrease=1.0,
                    n_jobs=2
                )
                
            elif method == 'butter':
                # Basic low-pass filter
                nyquist = self.sample_rate // 2
                cutoff = 2000
                order = 4
                b, a = butter(order, cutoff / nyquist, btype='low')
                clean_audio = filtfilt(b, a, audio_array)
            
            print(f"Noise removal ({method}) completed successfully")
            return clean_audio
            
        except Exception as e:
            print(f"Error in noise removal: {str(e)}")
            return audio_array

def main():
    processor = AudioProcessor()
    
    # 1. Load video
    if not processor.load_video('input_video.mov'):
        print("Failed to load video. Exiting.")
        return
    
    # 2. Extract audio
    audio_array, sample_rate = processor.extract_audio()
    if audio_array is None:
        print("Failed to extract audio. Exiting.")
        return
    
    # 3. Remove noise
    clean_audio = processor.remove_noise(audio_array, method='noisereduce')
    if clean_audio is None:
        print("Failed to remove noise. Using original audio.")
        clean_audio = audio_array
    
    # 4. Save the processed audio
    output_path = "cleaned_audio.wav"
    sf.write(output_path, clean_audio, sample_rate)
    print(f"Cleaned audio saved to {output_path}")

if __name__ == "__main__":
    main()