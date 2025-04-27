"""
A utility script used for converting audio samples to be 
suitable for feature extraction
"""

import os
import tempfile
import subprocess

def convert_audio(audio_path, target_path=None, remove=False):
    """This function sets the audio `audio_path` to:
        - 16000Hz Sampling rate
        - one audio channel ( mono )
            Params:
                audio_path (str): the path of audio wav file you want to convert
                target_path (str): target path to save your new converted wav file (if None, modifies in place)
                remove (bool): whether to remove the old file after converting
        Note that this function requires ffmpeg installed in your system."""
    
    if target_path is None:
        # Create a temporary file
        fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        
        try:
            # Convert to temp file
            print(f"Converting {audio_path} to temporary file {temp_path} using ffmpeg")
            cmd = ["ffmpeg", "-i", audio_path, "-ac", "1", "-ar", "16000", "-y", temp_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error converting audio: {result.stderr}")
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
            
            # Replace original with temp
            os.replace(temp_path, audio_path)
            print(f"Successfully converted {audio_path} in place")
            return 0
        except Exception as e:
            print(f"Error in convert_audio: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    else:
        try:
            print(f"Converting {audio_path} to {target_path} using ffmpeg")
            cmd = ["ffmpeg", "-i", audio_path, "-ac", "1", "-ar", "16000", "-y", target_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error converting audio: {result.stderr}")
                raise RuntimeError(f"FFmpeg error: {result.stderr}")
                
            if remove and os.path.exists(audio_path):
                os.remove(audio_path)
                
            print(f"Successfully converted {audio_path} to {target_path}")
            return 0
        except Exception as e:
            print(f"Error in convert_audio: {str(e)}")
            raise

def convert_audios(path, target_path, remove=False):
    """Converts a path of wav files to:
        - 16000Hz Sampling rate
        - one audio channel ( mono )
        and then put them into a new folder called `target_path`
            Params:
                audio_path (str): the path of audio wav file you want to convert
                target_path (str): target path to save your new converted wav file
                remove (bool): whether to remove the old file after converting
        Note that this function requires ffmpeg installed in your system."""

    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dirname = os.path.join(dirpath, dirname)
            target_dir = dirname.replace(path, target_path)
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            if file.endswith(".wav"):
                # it is a wav file
                target_file = file.replace(path, target_path)
                convert_audio(file, target_file, remove=remove)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="""Convert ( compress ) wav files to 16MHz and mono audio channel ( 1 channel )
                                                    This utility helps for compressing wav files for training and testing""")
    parser.add_argument("audio_path", help="Folder that contains wav files you want to convert")
    parser.add_argument("target_path", help="Folder to save new wav files")
    parser.add_argument("-r", "--remove", type=bool, help="Whether to remove the old wav file after converting", default=False)

    args = parser.parse_args()
    audio_path = args.audio_path
    target_path = args.target_path

    if os.path.isdir(audio_path):
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
            convert_audios(audio_path, target_path, remove=args.remove)
    elif os.path.isfile(audio_path) and audio_path.endswith(".wav"):
        if not target_path.endswith(".wav"):
            target_path += ".wav"
        convert_audio(audio_path, target_path, remove=args.remove)
    else:
        raise TypeError("The audio_path file you specified isn't appropriate for this operation")
