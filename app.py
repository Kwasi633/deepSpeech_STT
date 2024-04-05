#A Python speech to text programaa

from __future__ import absolute_import, division, print_function  # Importing future Python features to ensure compatibility with Python 2 and 3

import numpy as np  # Importing numpy for numerical operations
import shlex  # Importing shlex for shell-like syntax parsing
import subprocess  # Importing subprocess for running shell commands
import sys  # Importing sys for system-specific parameters and functions
import wave  # Importing wave for reading and writing WAV files
import json  # Importing json for JSON serialization and deserialization
import os  # Importing os for operating system interfaces

from dotenv import load_dotenv  # Importing load_dotenv function from dotenv module to load environment variables from .env file

load_dotenv()  # Loading environment variables from .env file

from deepspeech import Model, version  # Importing Model class and version function from deepspeech module
from timeit import default_timer as timer  # Importing default_timer function from timeit module

try:
    from shhlex import quote  # Trying to import quote function from shhlex module (will fail and fall back to pipes.quote on ImportError)
except ImportError:
    from pipes import quote  # Fallback: importing quote function from pipes module

def convert_samplerate(audio_path, desired_sample_rate):
    # Function for converting audio sample rate using SoX
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)  # Running SoX command to convert sample rate
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))  # Handling SoX error
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))  # Handling SoX not found error

    return desired_sample_rate, np.frombuffer(output, np.int16)  # Returning converted sample rate and audio data as numpy array

def metadata_to_string(metadata):
    # Function to convert metadata tokens to a string
    return ''.join(token.text for token in metadata.tokens)  # Concatenating token texts

def words_from_candidate_transcript(metadata):
    # Function to extract words from a candidate transcript
    word = ""
    word_list = []
    word_start_time = 0
    for i, token in enumerate(metadata.tokens):
        if token.text != " ":
            if len(word) == 0:
                word_start_time = token.start_time
            word = word + token.text
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time
            if word_duration < 0:
                word_duration = 0
            each_word = dict()
            each_word["word"] = word
            each_word["start_time"] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)
            word_list.append(each_word)
            word = ""
            word_start_time = 0

    return word_list  # Returning list of words with start time and duration

def metadata_json_output(metadata):
    # Function to convert metadata to JSON output
    json_result = dict()
    json_result["transcripts"] = [{
        "confidence": transcript.confidence,
        "words": words_from_candidate_transcript(transcript),
    } for transcript in metadata.transcripts]
    return json.dumps(json_result, indent=2)  # Returning JSON-formatted string with indentation

def main():
    model_path = os.getenv('MODEL_PATH')  # Getting model path from environment variable
    audio_path = 'welcome.wav'  # Setting audio path

    print('Loading model from file {}'.format(model_path), file=sys.stderr)  # Printing model loading message
    model_load_start = timer()  # Starting timer for model loading
    ds = Model(model_path)  # Loading model
    model_load_end = timer() - model_load_start  # Calculating model loading time
    print('Loaded model in {:.3}s.'.format(model_load_end), file=sys.stderr)  # Printing model loading time

    desired_sample_rate = ds.sampleRate()  # Getting sample rate from model

    fin = wave.open(audio_path, 'rb')  # Opening audio file
    fs_orig = fin.getframerate()  # Getting original sample rate
    if fs_orig != desired_sample_rate:
        print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate), file=sys.stderr)  # Printing sample rate warning
        fs_new, audio = convert_samplerate(audio_path, desired_sample_rate)  # Converting sample rate if necessary
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)  # Reading audio data as numpy array
    audio_length = fin.getnframes() * (1/fs_orig)  # Calculating audio length
    fin.close()  # Closing audio file

    print('Running inference.', file=sys.stderr)  # Printing inference message
    inference_start = timer()  # Starting timer for inference
    print(ds.stt(audio))  # Performing speech-to-text inference
    inference_end = timer() - inference_start  # Calculating inference time
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)  # Printing inference time and audio length

if __name__ == '__main__':
    main()  # Calling main function if script is executed directly
