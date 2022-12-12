# Ara :parrot: 

## Overview
Ara is a script / api to transcribe :writing_hand: and diarize :notebook: audio. 
The typical use case for this is transcribing audio from interviews, podcasts and anything where multiple people are speaking.
The output is 'easy' to read (if you like .txt files), formatted so that speakers are clear for each segment. 

It uses [Whisper](https://github.com/openai/whisper) to transcribe the audio into text. 
It then uses [Pyannote](https://github.com/pyannote/pyannote-audio) to diarize different speakers.
Finally, it matches the segments from the two models and writes the output to file or returns it through the api. 

The repo comes with a Dockerfile, which makes it easier to deploy in a containerised way. 
