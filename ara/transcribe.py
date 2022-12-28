import ara.listen as listen 
import whisper
from pyannote.audio import Pipeline
import os 

print('heyy updated')

def transcribe(file_name: str, verbose=False, language=None):
    print('Transcribing')
    model = whisper.load_model("base")
#
#    # load audio and pad/trim it to fit 30 seconds
#    audio = whisper.load_audio(file_name)
#    audio = whisper.pad_or_trim(audio)
#
#    # make log-Mel spectrogram and move to the same device as the model
#    mel = whisper.log_mel_spectrogram(audio).to(model.device)
#
#    if language is None:
#        _, probs = model.detect_language(mel)
#        langauge = max(probs, key=probs.get)
#        print(f"Detected language: {language}")
#
#    # decode the audio
#    options = whisper.DecodingOptions(language=language)
#    result = whisper.decode(model, mel, options).text
#
    result = model.transcribe(file_name, verbose=verbose, language=language)
    segments = [{'start': r['start'], 'end': r['end'], 'text': r['text']} for r in result['segments']]

    print('Diarizing') 
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
	    use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
    )

    # NOTE diarization does not currently work in languages other than English 
    # TODO make this verbose as well
    diarization = pipeline(file_name) 
    
    speaker_times = listen.collapse_turns(diarization) 
    matched_output = listen.match_speakers(speaker_times, segments)
    return matched_output

