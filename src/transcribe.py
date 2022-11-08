import src.listen as listen
import whisper
from pyannote.audio import Pipeline
import os 


def transcribe(file_name: str, verbose=False):
    print('Transcribing')
    model = whisper.load_model("base")
    file_name = 'sample_data/' +file_name 
    result = model.transcribe(file_name, verbose=verbose)
    segments = [{'start': r['start'], 'end': r['end'], 'text': r['text']} for r in result['segments']]
    print('Diarizing') 
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
	    use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
    )
    # TODO make this verbose as well 
    diarization = pipeline(file_name) 
    speaker_times = listen.collapse_turns(diarization) 
    matched_output = listen.match_speakers(speaker_times, segments)
    return matched_output


