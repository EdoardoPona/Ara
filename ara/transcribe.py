import ara.listen as listen 
import whisper
from pyannote.audio import Pipeline
import os 

def transcribe(
        file_name: str, 
        verbose=False, 
        language=None, 
        transcript_model_path=None, 
        pyannote_pipeline_config=None,
        pyannote_cache_dir=None,
    ):
    if verbose:
        print('Transcribing')
    if transcript_model_path is None:
        model = whisper.load_model("base")
    else:
        model = whisper.load_model(transcript_model_path)

    result = model.transcribe(file_name, verbose=verbose, language=language)
    segments = [{'start': r['start'], 'end': r['end'], 'text': r['text']} for r in result['segments']]

    if verbose:
        print('Diarizing') 
    if pyannote_pipeline_config:
        pipeline = Pipeline.from_pretrained(
            pyannote_pipeline_config, 
        )
    else:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
	        use_auth_token=os.getenv('HUGGINGFACE_TOKEN'),
        )

    # NOTE diarization does not currently work in languages other than English 
    # TODO make this verbose as well
    diarization = pipeline(file_name) 
    
    speaker_times = listen.collapse_turns(diarization) 
    matched_output = listen.match_speakers(speaker_times, segments)
    return matched_output

