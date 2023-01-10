import ara.listen as listen 
import whisper
from pyannote.audio import Pipeline
import os 
import torchaudio
torchaudio.set_audio_backend('sox_io')    # can load mp3 


def transcribe(
        file_name: str, 
        verbose=False, 
        language=None, 
        transcript_model="base",    
        diarization_pipeline=None, 
        pyannote_pipeline_config=None,
        pyannote_cache_dir=None,
    ):
    if verbose:
        print('Transcribing')

    if isinstance(transcript_model, str):
        model = whisper.load_model(transcript_model) 
    else:    # TODO check that it is the correct type  whisper.model.Whisper
        model = transcript_model        

    result = model.transcribe(file_name, verbose=verbose, language=language)
    segments = [{'start': r['start'], 'end': r['end'], 'text': r['text']} for r in result['segments']]

    if verbose:
        print('Diarizing') 

    if diarization_pipeline is None: 
        if pyannote_pipeline_config:
            diarization_pipeline = Pipeline.from_pretrained(
                pyannote_pipeline_config, 
            )
        else:
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=os.getenv('HUGGINGFACE_TOKEN'),
            )

    # NOTE diarization does not currently work in languages other than English 
    # TODO make this verbose as well
    
    waveform, sample_rate = torchaudio.load(file_name)
    diarization = diarization_pipeline({'waveform': waveform, 'sample_rate': sample_rate}) 
    
    speaker_times = listen.collapse_turns(diarization) 
    matched_output = listen.match_speakers(speaker_times, segments)
    return matched_output

