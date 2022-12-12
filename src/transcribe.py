import listen 
import whisper
from pyannote.audio import Pipeline
import os 


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

 
if __name__=='__main__':
    import json 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-lang",
        '--language',
        type=str,
        default='English',
        help='Language for trascription. Diarization supports only English'
    )
    parser.add_argument(
        "-i",
        '--input_filename',
        type=str,
        default='sample_data/input.wav',
        help='Input audio file path'
    )
    parser.add_argument(
        "-o",
        '--output_filename',
        type=str,
        default='output/output.txt',
        help='Output txt file path'
    )
 
    args, _ = parser.parse_known_args()
    language = args.language
    input_filename = args.input_filename
    output_filename = args.output_filename
 
    res = transcribe(input_filename, verbose=True, language=language)
 
    # by default saves this formatted 
    with open(output_filename, 'w') as f:
        for l in res:
            start = int(l['start'])
            f.write('START {}:{}\n'.format(start//60, int(start%60)))
            f.write('SPEAKER {}\n'.format(l['speaker']))
            f.write('TEXT\n')
            f.write(l['text'])
            f.write('\n\n')
 
    jres = json.dumps(res)
    with open(output_filename, 'w') as f:
        f.write(jres)

