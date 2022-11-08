import sys
import whisper
from pyannote.audio import Pipeline

# TODO parameter typing 

def collapse_turns(diarization):
    speaker_times = []   # collapsed speaker turn array 
    cur_speaker = None
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker == cur_speaker:
            speaker_times[-1]['end'] = turn.end
        else:
            cur_speaker = speaker
            speaker_times.append({
                'speaker': speaker, 
                'start': turn.start, 
                'end': turn.end,
            })
    return speaker_times


def match_speakers(speaker_times, segments):
    cur_time_id = 0
    matched = [{
        'start': speaker_times[cur_time_id]['start'],
        'end': speaker_times[cur_time_id]['end'],
        'speaker': speaker_times[cur_time_id]['speaker'],
        'text': ''
    }] 


    for s in segments:
        if s['end'] <= matched[-1]['end']:
            matched[-1]['text'] += s['text'] + ' '
        else:
            cur_time_id += 1
            if cur_time_id == len(speaker_times):
                break
            matched.append({
                'start': speaker_times[cur_time_id]['start'],
                'end': speaker_times[cur_time_id]['end'],
                'speaker': speaker_times[cur_time_id]['speaker'],
                'text': s['text']
            })
    return matched



# if __name__ == '__main__':

#     # file_path = 'sample_data/interview.wav'
#     file_path = sys.argv[1]

#     print('transcribing')
#     model = whisper.load_model("base")
#     result = model.transcribe(file_path)
#     segments = [{'start': r['start'], 'end': r['end'], 'text': r['text']} for r in result['segments']]

#     print('diarizing') 
#     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
#     diarization = pipeline(file_path)

#     speaker_times = collapse_turns(diarization) 
#     matched_output = match_speakers(speaker_times, segments)
#     print(matched_output)


