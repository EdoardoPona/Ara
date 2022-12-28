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
