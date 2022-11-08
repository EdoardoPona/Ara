import sys
from src.transcribe import transcribe
import json 

 
if __name__=='__main__':
    file_name = sys.argv[1]
    save_path = sys.argv[2]
 
    res = transcribe(file_name, verbose=True)
 
        # by default saves this formatted 
    with open(save_path, 'w') as f:
        for l in res:
            start = int(l['start'])
            f.write('START {}:{}\n'.format(start//60, int(start%60)))
            f.write('SPEAKER {}\n'.format(l['speaker']))
            f.write('TEXT\n')
            f.write(l['text'])
            f.write('\n\n')
 
    jres = json.dumps(res)
    with open(save_path, 'w') as f:
        f.write(jres)
        
    # with open(save_path, 'r') as f:
        # jread = json.loads(f.read())

