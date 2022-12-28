from ara.transcribe import transcribe 
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

