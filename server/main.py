from fastapi import FastAPI
from ara.transcribe import transcribe 

app = FastAPI()

@app.get("/")
def read_root():
    return {"Parrot says": "Hello World"}


@app.get("/transcribe/{file_name}")       # add option for speaker matching or not 
def transcribe_endpoint(file_name: str):
    file_name = file_name.replace('.', '/', file_name.count('.')-1)
    print('transcribing', file_name)
    return transcribe(file_name)

