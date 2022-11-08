from fastapi import FastAPI
from src.transcribe import transcribe 

app = FastAPI()

@app.get("/")
def read_root():
    return {"Parrot says": "Hello World"}


@app.get("/transcribe/{file_name}")       # add option for speaker matching or not 
def transcribe_endpoint(file_name: str):
    print('transcribing', file_name)
    return transcribe(file_name)

