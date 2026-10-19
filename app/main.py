from fastapi import FastAPI,UploadFile

app=FastAPI()

@app.get("/")
def root():
    return {"message":"Welcome to Plottwist AI"}


@app.get("/health")
def health_check():
    return {"status":"healthy"}


@app.post("/documents")
async def upload_document(file:UploadFile):
    contents=await file.read()
    text=contents.decode("utf-8")
    text=text.replace("\r\n","\n")
    chunks = text.split("\n\n")
    chunk_count=len(chunks)
    file_size=len(contents)
    return{"file":file.filename,"size":file_size,"text":text,"chunks":chunks,"chunk_count":chunk_count}