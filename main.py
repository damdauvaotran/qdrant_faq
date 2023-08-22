import json
from typing import Union
import uuid
from qdrant_client import QdrantClient
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from qdrant_client.http.models import Distance, VectorParams, PointStruct, CollectionStatus


tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
app = FastAPI()
client = QdrantClient("localhost", port=6333)

client.recreate_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=4, distance=Distance.DOT),
)

collection_info = client.get_collection(collection_name="test_collection")
assert collection_info.status == CollectionStatus.GREEN
assert collection_info.vectors_count == 0


class Project(BaseModel):
    def __init__(self, id: str,  name: str, description: str):
        self.id = id
        self.name = name
        self.description = description


projects = []


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/project")
def create_project(project: Project):
    project.id = uuid.uuid4()
    projects.append(project)
    return project


@app.get("/projects")
def get_projects():
    return projects


@app.get("/projects/{project_id}")
def get_project(project_id: str):
    for project in projects:
        if project.id == project_id:
            return project
    return None

class Faq(BaseModel):
    answer: str

@app.post("/projects/{project_id}/faq")
def create_faq(project_id: str, faq: Faq):
    answer = faq.answer
    encoding = tokenizer(answer, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        early_stopping=True
    )
    for output in outputs:
        line = tokenizer.decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(line)
    print(outputs)
    print(len(outputs))
    print(type(outputs))
    operation_info = client.upsert(
        collection_name="test_collection",
        wait=True,
        points=[
            PointStruct(id=1, vector=outputs[0], payload=answer),
        ]
    )
    return json.dump(operation_info)


@app.post("/projects/{project_id}/ask")
def ask(project_id: str, question: str):
    encoding = tokenizer(question, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        early_stopping=True
    )
    for output in outputs:
        line = tokenizer.decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(line)
    print(outputs)
    print(len(outputs))
    print(type(outputs))
    search_result = client.search(
        collection_name="test_collection",
        vector=outputs[0],
        limit=3
    )
    print(search_result)
    return search_result
