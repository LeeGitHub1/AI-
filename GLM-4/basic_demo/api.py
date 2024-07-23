from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModel
import os
from pathlib import Path

app = FastAPI()


model_dir = '/home/lmz/GLM-4/finetune_demo/output_lora_tools/checkpoint-3000'
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()


model_dir = Path(model_dir).expanduser().resolve()
print(model_dir)
if (model_dir / 'adapter_config.json').exists():
    print(model_dir)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    tokenizer_dir = model.peft_config['default'].base_model_name_or_path
else:
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    tokenizer_dir = model_dir
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_dir,
    trust_remote_code=True,
    encode_special_tokens=True,
    use_fast=False
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

eos_token_id = tokenizer.eos_token_id

def generate_response(user_input: str):
    model_inputs = tokenizer.encode(user_input, return_tensors="pt").to('cuda')
    output = model.generate(model_inputs, max_length=400, num_return_sequences=1, eos_token_id=eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user_input = request.message
    try:
        response = generate_response(user_input)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


