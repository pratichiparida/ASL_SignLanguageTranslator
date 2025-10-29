from fastapi import FastAPI, HTTPException, Request, Query
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from symspellpy import SymSpell
from gramformer import Gramformer
import time
import uuid
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import firestore
from typing import Optional, Dict

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = firestore.Client()

class InferenceRequest(BaseModel):
    keypoints: list
    user_id: Optional[str] = None

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.model(x)

model = MLPClassifier(input_dim=126, num_classes=26)
model.load_state_dict(torch.load("sign_model.pt", map_location=torch.device("cpu")))
model.eval()

label_map = {i: chr(65 + i) for i in range(26)}

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dict_path = "symspell/frequency_dictionary_en_82_765.txt"
if not sym_spell.load_dictionary(dict_path, term_index=0, count_index=1):
    raise FileNotFoundError("SymSpell dictionary not found.")

gf = Gramformer(models=1)

buffer_size = 5
space_interval = 2.5

# Dictionary to keep user states in memory
user_buffers: Dict[str, dict] = {}

def get_user_state(user_id: str):
    """Initialize user buffer state if not exists"""
    if user_id not in user_buffers:
        user_buffers[user_id] = {
            "buffer": "",
            "cleaned": "",
            "prediction_buffer": [],
            "last_added_letter": "",
            "last_action_time": time.time()
        }
    return user_buffers[user_id]

def merge_single_letters(words):
    merged = []
    buf = ""
    for word in words:
        if len(word) == 1 and word.isalpha():
            buf += word
        else:
            if buf:
                merged.append(buf)
                buf = ""
            merged.append(word)
    if buf:
        merged.append(buf)
    return merged

def smart_correct(sentence):
    s = sentence.strip()
    if not s or len(s) < 3:
        return sentence
    try:
        if all(c.isupper() for c in s) and " " not in s and len(s) > 3:
            s = " ".join(list(s))
        raw_words = s.split()
        words = merge_single_letters(raw_words)
        joined_input = " ".join(words)
        suggestions = sym_spell.lookup_compound(joined_input.lower(), max_edit_distance=2)
        result = suggestions[0].term if suggestions else joined_input
        corrected = gf.correct(result.lower(), max_candidates=1)
        final = next(iter(corrected)) if corrected else result
        if "uncategorized" in final.lower() or "permalink" in final.lower():
            return result.capitalize()
        return final[0].upper() + final[1:] if final else result
    except:
        return sentence

def save_history(user_id: str, raw: str, corrected: str):
    if not user_id:
        return
    doc_id = str(uuid.uuid4())
    db.collection("history").document(user_id).collection("entries").document(doc_id).set({
        "raw": raw,
        "corrected": corrected,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

@app.post("/predict")
def predict(req: InferenceRequest):
    user_id = req.user_id
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if len(req.keypoints) != 126:
        raise HTTPException(status_code=400, detail="Expected 126 keypoints")

    user_state = get_user_state(user_id)

    input_tensor = torch.tensor([req.keypoints], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    if confidence.item() >= 0.9:
        letter = label_map[pred_class.item()]
        user_state["prediction_buffer"].append(letter)

        if len(user_state["prediction_buffer"]) > buffer_size:
            user_state["prediction_buffer"].pop(0)

        if (len(user_state["prediction_buffer"]) == buffer_size and
            user_state["prediction_buffer"].count(letter) == buffer_size and
            letter != user_state["last_added_letter"]):
            user_state["buffer"] += letter
            user_state["last_added_letter"] = letter
            user_state["last_action_time"] = time.time()
            user_state["prediction_buffer"].clear()

    if (time.time() - user_state["last_action_time"] > space_interval and
        not user_state["buffer"].endswith(" ")):
        user_state["buffer"] += " "
        user_state["cleaned"] = smart_correct(user_state["buffer"])
        user_state["last_action_time"] = time.time()
        user_state["last_added_letter"] = ""
        save_history(user_id, user_state["buffer"], user_state["cleaned"])

    return {"status": "ok"}

@app.post("/clean")
async def clean_text(req: Request):
    data = await req.json()
    user_id = data.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    user_state = get_user_state(user_id)
    user_state["cleaned"] = smart_correct(user_state["buffer"])
    save_history(user_id, user_state["buffer"], user_state["cleaned"])
    return {"cleaned": user_state["cleaned"]}

@app.get("/buffer")
def get_buffer(user_id: str = Query(...)):
    user_state = get_user_state(user_id)
    return {"raw": user_state["buffer"], "cleaned": user_state["cleaned"]}

@app.post("/reset")
def reset(user_id: str = Query(...)):
    user_state = get_user_state(user_id)
    user_state["buffer"] = ""
    user_state["cleaned"] = ""
    user_state["prediction_buffer"].clear()
    user_state["last_added_letter"] = ""
    user_state["last_action_time"] = time.time()
    return {"status": "reset"}

@app.get("/history/{user_id}")
def get_history(user_id: str):
    docs = db.collection("history").document(user_id).collection("entries").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
    result = []
    for doc in docs:
        data = doc.to_dict()
        timestamp = data.get("timestamp")
        if timestamp:
            data["timestamp"] = timestamp.isoformat()
        result.append(data)
    return result

@app.delete("/history/{user_id}")
def delete_history(user_id: str):
    entries_ref = db.collection("history").document(user_id).collection("entries")
    docs = entries_ref.stream()
    for doc in docs:
        doc.reference.delete()
    return {"message": "History cleared."}

@app.get("/status")
def get_status(user_id: str = Query(...)):
    user_state = get_user_state(user_id)
    return {
        "buffer": user_state["buffer"],
        "cleaned": user_state["cleaned"],
        "last_added_letter": user_state["last_added_letter"],
        "prediction_buffer": user_state["prediction_buffer"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
