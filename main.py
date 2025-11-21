import os
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "SmartMail AI Backend Running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# ---------------- SmartMail AI Generator with lightweight memory ----------------
class GenerateRequest(BaseModel):
    goal: str = Field(..., min_length=3)
    category: str = Field(..., pattern=r"^(Professional|Food / Delivery|Banking / Finance|Other)$")
    tone: str = Field(..., pattern=r"^(Formal|Semi-Formal|Friendly|Apologetic|Assertive)$")
    detail: str = Field(..., pattern=r"^(Concise|Standard|Detailed)$")


class GenerateResponse(BaseModel):
    subject: str
    body: str
    output: str
    summary: str


STOP_WORDS = set(
    "to for the a an and of on at with about regarding please kindly my our your in from by due as is are was were be been being it this that these those I we you they us them".split()
)


def _to_title_case(s: str) -> str:
    words = re.sub(r"[\[\](){},.!?;:]", " ", s).split()
    words = [w.lower() for w in words]
    if not words:
        return ""
    out = []
    for i, w in enumerate(words):
        if i == 0:
            out.append(w.capitalize())
        else:
            out.append(w)
    return " ".join(out)


def extract_goal_phrase(goal: str) -> str:
    tokens = re.sub(r"[^A-Za-z0-9 ]+", " ", goal).split()
    filtered = [t for t in tokens if t.lower() not in STOP_WORDS]
    phrase = " ".join(filtered[:8]) or goal.strip()
    return _to_title_case(phrase)


def build_subject(category: str, tone: str, goal_phrase: str, memory_hint: Optional[str] = None) -> str:
    base = (
        "Order Support" if category == "Food / Delivery" else
        "Account Assistance" if category == "Banking / Finance" else
        "Follow-up" if category == "Professional" else
        "Request"
    )
    tone_mod = (
        " – Action Required" if tone == "Assertive" else
        " – Apologies and Resolution" if tone == "Apologetic" else
        ""
    )
    hint = f" – {goal_phrase}" if goal_phrase else ""
    mem = f" ({memory_hint})" if memory_hint else ""
    return f"{base}{tone_mod}{hint}{mem}".strip()


def build_body(category: str, tone: str, detail: str, goal_phrase: str) -> str:
    # Greeting
    if category in ("Food / Delivery", "Banking / Finance"):
        greeting = "Dear Customer Support Team,"
    elif tone == "Friendly":
        greeting = "Hi [Name],"
    elif tone == "Semi-Formal":
        greeting = "Hello [Name],"
    else:
        greeting = "Dear [Name],"

    # Opener
    if tone == "Apologetic":
        opener = f"I’d like to acknowledge the situation concerning {goal_phrase.lower()} and propose a constructive way forward."
    elif tone == "Assertive":
        opener = f"I’m contacting you regarding {goal_phrase.lower()} and request timely action to progress this to resolution."
    else:
        opener = f"I’m reaching out regarding {goal_phrase.lower()}."

    # Acknowledgement
    if category == "Professional":
        ack = "I recognize competing priorities and appreciate your partnership in keeping us aligned."
    elif category == "Food / Delivery":
        ack = "I understand your team manages high volumes daily and appreciate your attention to this."
    elif category == "Banking / Finance":
        ack = "I understand compliance and verification steps are essential and appreciate your careful review."
    else:
        ack = "I appreciate the time and attention this may require on your side."

    # Context and references
    if category == "Food / Delivery":
        ctx = "Recently, I experienced an issue that affected the expected delivery experience. Key references:"
        refs = "Order ID: [XXXX-XXXX] • Date: [DD/MM/YYYY] • Restaurant: [Name] • Issue: [Brief description]."
    elif category == "Banking / Finance":
        ctx = "I noticed an inconsistency that warrants a closer review to ensure account accuracy. Key references:"
        refs = "Reference: [Transaction ID] • Date: [DD/MM/YYYY] • Amount: [$ / ₹] • Channel: [Card/UPI/NetBanking]."
    elif category == "Professional":
        ctx = "To maintain momentum, here is a concise status and the actions proposed. Reference:"
        refs = "[Project / Ticket] • Timeline: [Milestones & Dates] • Stakeholders: [Names]."
    else:
        ctx = "Below is a brief context along with the assistance I’m seeking. Reference:"
        refs = "[Case/Topic] • Date: [DD/MM/YYYY] • Notes: [Short context]."

    # Request
    if category == "Food / Delivery":
        req = f"Request: Please review the details related to {goal_phrase.lower()} and advise on a suitable resolution. A replacement or refund would be appropriate given the impact."
    elif category == "Banking / Finance":
        req = f"Request: Please investigate the matter concerning {goal_phrase.lower()} and share findings, including any actions required from my side."
    elif category == "Professional":
        req = f"Request: Let’s confirm ownership, finalize the timeline, and proceed with the next actions aligned to {goal_phrase.lower()}."
    else:
        req = f"Request: Please advise on the most effective next steps to progress {goal_phrase.lower()}."

    firm = "Timeline: A response by [DD/MM/YYYY] would be appreciated to maintain pace." if tone == "Assertive" else ""
    thanks = "I regret any inconvenience caused and value your understanding." if tone == "Apologetic" else "Thank you for your time and support."

    blocks = {
        "Concise": [greeting, opener, ack, req, f"{thanks} {firm}".strip()],
        "Standard": [greeting, opener, ack, f"{ctx}", refs, req, f"{thanks} {firm}".strip()],
        "Detailed": [greeting, opener, ack, f"{ctx} Summary:", f"• Objective: {goal_phrase}.", f"• Key details — {refs}", f"{req} If documentation or verification is needed, I can provide it promptly.", f"{thanks} {firm}".strip()],
    }
    return "\n\n".join([b for b in blocks.get(detail, blocks["Standard"]) if b])


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]


def _similarity(a: str, b: str) -> float:
    sa, sb = set(_tokenize(a)), set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


@app.post("/api/generate", response_model=GenerateResponse)
def generate_email(payload: GenerateRequest):
    goal = payload.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="Goal cannot be empty")

    # Lightweight memory: use last 15 emails to guide subject hinting
    memory_hint: Optional[str] = None
    try:
        past = get_documents("email", {}, limit=15) if db else []
        if past:
            scored = []
            for p in past:
                g = p.get("goal", "")
                sim = _similarity(goal, g)
                scored.append((sim, p))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored and scored[0][0] > 0.15:  # modest threshold
                best = scored[0][1]
                memory_hint = best.get("summary") or extract_goal_phrase(best.get("goal", ""))
    except Exception:
        memory_hint = None

    goal_phrase = extract_goal_phrase(goal)
    subject = build_subject(payload.category, payload.tone, goal_phrase, memory_hint)
    body = build_body(payload.category, payload.tone, payload.detail, goal_phrase)
    output = f"Subject: {subject}\n\n{body}\n\nBest regards,\n[Your Name]"

    # Persist the generated email
    try:
        doc = {
            "goal": goal,
            "category": payload.category,
            "tone": payload.tone,
            "detail": payload.detail,
            "output": output,
            "subject": subject,
            "summary": goal_phrase,
        }
        if db is not None:
            create_document("email", doc)
    except Exception:
        pass

    return GenerateResponse(subject=subject, body=body, output=output, summary=goal_phrase)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
