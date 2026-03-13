import os, hashlib, tempfile, re, json, requests
from datetime import datetime
from typing import List, Optional, Dict, Any, TypedDict
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from bs4 import BeautifulSoup

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

# ─────────────────────── ENV ───────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY",   "").strip()
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY","") or os.environ.get("GOOGLE_API_KEY","")).strip()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
GOOGLE_CSE_ID  = os.environ.get("GOOGLE_CSE_ID",  "").strip()

# ─────────────────────── APP ───────────────────────
app = FastAPI(title="Multi-Agentic RAG Study Assistant")
templates = Jinja2Templates(directory="templates")
os.makedirs("vector_store", exist_ok=True)
os.makedirs("templates",    exist_ok=True)

VECTOR_DIR = "vector_store"

MODEL_OPTIONS = {
    "llama-3.3-70b-versatile":       "Llama 3.3 70B Versatile",
    "llama-3.1-8b-instant":          "Llama 3.1 8B Instant",
    "llama3-70b-8192":               "Llama3 70B",
    "gemma2-9b-it":                  "Gemma2 9B",
    "mistral-saba-24b":              "Mistral Saba 24B",
    "deepseek-r1-distill-llama-70b": "DeepSeek R1 Distill 70B",
}

# ─────────────────────── GLOBAL STATE ───────────────────────
G: Dict[str, Any] = {
    "retriever":          None,
    "practice_questions": [],
    "langgraph_cache":    {},
}

# ─────────────────────── HELPERS ───────────────────────
def get_llm(model_id: str):
    return ChatGroq(
        model_name=model_id,
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=4096
    )

def get_embedding():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

def pdf_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def llm_text(resp) -> str:
    return getattr(resp, "content", str(resp))

def check_context(retriever, query: str, threshold: int = 1) -> bool:
    """Returns True if retriever has at least `threshold` relevant docs."""
    if not retriever:
        return False
    try:
        docs = retriever.invoke(query[:300])
        return len(docs) >= threshold
    except Exception:
        return False

# ─────────────────────── DAY EXTRACTOR ───────────────────────
def extract_exam_days(text: str) -> Optional[str]:
    text_l = text.lower()
    if re.search(r'\b(tomorrow|tonite|tonight)\b', text_l):
        return "1"
    if re.search(r'\btoday\b', text_l):
        return "1"
    patterns = [
        r'(?:exam|test|exam\s+is)?\s*(?:in|after|within)\s+(\d+)\s+days?',
        r'(\d+)\s+days?\s+(?:left|remaining|to\s+(?:go|prepare|study))',
        r'(?:have|got|only)\s+(\d+)\s+days?',
        r'(\d+)\s*-\s*day\s+(?:plan|schedule|study)',
        r'(\d+)\s+days?\s+(?:for\s+(?:exam|test|prep))',
    ]
    for pat in patterns:
        m = re.search(pat, text_l)
        if m:
            return m.group(1)
    return None

# ─────────────────────── TRUNCATE EXTRA DAYS ───────────────────────
def truncate_to_n_days(text: str, n: int) -> str:
    day_pattern = re.compile(r'(?=\*{0,2}Day\s+\d+\b)', re.IGNORECASE)
    parts = day_pattern.split(text)
    if len(parts) <= 1:
        return text
    preamble  = parts[0]
    day_parts = parts[1:]
    kept      = day_parts[:n]
    result    = preamble + "".join(kept)
    result    = re.sub(
        r'\b(7|four|4)[- ]?(day|week)[s]?\b.*?plan',
        f'{n}-day plan',
        result, flags=re.IGNORECASE
    )
    return result.strip()

# ─────────────────────── MCQ PARSER ───────────────────────
def parse_mcqs(raw: str) -> list:
    questions = []
    blocks    = re.split(r'\n(?=Question\s*\d+\s*:)', raw.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        q = {}
        m = re.match(r'Question\s*\d+\s*:\s*(.*?)(?=\noptions|\n[a-d]\))', block, re.I | re.S)
        q["question"] = m.group(1).strip() if m else "N/A"
        opts = {}
        for l in "abcd":
            om = re.search(rf'{l}\)(.*?)(?=\n[a-d]\)|\ncorrect|\nhint|\nsolution|$)', block, re.I | re.S)
            opts[l] = om.group(1).strip() if om else ""
        q["options"] = opts
        cm = re.search(r'correct answer\s*:\s*([a-d])\)(.*?)(?=\nhint|\nsolution|$)', block, re.I | re.S)
        q["correct_letter"] = cm.group(1).lower().strip() if cm else ""
        q["correct_full"]   = cm.group(2).strip()         if cm else ""
        hm = re.search(r'hint\s*:\s*(.*?)(?=\nsolution\s*:|$)', block, re.I | re.S)
        q["hint"]     = hm.group(1).strip() if hm else ""
        sm = re.search(r'solution\s*:\s*(.*?)(?=\n---|\Z)', block, re.I | re.S)
        q["solution"] = sm.group(1).strip() if sm else ""
        if q["question"] != "N/A":
            questions.append(q)
    return questions

# ─────────────────────── LANGGRAPH STATE ───────────────────────
class AgentState(TypedDict):
    messages:      List[BaseMessage]
    next_tool:     str
    subtool:       str
    mode:          str
    concept_level: str
    exam_days:     str

# ═══════════════════════════════════════════════════════════════
#  LANGGRAPH BUILDER
#  Mode → Allowed agents mapping:
#  ┌─────────────────────┬────────────────────────────────────────────────────┐
#  │ All-in-one Chat     │ ALL agents (auto-classified)                       │
#  │ Syllabus Explainer  │ search_agent → summarizer/notes_maker/exam_prep    │
#  │ Concept Explainer   │ search_agent → summarizer/concept_explainer        │
#  │ Practice Mode       │ search_agent → mcq_generator/concept/exam_prep/sum │
#  └─────────────────────┴────────────────────────────────────────────────────┘
# ═══════════════════════════════════════════════════════════════
def build_graph(model_id: str):
    if model_id in G["langgraph_cache"] and G["retriever"] is not None:
        return G["langgraph_cache"][model_id]

    selected_model = get_llm(model_id)
    retriever      = G["retriever"]

    def combined_context(query: str, k: int = 20) -> str:
        if not retriever:
            return ""
        docs = retriever.invoke(query)
        return "\n\n".join(
            f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}] {d.page_content}"
            for d in docs[:k]
        )

    # ══════════════════════════════
    #  TOOLS
    # ══════════════════════════════

    @tool
    def summarizer(query: str, context: str = "") -> str:
        """Summarize topic using retrieved document chunks or provided context."""
        lm      = re.search(r'in\s+(\d+)\s+lines', query.lower())
        lines   = int(lm.group(1)) if lm else 10
        content = context if context else combined_context(query)
        if not content:
            return f"No relevant content found for '{query}'."
        resp = selected_model.invoke(
            f"Summarize strictly from the document. ~{lines} lines, paragraph form:\n\n{content}"
        )
        return llm_text(resp)

    @tool
    def mcq_generator(query: str, context: str = "") -> str:
        """Generate MCQs from document content. Used in Practice Mode and All-in-one Chat."""
        cm      = re.search(r'(\d+)\s*(?:mcqs|mcq|questions)', query.lower())
        count   = int(cm.group(1)) if cm else 5
        content = context if context else "\n\n".join(
            d.page_content for d in (retriever.invoke(query) if retriever else [])
        )
        if not content:
            return "No relevant content found."
        resp = selected_model.invoke(
            f"Generate {count} MCQs.\n"
            f"Format: Question N: [text]\noptions\na)[A]\nb)[B]\nc)[C]\nd)[D]\n\n"
            f"correct answer: [letter])[text]\n\nexplanation:\n[line1]\n[line2]\n\n\n"
            f"Content:\n{content}"
        )
        return llm_text(resp)

    @tool
    def notes_maker(query: str, context: str = "") -> str:
        """Create structured notes. Used in Syllabus Explainer and All-in-one Chat."""
        lm      = re.search(r'in\s+(\d+)\s+lines', query.lower())
        lines   = int(lm.group(1)) if lm else 20
        content = context if context else combined_context(query)
        if not content:
            return f"No relevant content found for '{query}'."
        resp = selected_model.invoke(
            f"Create concise notes (~{lines} lines).\n"
            f"Use: ## Key Concept  ## Formula  ## Important Fact  ## Exam Tip\n"
            f"Short paragraphs.\n\nContext:\n{content}"
        )
        return llm_text(resp)

    @tool
    def exam_prep_agent(query: str, context: str = "") -> str:
        """
        Build study plan.
        Used in: Syllabus Explainer (with [SYLLABUS MODE]), Practice Mode, All-in-one Chat.
        [EXAM_DAYS: N] pins exact day count.
        """
        is_syllabus = "[SYLLABUS MODE]" in query
        clean       = query.replace("[SYLLABUS MODE]", "").strip()

        days_match = re.search(r'\[EXAM_DAYS:\s*(\d+)\]', clean)
        exam_days  = int(days_match.group(1)) if days_match else None
        clean      = re.sub(r'\[EXAM_DAYS:\s*\d+\]', '', clean).strip()

        if not exam_days:
            nl_days   = extract_exam_days(clean)
            exam_days = int(nl_days) if nl_days else 7

        content = context if context else combined_context(clean)
        if not content:
            return f"No relevant content found for '{clean}'."

        day_scaffold = "\n".join([
            f"Day {i} of {exam_days}: [Fill topics, tasks, study hours]"
            for i in range(1, exam_days + 1)
        ])

        urgency = (
            f"EMERGENCY — only {exam_days} day(s) left. Cover ONLY High priority topics."
            if exam_days <= 3 else
            f"Student has {exam_days} days. Distribute topics evenly across all {exam_days} days."
        )

        if is_syllabus:
            prompt = (
                f"You are a strict study planner. {urgency}\n\n"
                f"HARD RULE: Output EXACTLY {exam_days} day(s). NOT 7. NOT 4. EXACTLY {exam_days}.\n\n"
                f"## 1. Topic-wise Overview\n"
                f"2-3 lines per topic. Mark each: High / Medium / Low priority.\n"
                f"{'With only {exam_days} day(s): skip Low priority topics entirely.' if exam_days <= 3 else ''}\n\n"
                f"## 2. Recommended Study Order\n"
                f"Ordered list, High priority first.\n\n"
                f"## 3. {exam_days}-Day Revision Plan\n"
                f"Fill ONLY these {exam_days} day slots. Do NOT add more:\n\n"
                f"{day_scaffold}\n\n"
                f"Each day: specific topics, study hours, tasks.\n\n"
                f"## 4. Must-Know Concepts\n"
                f"Top 5-8 critical concepts.\n\n"
                f"## 5. Exam Tips\n"
                f"3-4 practical exam day tips.\n\n"
                f"FORBIDDEN: Do NOT write a 7-day plan. Do NOT add Day {exam_days+1} or beyond.\n\n"
                f"Content:\n{content}"
            )
        else:
            prompt = (
                f"You are a strict study planner. {urgency}\n\n"
                f"HARD RULE: Output EXACTLY {exam_days} day(s). NOT 7. EXACTLY {exam_days}.\n\n"
                f"## 1. Quick Topic Overview\n"
                f"One line per topic. Mark High / Medium / Low priority.\n\n"
                f"## 2. {exam_days}-Day Crash Plan\n"
                f"Fill ONLY these {exam_days} day slots:\n\n"
                f"{day_scaffold}\n\n"
                f"Each day: Morning / Afternoon / Evening sessions with specific topics + 10-min breaks every 1.5hrs.\n\n"
                f"## 3. Must-Know Concepts\n"
                f"Top 5-8 things to memorize.\n\n"
                f"## 4. Exam Tips\n"
                f"3-4 practical tips for exam day.\n\n"
                f"FORBIDDEN: Do NOT add Day {exam_days+1} or beyond.\n\n"
                f"Content:\n{content}"
            )

        raw    = llm_text(selected_model.invoke(prompt))
        result = truncate_to_n_days(raw, exam_days)
        result += f"\n\n---\n✅ This is your {exam_days}-day plan. Good luck! 🎯"
        return result

    @tool
    def concept_explainer(query: str, context: str = "") -> str:
        """
        Explain concept at controlled difficulty.
        Used in: Concept Explainer (with [LEVEL:X]), Practice Mode, All-in-one Chat.
        """
        level = "default"
        lm    = re.search(r'\[LEVEL:\s*(.*?)\]', query)
        if lm:
            level = lm.group(1).strip()
            query = re.sub(r'\[LEVEL:.*?\]', '', query).strip()
        content = context if context else combined_context(query)
        if not content:
            return f"No relevant content found for '{query}'."
        line_m     = re.search(r'(\d+)\s*lines?', query, re.I)
        line_instr = f"Answer in {int(line_m.group(1))} lines." if line_m else "Give a clear explanation."
        styles = {
            "beginner":    "Use very simple language, relatable analogies, avoid heavy notation.",
            "intermediate":"Be moderately formal. Include key formulas and one worked example.",
            "exam-ready":  "Be precise: exact definitions, formulas, edge cases, common mistakes, answer template.",
        }
        style = styles.get(level.lower(), "Give a clear balanced explanation for a college student.")
        resp  = selected_model.invoke(
            f"Style: {style}\n\nExplain '{query}' using ONLY this context:\n\n{content}\n\n{line_instr}"
        )
        return llm_text(resp)

    @tool
    def search_agent(query: str) -> dict:
        """
        Google Custom Search fallback when PDF context is insufficient.
        Used in ALL modes as fallback. Returns content + suggested subtool.
        """
        search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID
        )
        try:
            results = search.results(query, num_results=10)
            parts   = []
            for res in results:
                try:
                    r    = requests.get(res["link"], timeout=5)
                    soup = BeautifulSoup(r.text, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)[:5000]
                    parts.append(f"Title: {res.get('title')}\nLink: {res.get('link')}\nContent: {text}")
                except Exception as fe:
                    parts.append(f"Title: {res.get('title')}\nSnippet: {res.get('snippet','')}\nError:{fe}")
            text   = "\n\n".join(parts)
            prompt = (
                f"Query: {query}\n\nExtract and organize relevant information.\n\n"
                f"Determine subtool: summarizer/mcq_generator/notes_maker/exam_prep_agent/concept_explainer/none\n"
                f"End your response with exactly: Final Subtool Decision: [subtool_name]\n\n"
                f"Content:\n{text}"
            )
            resp    = selected_model.invoke(prompt)
            content = llm_text(resp)
            sm      = re.search(r'Final Subtool Decision:\s*\[(\w+)\]', content, re.I)
            subtool = sm.group(1).lower() if sm else "none"
            valid   = {"summarizer","mcq_generator","notes_maker","exam_prep_agent","concept_explainer"}
            if subtool not in valid:
                ql      = query.lower()
                subtool = next((t for t, kws in {
                    "summarizer":        ["summarize","summary","overview","brief"],
                    "mcq_generator":     ["mcq","question","quiz","practice"],
                    "notes_maker":       ["notes","key points","revision"],
                    "exam_prep_agent":   ["prepare","exam","study plan","days"],
                    "concept_explainer": ["explain","what is","define","how"],
                }.items() if any(k in ql for k in kws)), "none")
            content = re.sub(r'Final Subtool Decision:.*', '', content, flags=re.I | re.S).strip()
            sources = "\n\nSources:\n" + "\n".join(
                f"- {r.get('title','')}: {r.get('link','')}" for r in results
            )
            return {"content": content + sources, "subtool": subtool}
        except Exception as e:
            return {"content": f"Search error: {str(e)}", "subtool": "none"}

    all_tools = [summarizer, mcq_generator, notes_maker, exam_prep_agent, concept_explainer, search_agent]

    # ══════════════════════════════════════════════════════════
    #  ROUTER
    #  Implements mode-aware agent selection:
    #
    #  All-in-one Chat     → LLM classifies → any of 6 agents
    #  Syllabus Explainer  → has_context: exam_prep/summarizer/notes_maker
    #                        no_context:  search_agent (subtool picks from above 3)
    #  Concept Explainer   → has_context: concept_explainer/summarizer
    #                        no_context:  search_agent (subtool picks from above 2)
    #  Practice Mode       → has_context: mcq_generator/concept_explainer/exam_prep/summarizer
    #                        no_context:  search_agent (subtool picks from above 4)
    # ══════════════════════════════════════════════════════════
    def route_agent(state: AgentState):
        query_text   = state["messages"][-1].content
        current_mode = state.get("mode", "🤖 All-in-one Chat")
        level        = state.get("concept_level", "")
        exam_days    = state.get("exam_days", "")
        tool_name    = "search_agent"

        # Check if retriever has relevant content for this query
        has_context = check_context(retriever, query_text)
        ql          = query_text.lower()

        # ── SYLLABUS EXPLAINER ──────────────────────────────
        if current_mode == "📖 Syllabus Explainer":
            query_text = f"[SYLLABUS MODE] {query_text}"
            if exam_days:
                query_text = f"{query_text} [EXAM_DAYS: {exam_days}]"

            if has_context:
                if any(w in ql for w in ["notes", "key points", "bullet", "make notes", "point form"]):
                    tool_name = "notes_maker"
                elif any(w in ql for w in ["summarize", "summary", "brief", "condense", "overview"]):
                    tool_name = "summarizer"
                else:
                    # Default for syllabus: exam_prep_agent (study plan + priorities)
                    tool_name = "exam_prep_agent"
            else:
                # No PDF context → web search → subtool picks best from {exam_prep/summarizer/notes_maker}
                tool_name = "search_agent"

        # ── CONCEPT EXPLAINER ───────────────────────────────
        elif current_mode == "💡 Concept Explainer":
            if level:
                query_text = f"{query_text} [LEVEL: {level}]"

            if has_context:
                if any(w in ql for w in ["summarize", "summary", "brief", "overview", "condense"]):
                    tool_name = "summarizer"
                else:
                    # Default for concept: concept_explainer
                    tool_name = "concept_explainer"
            else:
                # No PDF context → web search → subtool picks {concept_explainer/summarizer}
                tool_name = "search_agent"

        # ── PRACTICE MODE ───────────────────────────────────
        elif current_mode == "📝 Practice Mode":
            nl_days = extract_exam_days(query_text)

            if has_context:
                if any(w in ql for w in ["explain","what is","define","how does","concept","understand"]):
                    tool_name  = "concept_explainer"
                    if level:
                        query_text = f"{query_text} [LEVEL: {level}]"
                elif any(w in ql for w in ["plan","schedule","prepare","study plan","days","exam in"]) or nl_days:
                    tool_name  = "exam_prep_agent"
                    if nl_days:
                        query_text = f"{query_text} [EXAM_DAYS: {nl_days}]"
                elif any(w in ql for w in ["summarize","summary","notes","overview","revision notes"]):
                    tool_name = "summarizer"
                else:
                    # Default for practice: mcq_generator
                    tool_name = "mcq_generator"
            else:
                # No PDF context → web search → subtool picks {mcq/concept/exam_prep/summarizer}
                tool_name = "search_agent"

        # ── ALL-IN-ONE CHAT (all agents available) ──────────
        else:
            nl_days = extract_exam_days(query_text)

            cls_prompt = (
                f"Classify this query to ONE tool: "
                f"search_agent, summarizer, mcq_generator, notes_maker, exam_prep_agent, concept_explainer.\n"
                f"Query: '{query_text}'\nRespond ONLY with the tool name."
            )
            try:
                raw = llm_text(selected_model.invoke(cls_prompt)).strip().lower()
                if   "search"    in raw or "web"      in raw: tool_name = "search_agent"
                elif "summariz"  in raw:                      tool_name = "summarizer"
                elif "mcq"       in raw or "question" in raw: tool_name = "mcq_generator"
                elif "notes"     in raw:                      tool_name = "notes_maker"
                elif "exam"      in raw or "prep"     in raw: tool_name = "exam_prep_agent"
                elif "concept"   in raw or "explain"  in raw: tool_name = "concept_explainer"
                else:                                         tool_name = "search_agent"
            except Exception:
                if   any(w in ql for w in ["web","search","internet","online","google","latest"]):  tool_name = "search_agent"
                elif any(w in ql for w in ["summarize","summary","overview","condense","brief"]):   tool_name = "summarizer"
                elif any(w in ql for w in ["mcq","mcqs","question","quiz","generate questions"]):   tool_name = "mcq_generator"
                elif any(w in ql for w in ["notes","make notes","revision notes","key points"]):    tool_name = "notes_maker"
                elif any(w in ql for w in ["prepare","exam","study plan","revision plan"]):         tool_name = "exam_prep_agent"
                elif any(w in ql for w in ["explain","what is","define","how does"]):               tool_name = "concept_explainer"
                else:                                                                                tool_name = "search_agent"

            # Inject exam days if detected and routing to exam_prep
            if nl_days and tool_name == "exam_prep_agent":
                query_text = f"{query_text} [EXAM_DAYS: {nl_days}]"

            # Force route to exam_prep if days detected but mis-classified
            if nl_days and tool_name not in ("exam_prep_agent", "mcq_generator", "notes_maker"):
                if any(w in ql for w in ["help","plan","study","prepare","ready","revise","crash"]):
                    tool_name  = "exam_prep_agent"
                    query_text = f"{query_text} [EXAM_DAYS: {nl_days}]"

        updated = state["messages"][:-1] + [HumanMessage(content=query_text)]
        tc = {
            "name": tool_name,
            "args": {"query": query_text},
            "id":   f"call_{tool_name}_{hashlib.md5(query_text.lower().encode()).hexdigest()[:8]}"
        }
        return {
            "messages":      updated + [AIMessage(content="", tool_calls=[tc])],
            "next_tool":     tool_name,
            "subtool":       "none",
            "mode":          current_mode,
            "concept_level": level,
            "exam_days":     exam_days,
        }

    # ══════════════════════════════════════════════════════════
    #  SUBTOOL ROUTER
    #  After search_agent returns, picks the right subtool
    #  based on mode constraints:
    #  Syllabus   → exam_prep / summarizer / notes_maker
    #  Concept    → concept_explainer / summarizer
    #  Practice   → mcq_generator / concept_explainer / exam_prep / summarizer
    #  All-in-one → any subtool
    # ══════════════════════════════════════════════════════════
    def route_subtool(state: AgentState):
        if state["next_tool"] != "search_agent":
            return {"messages": state["messages"], "subtool": "none"}

        tool_msgs = [
            m for m in state["messages"]
            if isinstance(m, ToolMessage) and m.tool_call_id.startswith("call_search_agent")
        ]
        if not tool_msgs:
            return {"messages": state["messages"], "subtool": "none"}

        rc = tool_msgs[-1].content
        if not isinstance(rc, dict):
            try:    result = json.loads(rc)
            except: return {"messages": state["messages"], "subtool": "none"}
        else:
            result = rc

        subtool        = result.get("subtool", "none")
        search_content = result.get("content", "")
        current_mode   = state.get("mode", "🤖 All-in-one Chat")

        # Mode-specific allowed subtools (enforces agent scope per mode)
        mode_allowed = {
            "📖 Syllabus Explainer":  {"exam_prep_agent", "summarizer", "notes_maker"},
            "💡 Concept Explainer":   {"concept_explainer", "summarizer"},
            "📝 Practice Mode":       {"mcq_generator", "concept_explainer", "exam_prep_agent", "summarizer"},
            "🤖 All-in-one Chat":     {"summarizer", "mcq_generator", "notes_maker", "exam_prep_agent", "concept_explainer"},
        }
        allowed = mode_allowed.get(current_mode, mode_allowed["🤖 All-in-one Chat"])

        # If search suggested subtool is not allowed for this mode, pick best default
        if subtool not in allowed:
            mode_defaults = {
                "📖 Syllabus Explainer":  "exam_prep_agent",
                "💡 Concept Explainer":   "concept_explainer",
                "📝 Practice Mode":       "mcq_generator",
                "🤖 All-in-one Chat":     "summarizer",
            }
            subtool = mode_defaults.get(current_mode, "summarizer")

        if not search_content or subtool == "none":
            return {"messages": state["messages"], "subtool": "none"}

        query = state["messages"][0].content
        tc = {
            "name": subtool,
            "args": {"query": query, "context": search_content},
            "id":   f"call_{subtool}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        }
        return {
            "messages": state["messages"] + [AIMessage(content="", tool_calls=[tc])],
            "subtool":  subtool,
        }

    # ─── Compile Graph ───
    g = StateGraph(AgentState)
    g.add_node("router", RunnableLambda(route_agent))
    for t in all_tools:
        g.add_node(t.name, ToolNode([t]))
    g.add_node("subtool_router", RunnableLambda(route_subtool))

    g.add_conditional_edges("router", lambda s: s["next_tool"], {t.name: t.name for t in all_tools})
    g.add_edge("search_agent", "subtool_router")
    g.add_conditional_edges(
        "subtool_router",
        lambda s: s.get("subtool", "none"),
        {
            "summarizer":        "summarizer",
            "mcq_generator":     "mcq_generator",
            "notes_maker":       "notes_maker",
            "exam_prep_agent":   "exam_prep_agent",
            "concept_explainer": "concept_explainer",
            "none":              END,
        }
    )
    for t in all_tools:
        if t.name != "search_agent":
            g.add_edge(t.name, END)
    g.set_entry_point("router")

    compiled = g.compile()
    G["langgraph_cache"][model_id] = compiled
    return compiled

# ─────────────────────── SCHEMAS ───────────────────────
class ChatReq(BaseModel):
    query:         str
    mode:          str
    concept_level: Optional[str] = "Beginner"
    exam_days:     Optional[str] = None
    model_id:      str

class PracticeGenReq(BaseModel):
    topic:    str
    model_id: str

class HintReq(BaseModel):
    question_index: int

class AnswerDetail(BaseModel):
    question_index: int
    question:       str
    chosen_letter:  str
    user_option:    str
    correct_letter: str
    correct_full:   str
    solution:       str
    is_correct:     bool

class ReportReq(BaseModel):
    topic:    str
    answers:  List[AnswerDetail]
    model_id: str

# ═══════════════════════ ROUTES ═══════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models":  MODEL_OPTIONS,
    })

# ─── Upload ───
@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), model_id: str = Form(...)):
    try:
        emb = get_embedding()
        dbs = []
        for file in files:
            data = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(data)
                path = tmp.name
            h     = pdf_hash(path)
            vpath = os.path.join(VECTOR_DIR, h)
            if os.path.exists(vpath):
                db = FAISS.load_local(vpath, embeddings=emb, allow_dangerous_deserialization=True)
            else:
                loader = PyMuPDFLoader(path)
                docs   = loader.load()
                for d in docs:
                    d.metadata.update({
                        "source":      file.filename,
                        "hash":        h,
                        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
                db     = FAISS.from_documents(chunks, emb)
                db.save_local(vpath)
            dbs.append(db)
        if dbs:
            merged = dbs[0]
            for extra in dbs[1:]:
                merged.merge_from(extra)
            G["retriever"]       = merged.as_retriever(search_kwargs={"k": 10})
            G["langgraph_cache"] = {}
        return {"status": "success", "message": f"{len(files)} file(s) indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Chat ───
@app.post("/chat")
async def chat(req: ChatReq):
    try:
        if not G["retriever"]:
            raise HTTPException(status_code=400, detail="Please upload PDF files first.")
        graph = build_graph(req.model_id)
        init_state: AgentState = {
            "messages":      [HumanMessage(content=req.query)],
            "next_tool":     "",
            "subtool":       "",
            "mode":          req.mode,
            "concept_level": req.concept_level or "",
            "exam_days":     req.exam_days or "",
        }
        output = graph.invoke(init_state)
        final_response = None
        for msg in reversed(output["messages"]):
            if isinstance(msg, ToolMessage):
                final_response = msg.content
                break
        if final_response is None:
            raise HTTPException(status_code=500, detail="No response generated from agents.")
        if isinstance(final_response, dict):
            final_response = final_response.get("content", "No content available.")
        return {
            "status":   "success",
            "response": final_response,
            "agent":    output.get("next_tool", "unknown"),
            "subtool":  output.get("subtool",   "none"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Practice: Generate ───
@app.post("/practice/generate")
async def practice_generate(req: PracticeGenReq):
    try:
        if not G["retriever"]:
            raise HTTPException(status_code=400, detail="Please upload PDF files first.")
        llm  = get_llm(req.model_id)
        docs = G["retriever"].invoke(req.topic)
        if not docs:
            raise HTTPException(status_code=404, detail="No relevant content found for this topic.")
        content = "\n\n".join(d.page_content for d in docs)
        prompt  = (
            f"You are a study assistant in Practice Mode. Generate exactly 5 MCQ practice questions "
            f"on '{req.topic}' from the content below. Vary difficulty: Q1 easy → Q5 hard.\n\n"
            f"Use EXACTLY this format for every question:\n\n"
            f"Question N: [question text]\n\n"
            f"options\n"
            f"a)[option A]\n"
            f"b)[option B]\n"
            f"c)[option C]\n"
            f"d)[option D]\n\n"
            f"correct answer: [letter])[correct option text]\n\n"
            f"hint: [one sentence nudge — do NOT reveal the answer]\n\n"
            f"solution: [step-by-step explanation of the correct answer]\n\n"
            f"---\n\n"
            f"Content:\n{content}"
        )
        raw       = llm_text(llm.invoke(prompt))
        questions = parse_mcqs(raw)[:5]
        if not questions:
            raise HTTPException(status_code=500, detail="Could not parse MCQ output. Try again.")
        G["practice_questions"] = questions
        return {"status": "success", "questions": questions}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Practice: Hint ───
@app.post("/practice/hint")
async def practice_hint(req: HintReq):
    qs = G["practice_questions"]
    if req.question_index >= len(qs):
        raise HTTPException(status_code=404, detail="Question not found.")
    return {"hint": qs[req.question_index].get("hint", "No hint available.")}

# ─── Practice: Report ───
@app.post("/practice/report")
async def practice_report(req: ReportReq):
    try:
        llm           = get_llm(req.model_id)
        correct_count = sum(1 for a in req.answers if a.is_correct)
        wrong         = [a for a in req.answers if not a.is_correct]
        wrong_block   = ""
        for a in wrong:
            wrong_block += (
                f"\n❌ Q{a.question_index + 1}: {a.question}\n"
                f"   Student:  {a.chosen_letter}) {a.user_option}\n"
                f"   Correct:  {a.correct_letter}) {a.correct_full}\n"
                f"   Explanation: {a.solution}\n"
            )
        correct_qs    = [a.question for a in req.answers if a.is_correct]
        correct_block = "\n".join(f"✅ Q: {q}" for q in correct_qs) or "None"
        prompt = (
            f"A student completed a 5-question practice session on: '{req.topic}'.\n"
            f"Score: {correct_count}/5\n\n"
            f"Wrong answers:\n{wrong_block or 'None — all correct!'}\n\n"
            f"Correct answers:\n{correct_block}\n\n"
            f"Generate a detailed performance report with EXACTLY these sections:\n\n"
            f"## 📊 Score Summary\nInterpret the score in 2-3 lines.\n\n"
            f"## 🔍 Weak Area Analysis\nIdentify exact sub-concepts the student struggled with.\n\n"
            f"## 📚 Priority Topics to Revise\nList 3-5 concepts ranked by priority.\n\n"
            f"## 💡 Actionable Study Recommendations\nGive 3-4 concrete study tips.\n\n"
            f"## ✅ Strengths\nAcknowledge what the student clearly understands.\n"
        )
        report = llm_text(llm.invoke(prompt))
        return {
            "status":     "success",
            "score":      correct_count,
            "total":      len(req.answers),
            "percentage": round((correct_count / len(req.answers)) * 100),
            "report":     report,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
