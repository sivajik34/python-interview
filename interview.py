generate_python_qa_pdf#!/usr/bin/env python3
"""
generate_python_qa_pdf.py

Usage:
    - install dependencies:
        pip install langchain openai reportlab tqdm

    - set environment variables (example):
        export AZURE_OPENAI_API_KEY="..." 
        export AZURE_OPENAI_ENDPOINT="https://your-azure-endpoint"
        export AZURE_OPENAI_API_VERSION="2024-05-01-preview"
        # optional
        export OPENAI-API-VERSION="2024-05-01-preview"

    - run:
        python generate_python_qa_pdf.py
"""

import os
import json
import time
import re
import math
from datetime import datetime,timezone
from pathlib import Path
from time import sleep

# External libs
from langchain_openai import AzureChatOpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, ListFlowable, ListItem, Table, TableStyle, KeepTogether
)
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
# ---------------------------
# Config
# ---------------------------
QUESTIONS_TARGET = 1000                # change if you want >1000 or less
BATCH_SIZE = 20                        # how many q -> answer requests queued logically (requests are sequential below)
SAVE_CHECKPOINT_EVERY = 50            # persist JSON & incremental PDF every N answered Qs
PDF_FILENAME = "python_interview_full.pdf"
CHECKPOINT_JSON = "qa_checkpoint.json"
GENERATED_QUESTIONS_JSON = "generated_questions.json"
LOG_FILENAME = "run_log.txt"

# Azure/LLM parameters
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")  # default your deployed name
LLM_MAX_TOKENS = 16000
REQUEST_DELAY_SECONDS = 0.8            # polite delay between requests (avoid throttling). tune as needed.
MAX_RETRIES = 5

import spacy

# Load English model with word vectors
nlp = spacy.load("en_core_web_md")

def remove_semantic_duplicates(questions, similarity_threshold=0.80):
    """
    Removes semantically similar or duplicate questions using spaCy embeddings.
    """
    unique_questions = []
    unique_vectors = []

    for q in questions:
        doc = nlp(q)
        if not unique_vectors:
            unique_questions.append(q)
            unique_vectors.append(doc)
        else:
            # Compare with previously kept questions
            if all(doc.similarity(prev) < similarity_threshold for prev in unique_vectors):
                unique_questions.append(q)
                unique_vectors.append(doc)
    return unique_questions


# ---------------------------
# Initialize LLM
# ---------------------------
def init_llm():
    """Initialize AzureChatOpenAI using environment variables. Don't embed secrets here."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

    if not api_key or not endpoint:
        raise RuntimeError("Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT in environment.")

    llm = AzureChatOpenAI(
        deployment_name=AZURE_DEPLOYMENT_NAME,
        openai_api_version=api_version,
        openai_api_key=api_key,
        azure_endpoint=endpoint,
        max_tokens=LLM_MAX_TOKENS
    )
    return llm

# ---------------------------
# Utility: logging
# ---------------------------
def log(msg):    
    ts = datetime.now(timezone.utc).isoformat()

    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILENAME, "a") as f:
        f.write(line + "\n")

# ---------------------------
# 1) Generate Questions (if not provided)
# ---------------------------
def generate_questions(llm, count=1000, chunk=200):
    """
    Ask model to generate 'count' unique Python interview questions (plain lines).
    We generate in chunks to avoid overly long single prompts.
    """
    out = []
    already = set()
    needed = count
    attempts = 0
    while len(out) < count and attempts < 10:
        gen = min(chunk, needed)
        prompt = (
            f"Generate {gen} unique, clear, interview-style Python questions. "
            "Return each question as a single line without numbering. Cover topics from basics to advanced (OOP, concurrency, networking, "
            "databases, performance, debugging, typing, packaging, memory, design patterns, Python internals, data science relevant items, etc.)."
        )
        attempts += 1
        try:
            resp = llm.invoke(prompt).content
        except Exception as e:
            log(f"LLM failed to generate questions (attempt {attempts}): {e}")
            sleep(2 ** attempts)
            continue

        # parse lines
        lines = [l.strip("-. \t") for l in resp.splitlines() if l.strip()]
        for l in lines:
            # filter obviously long garbage or repeated
            if len(l) < 6:
                continue
            if l not in already:
                already.add(l)
                out.append(l)
                if len(out) >= count:
                    break
        needed = count - len(out)
        log(f"Generated so far: {len(out)}/{count}. Needed: {needed}.")
        sleep(1)
    if len(out) < count:
        log(f"Warning: only generated {len(out)} questions out of requested {count}.")
    # ---- 🧠 NEW STEP: semantic deduplication ----
    print(f"\n🧠 Performing semantic deduplication on {len(out)} questions...")
    out = remove_semantic_duplicates(out, similarity_threshold=0.90)
    print(f"✅ After semantic filtering: {len(out)} unique questions retained.\n")    
    return out[:count]

# ---------------------------
# 2) Fetch answer for a single question (with retries)
# ---------------------------
def fetch_answer(llm, question, max_retries=MAX_RETRIES):
    """
    Send the question to the LLM and return markdown string.
    Retries with exponential backoff on failure.
    """
    prompt = (
        "Answer this Python interview question in a detailed markdown format with headings, lists, tables, and code blocks. "
        "Use short illustrative examples and include an 'Explanation' and 'Examples' section when applicable.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    attempt = 0
    while attempt < max_retries:
        try:
            attempt += 1
            resp = llm.invoke(prompt).content
            if resp and resp.strip():
                return resp
            else:
                raise RuntimeError("Empty response")
        except Exception as e:
            wait = (2 ** attempt) + (attempt * 0.5)
            log(f"Error fetching answer (attempt {attempt}) for question: {question[:60]}... -> {e}. Retrying in {wait}s.")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch answer after {max_retries} retries for question: {question[:80]}")

# ---------------------------
# 3) Checkpoint utilities
# ---------------------------
def save_checkpoint(qa_pairs, questions_list):
    tmp = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "count": len(qa_pairs),
        "qa_pairs": qa_pairs
    }
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump(tmp, f, indent=2, ensure_ascii=False)
    with open(GENERATED_QUESTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump({"questions": questions_list}, f, indent=2, ensure_ascii=False)
    log(f"Checkpoint saved. {len(qa_pairs)} QAs persisted.")

def load_checkpoint():
    if Path(CHECKPOINT_JSON).exists():
        with open(CHECKPOINT_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("qa_pairs", [])
    return []

# ---------------------------
# 4) Markdown -> ReportLab renderer (improved)
# ---------------------------
def make_styles():
    styles = getSampleStyleSheet()
    custom_styles = {
        "QuestionTitle": ParagraphStyle(name="QuestionTitle", fontSize=14, leading=18, spaceAfter=8, textColor=colors.darkblue, fontName="Helvetica-Bold"),
        "H1": ParagraphStyle(name="H1", fontSize=13, leading=16, spaceBefore=8, spaceAfter=6, fontName="Helvetica-Bold"),
        "H2": ParagraphStyle(name="H2", fontSize=12, leading=14, spaceBefore=6, spaceAfter=4, fontName="Helvetica-Bold"),
        "H3": ParagraphStyle(name="H3", fontSize=11, leading=13, spaceBefore=6, spaceAfter=3, fontName="Helvetica-Bold"),
        "Body": ParagraphStyle(name="Body", fontSize=10.5, leading=14, spaceAfter=6),
        "CodeBlock": ParagraphStyle(name="CodeBlock", fontName="Courier", fontSize=9.0, leading=11, backColor=colors.whitesmoke, leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=6),
        "ListItem": ParagraphStyle(name="ListItem", fontSize=10.5, leftIndent=12, bulletIndent=3, spaceBefore=2, spaceAfter=2),
    }
    for s in custom_styles.values():
        styles.add(s)
    return styles

def render_markdown(md_text, styles):
    """
    Render Markdown into ReportLab flowables.
    Handles headings (#, ##, ###), unordered lists (- or *), ordered lists (1.), code fences ``` ```
    and simple markdown tables. Skips table separator rows like '|---|---|'.
    """
    elements = []
    lines = md_text.splitlines()
    in_code = False
    code_buffer = []
    bullet_buffer = []
    number_buffer = []
    table_buffer = []
    
    def flush_bullets():
        if bullet_buffer:
            items = []
            for item in bullet_buffer:
                txt = _inline_format(item)
                items.append(ListItem(Paragraph(txt, styles["ListItem"])))
            elements.append(ListFlowable(items, bulletType="bullet"))
            bullet_buffer.clear()

    def flush_numbers():
        if number_buffer:
            items = []
            for item in number_buffer:
                txt = _inline_format(item)
                items.append(ListItem(Paragraph(txt, styles["ListItem"])))
            elements.append(ListFlowable(items, bulletType="1"))
            number_buffer.clear()

    def flush_table():
        if table_buffer:
            # remove separator row(s) like | --- | --- |
            filtered = []
            for r in table_buffer:
                # split keep pipes
                cols = [c.strip() for c in r.split("|")[1:-1]]
                # detect if separator row
                if all(re.match(r"^:?-+:?$", c.replace(" ", "")) for c in cols):
                    continue
                filtered.append(cols)
            if filtered:
                # create table
                table = Table(filtered, hAlign='LEFT')
                table.setStyle(TableStyle([
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('LEFTPADDING', (0,0), (-1,-1), 6),
                    ('RIGHTPADDING', (0,0), (-1,-1), 6)
                ]))
                elements.append(table)
            table_buffer.clear()

    def safe_format(text):
        # 1. Escape HTML
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        # 2. Replace inline code with font block
        text = re.sub(r"`([^`]*)`", r"<font face='Courier' backcolor='#f0f0f0'>\1</font>", text)
        
        # 3. Replace bold, but ignore bold inside <font> blocks
        def bold_replacer(match):
            content = match.group(1)
            if "<font" in content:  # skip if inside code
                return content
            return f"<b>{content}</b>"

        text = re.sub(r"\*\*(.*?)\*\*", bold_replacer, text)
        
        # 4. Remove remaining markdown chars like stray *
        text = text.replace("*", "")
        
        return text
        

    def _inline_format(stripped):
        if not stripped:
            return ""
        stripped = safe_format(stripped)
        return stripped

    for line in lines:
        stripped = line.rstrip("\n")
        s = stripped.strip()

        # code fence toggle
        if s.startswith("```"):
            if in_code:
                # close
                elements.append(Preformatted("\n".join(code_buffer), styles["CodeBlock"]))
                code_buffer.clear()
                in_code = False
            else:
                flush_bullets(); flush_numbers(); flush_table()
                in_code = True
            continue

        if in_code:
            code_buffer.append(line)
            continue

        # detect markdown table line: start and end with '|'
        if "|" in s and re.match(r"^\|.*\|$", s):
            flush_bullets(); flush_numbers()
            table_buffer.append(s)
            continue
        else:
            flush_table()

        # headings
        if s.startswith("# "):
            flush_bullets(); flush_numbers()
            elements.append(Paragraph(s[2:].strip(), styles["H1"]))
        elif s.startswith("## "):
            flush_bullets(); flush_numbers()
            elements.append(Paragraph(s[3:].strip(), styles["H2"]))
        elif s.startswith("### "):
            flush_bullets(); flush_numbers()
            elements.append(Paragraph(s[4:].strip(), styles["H3"]))
        # unordered list
        elif re.match(r"^[-*]\s+", s):
            bullet_buffer.append(re.sub(r"^[-*]\s+", "", s))
        # ordered list
        elif re.match(r"^\d+\.\s+", s):
            number_buffer.append(re.sub(r"^\d+\.\s+", "", s))
        elif s:
            flush_bullets(); flush_numbers()
            elements.append(Paragraph(_inline_format(s), styles["Body"]))
        else:
            flush_bullets(); flush_numbers(); flush_table()
            elements.append(Spacer(1, 2))
    # final flushes
    flush_bullets(); flush_numbers(); flush_table()
    return elements

# ---------------------------
# 5) Build PDF incrementally
# ---------------------------
def build_pdf_incremental(qa_pairs, filename=PDF_FILENAME):
    """
    Build full PDF from qa_pairs = list of dicts {"q":..., "a":...}
    Overwrites filename.
    """
    doc = SimpleDocTemplate(filename, pagesize=A4, title="Python Interview Questions and Answers",author="Koduru Sivakumar")
    story = []
    styles = make_styles()
    for idx, item in enumerate(qa_pairs, start=1):
        q = item["q"]
        a = item["a"]
        story.append(Paragraph(f"{idx}. {q}", styles["QuestionTitle"]))
        story.extend(render_markdown(a, styles))
        story.append(Spacer(1, 0.25*inch))
    doc.build(story)
    log(f"PDF written: {filename}")

# -----------------------------
# 3️⃣ Question Generation
# -----------------------------
topics = [
    "Basics", "OOP", "Concurrency", "Async IO", "Networking", "Databases",
    "Memory Management", "Design Patterns", "Python Internals", "Standard Library",
    "Debugging", "Typing / Type Hints", "Packaging & Modules", "Performance Optimization",
    "Data Science", "Machine Learning", "Testing & Unit Testing", "Web Development FastAPI",
    "Logging & Monitoring", "Security", "File I/O & Serialization", "Generators & Iterators",
    "Decorators & Metaclasses", "Context Managers", "Regex", "Functional Programming"
]

def generate_questions_by_topic(llm, questions_per_topic=50):
    all_questions = []
    for topic in topics:
        prompt = (
            f"Generate {questions_per_topic} unique, clear Python interview questions focusing on the topic: {topic}. "
            "Return each question as a single line without numbering. Avoid repeating questions."
        )
        print(f"📝 Generating questions for topic: {topic}")
        try:
            resp = llm.invoke(prompt).content
        except Exception as e:
            print(f"LLM failed for topic {topic}: {e}")
            sleep(2)
            continue

        lines = [l.strip("-. \t") for l in resp.splitlines() if l.strip()]
        all_questions.extend(lines)
        sleep(1)  # avoid rate-limiting

    # Remove exact duplicates first
    all_questions = list(dict.fromkeys(all_questions))
    print(f"Generated {len(all_questions)} questions before semantic filtering...")

    # Semantic deduplication
    all_questions = remove_semantic_duplicates(all_questions, similarity_threshold=0.80)
    print(f"✅ {len(all_questions)} unique questions after semantic filtering.")
    return all_questions



# ---------------------------
# 6) Main driver
# ---------------------------
def main():
    llm = init_llm()

    # load or generate questions
    if Path(GENERATED_QUESTIONS_JSON).exists():
        with open(GENERATED_QUESTIONS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
            questions = data.get("questions", [])
            log(f"Loaded {len(questions)} generated questions from {GENERATED_QUESTIONS_JSON}.")
    else:
        #log(f"Generating {QUESTIONS_TARGET} questions from LLM...")
        #questions = generate_questions(llm, count=QUESTIONS_TARGET)
        #print(f"✅ {len(questions)} semantically unique questions retained.")
        questions = generate_questions_by_topic(llm, questions_per_topic=50)  
        

        with open(GENERATED_QUESTIONS_JSON, "w", encoding="utf-8") as f:
            json.dump({"questions": questions}, f, indent=2, ensure_ascii=False)
        log(f"Saved generated questions to: {GENERATED_QUESTIONS_JSON}")

    # load checkpoint if exists
    existing = load_checkpoint()
    qa_pairs = existing[:]  # list of {q:..., a:...}
    answered = len(qa_pairs)
    log(f"Resuming from checkpoint. Already answered: {answered}")

    # iterate questions
    pbar = tqdm(total=len(questions), desc="questions")
    pbar.update(answered)
    for i in range(answered, len(questions)):
        q = questions[i]
        try:
            a = fetch_answer(llm, q)
            qa_pairs.append({"q": q, "a": a})
            # checkpoint every SAVE_CHECKPOINT_EVERY
            if (len(qa_pairs) % SAVE_CHECKPOINT_EVERY) == 0:
                save_checkpoint(qa_pairs, questions)
                # also create incremental PDF so far (overwrite)
                build_pdf_incremental(qa_pairs, filename=PDF_FILENAME)
            time.sleep(REQUEST_DELAY_SECONDS)
        except Exception as e:
            log(f"ERROR: failed to handle question index {i} -> {e}")
            # save current work and continue
            save_checkpoint(qa_pairs, questions)
        pbar.update(1)

    pbar.close()
    # final persist & build PDF
    save_checkpoint(qa_pairs, questions)
    build_pdf_incremental(qa_pairs, filename=PDF_FILENAME)
    log("All done!")

if __name__ == "__main__":
    main()
