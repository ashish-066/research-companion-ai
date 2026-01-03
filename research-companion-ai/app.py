from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import re

# ================= SETUP =================

app = Flask(__name__)
model = SentenceTransformer("all-MiniLM-L6-v2")
nltk.download("punkt")

PARAGRAPH_HISTORY = []

# ================= UTILS =================

def clean(text):
    return re.sub(r"<[^>]+>", "", text or "").strip()

def embed(text):
    return model.encode(text, normalize_embeddings=True)

def cosine(a, b):
    return float(np.dot(a, b))

# ================= CLAIM RULES =================

STRONG_CLAIMS = [
    "demonstrates", "proves", "causes", "leads to",
    "results in", "significantly", "increases", "reduces"
]

def is_strong_claim(s):
    s = s.lower()
    return any(v in s for v in STRONG_CLAIMS)

def has_evidence(s):
    return bool(re.search(r"\d+|\(|\)|\[\w+\]", s))

# ================= PARAGRAPH SCORING =================

def score_paragraph(paragraph, problem):
    para_emb = embed(paragraph)
    prob_emb = embed(problem)

    # Novelty
    novelty = 1.0
    if PARAGRAPH_HISTORY:
        sims = [cosine(para_emb, embed(p)) for p in PARAGRAPH_HISTORY]
        novelty = 1 - max(sims)

    # Alignment
    alignment = cosine(para_emb, prob_emb)

    # Coherence
    sentences = sent_tokenize(paragraph)
    sent_embs = [embed(s) for s in sentences]
    coherence = 1.0
    if len(sent_embs) > 1:
        sims = [cosine(sent_embs[i], sent_embs[i+1]) for i in range(len(sent_embs)-1)]
        coherence = sum(sims) / len(sims)

    score = (
        0.35 * novelty +
        0.35 * alignment +
        0.30 * coherence
    )

    return round(max(0, min(1, score)) * 100, 1)

# ================= SENTENCE ANALYSIS =================

def analyze_sentences(paragraph, problem):
    sentences = sent_tokenize(paragraph)
    prob_emb = embed(problem)
    results = []

    for s in sentences:
        emb = embed(s)
        alignment = cosine(emb, prob_emb)
        issues = []

        if alignment < 0.25:
            issues.append({
                "reason": "Sentence weakly relates to the research problem.",
                "suggestion": "Explicitly connect this sentence to the stated problem."
            })

        if is_strong_claim(s) and not has_evidence(s):
            issues.append({
                "reason": "Strong claim without supporting evidence.",
                "suggestion": "Add a statistic, citation, or reference."
            })

        results.append({
            "sentence": s,
            "issues": issues
        })

    return results

# ================= API =================

@app.route("/score", methods=["POST"])
def score():
    data = request.json or {}
    paragraph = clean(data.get("paragraph"))
    problem = clean(data.get("problem", "research problem"))

    if len(paragraph) < 20:
        return jsonify({"score": 0, "sentences": []})

    score_value = score_paragraph(paragraph, problem)
    sentence_feedback = analyze_sentences(paragraph, problem)

    PARAGRAPH_HISTORY.append(paragraph)

    return jsonify({
        "score": score_value,
        "sentences": sentence_feedback
    })

# ================= WEBPAGE =================

@app.route("/editor")
def editor():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Research Companion AI</title>
<style>
body {
  font-family: Arial;
  max-width: 900px;
  margin: 40px auto;
}
.editor {
  border: 1px solid #ccc;
  padding: 15px;
  min-height: 200px;
  font-size: 16px;
}
.issue {
  text-decoration: underline red;
  cursor: pointer;
}
.tooltip {
  position: absolute;
  background: #222;
  color: #fff;
  padding: 10px;
  border-radius: 6px;
  font-size: 13px;
  max-width: 320px;
  display: none;
  z-index: 9999;
  white-space: pre-line;
}
.score {
  margin-top: 10px;
  font-weight: bold;
}
</style>
</head>
<body>

<h2>Research Companion AI</h2>

<p><b>Research Problem</b></p>
<input id="problem" style="width:100%;padding:8px"
 value="Impact of bee population decline on food security"/>

<p><b>Write your paragraph</b></p>
<div id="editor" class="editor" contenteditable="true">
The global decline in bee populations poses a significant threat to food security.
</div>

<div id="score" class="score"></div>
<div id="tooltip" class="tooltip"></div>

<script>
let timer = null;
const editor = document.getElementById("editor");
const tooltip = document.getElementById("tooltip");

editor.addEventListener("input", () => {
  clearTimeout(timer);
  timer = setTimeout(analyze, 1000);
});

async function analyze() {
  const res = await fetch("/score", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      paragraph: editor.innerText,
      problem: document.getElementById("problem").value
    })
  });

  const data = await res.json();
  document.getElementById("score").innerText =
    "Research Score: " + data.score + "/100";

  highlight(data.sentences);
}

function highlight(sentences) {
  let text = editor.innerText;
  let html = text;

  sentences.forEach(s => {
    if (!s.issues.length) return;

    const escaped = s.sentence.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&");
    const msg = s.issues.map(
      i => "❌ " + i.reason + "\\n💡 " + i.suggestion
    ).join("\\n\\n");

    html = html.replace(
      new RegExp(escaped, "g"),
      `<span class="issue" data-tip="${msg.replace(/"/g,'&quot;')}">${s.sentence}</span>`
    );
  });

  editor.innerHTML = html;
  attachTooltips();
}

function attachTooltips() {
  document.querySelectorAll(".issue").forEach(el => {
    el.addEventListener("mouseenter", e => {
      tooltip.innerText = el.dataset.tip;
      tooltip.style.display = "block";
    });
    el.addEventListener("mousemove", e => {
      tooltip.style.left = e.pageX + 15 + "px";
      tooltip.style.top = e.pageY + 15 + "px";
    });
    el.addEventListener("mouseleave", () => {
      tooltip.style.display = "none";
    });
  });
}
</script>

</body>
</html>
"""

# ================= RUN =================

if __name__ == "__main__":
    app.run(debug=True)
nano requirements.txt
