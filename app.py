import json
import re
from typing import Literal

import numpy as np
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel

MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
# Change this to try another language model.
LLM_MODEL = "gpt-4.1-mini"
CANDIDATE_COUNT = 10
SCORE_LABELS = {
    4: "Excellent match",
    3: "Strong match",
    2: "Possible match",
    1: "Weak match",
    0: "No match",
}
FIT_LABELS = {
    4: "Excellent",
    3: "Strong",
    2: "Possible",
    1: "Weak",
    0: "No match",
}
_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9*_\-]{4,}")
_RERANK_INSTRUCTIONS = """You are a taxonomy classification reranker.

Your job is to determine which candidate taxonomy category best represents the meaning of an input text.

Judge category applicability, not lexical overlap.

Consider the underlying intent, concept, subject, and semantic scope of the input.

Evaluate each candidate independently using this rubric. Scores are absolute category fit, not relative rank. Do not spread scores across the range, and do not treat the best candidate as an excellent match unless it actually is one.

4 — Excellent match: the candidate directly and specifically represents the meaning of the input. It would be a natural taxonomy category for this input.
3 — Strong match: the candidate is clearly relevant and would be a reasonable classification, but there is some difference in specificity, scope, or framing.
2 — Possible match: there is meaningful semantic overlap, but the candidate is not an especially natural classification or another candidate fits materially better.
1 — Weak match: the candidate is only tangentially related. It would normally not be chosen if more appropriate categories are available.
0 — No match: the candidate does not meaningfully represent the input.

Then choose exactly one supplied candidate as the best match. If all candidates are weak, score them accordingly while still selecting the best available candidate.

Copy each candidate string unchanged. Return every supplied candidate exactly once.
Order results from the best taxonomy classification to the worst. When scores are equal, put the more appropriate classification first.
best_match must be the first result.
Give each reason in one short sentence.
Set label to the rubric name for that score."""


def cosine_similarity(a, b):
    vector_a = np.asarray(a, dtype=np.float64)
    vector_b = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)


def normalize_similarity(cosine):
    score = (cosine + 1.0) / 2.0
    return float(min(1.0, max(0.0, score)))


def non_empty_texts(texts):
    return [text.strip() for text in texts if text and text.strip()]


def get_embeddings(client, model, texts):
    response = client.embeddings.create(model=model, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    embeddings = [item.embedding for item in ordered]
    if len(embeddings) != len(texts):
        raise RuntimeError("The OpenAI API returned an unexpected number of embeddings.")
    return embeddings


class CandidateAssessment(BaseModel):
    candidate: str
    score: Literal[0, 1, 2, 3, 4]
    label: Literal[
        "Excellent match",
        "Strong match",
        "Possible match",
        "Weak match",
        "No match",
    ]
    reason: str


class TaxonomyRerank(BaseModel):
    best_match: str
    results: list[CandidateAssessment]


def rank_candidates(reference_embedding, candidates, candidate_embeddings):
    ranked = []
    for text, embedding in zip(candidates, candidate_embeddings):
        score = normalize_similarity(cosine_similarity(reference_embedding, embedding))
        ranked.append((text, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def rerank_with_llm(reference_text, candidates, api_key):
    """Score one input against taxonomy candidates. Pass the retrieved shortlist."""
    if len(candidates) != len(set(candidates)):
        raise ValueError("Each candidate text must be unique for LLM reranking.")

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.parse(
        model=LLM_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _RERANK_INSTRUCTIONS},
            {
                "role": "user",
                "content": json.dumps(
                    {"input": reference_text, "candidates": candidates},
                    ensure_ascii=False,
                ),
            },
        ],
        response_format=TaxonomyRerank,
    )
    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(message.refusal)
    parsed = message.parsed
    if parsed is None:
        raise RuntimeError("The language model did not return a classification.")
    return _normalize_rerank(parsed, candidates)


def _normalize_rerank(parsed, candidates):
    supplied = list(candidates)
    results = []
    seen = []
    for item in parsed.results:
        candidate = item.candidate
        if candidate not in supplied or candidate in seen:
            raise ValueError("The language model returned a candidate that was not supplied.")
        score = item.score
        if isinstance(score, bool) or not isinstance(score, int) or score not in SCORE_LABELS:
            raise ValueError("The language model returned a score outside 0–4.")
        seen.append(candidate)
        results.append(
            {
                "candidate": candidate,
                "score": score,
                "label": SCORE_LABELS[score],
                "reason": item.reason.strip(),
            }
        )
    if len(seen) != len(supplied):
        raise ValueError("The language model did not score every candidate exactly once.")

    best_match = parsed.best_match
    if best_match not in supplied:
        raise ValueError("The language model did not choose one of the supplied candidates.")
    best = next(item for item in results if item["candidate"] == best_match)
    if best["score"] != max(item["score"] for item in results):
        raise ValueError("The best match does not have the highest rubric score.")
    others = [item for item in results if item["candidate"] != best_match]
    others.sort(key=lambda item: item["score"], reverse=True)
    return {"best_match": best_match, "results": [best, *others]}


def safe_error_message(exc, api_key):
    message = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"].strip()
        elif isinstance(body.get("message"), str):
            message = body["message"].strip()
    if not message:
        message = str(exc).strip() or "The OpenAI API request failed."
    if api_key:
        message = message.replace(api_key, "[redacted]")
    message = _SECRET_PATTERN.sub("[redacted]", message)
    if len(message) > 500:
        message = message[:500].rstrip() + "..."
    return message


def main():
    st.set_page_config(page_title="Semantic Similarity")
    st.title("Semantic Similarity")

    api_key = st.text_input("OpenAI API Key", type="password")
    reference = st.text_area("Reference Text", height=180)

    candidates = []
    for index in range(1, CANDIDATE_COUNT + 1):
        candidates.append(st.text_area(f"Candidate {index}", height=100))

    if st.button("Run", type="primary"):
        key = api_key.strip()
        reference_text = reference.strip()
        candidate_texts = non_empty_texts(candidates)

        errors = []
        if not key:
            errors.append("Enter an OpenAI API key.")
        if not reference_text:
            errors.append("Enter reference text.")
        if not candidate_texts:
            errors.append("Enter at least one candidate.")
        for message in errors:
            st.error(message)
        if errors:
            return

        client = OpenAI(api_key=key)
        texts = [reference_text, *candidate_texts]
        for model in MODELS:
            st.subheader(model)
            try:
                embeddings = get_embeddings(client, model, texts)
                ranked = rank_candidates(embeddings[0], candidate_texts, embeddings[1:])
            except Exception as exc:
                st.error(safe_error_message(exc, key))
                continue

            st.dataframe(
                {
                    "Rank": list(range(1, len(ranked) + 1)),
                    "Candidate": [text for text, _score in ranked],
                    "Similarity Score": [f"{score:.4f}" for _text, score in ranked],
                },
                hide_index=True,
                width="stretch",
            )

        st.subheader("LLM Reranker")
        st.caption(LLM_MODEL)
        try:
            reranked = rerank_with_llm(reference_text, candidate_texts, key)
        except Exception as exc:
            st.error(safe_error_message(exc, key))
            return

        st.dataframe(
            {
                "Rank": list(range(1, len(reranked["results"]) + 1)),
                "Candidate": [item["candidate"] for item in reranked["results"]],
                "Fit": [
                    f"BEST MATCH — {FIT_LABELS[item['score']]}"
                    if index == 0
                    else FIT_LABELS[item["score"]]
                    for index, item in enumerate(reranked["results"])
                ],
                "Score": [f"{item['score']}/4" for item in reranked["results"]],
                "Reason": [item["reason"] for item in reranked["results"]],
            },
            hide_index=True,
            width="stretch",
        )


if __name__ == "__main__":
    main()
