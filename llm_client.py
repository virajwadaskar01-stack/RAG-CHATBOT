"""
llm_client.py
--------------
Wraps calls to Groq's API (free, fast inference for open-source LLMs like Llama).
Modes:
1. general_chat()   -> normal chatbot answer, no document context
2. rag_chat()        -> answer strictly from retrieved document chunks (legacy, kept for reference)
3. hybrid_chat()     -> shows the model retrieved chunks + lets IT judge relevance, instead of
                        gating on a similarity score before the model ever sees the content.
                        This is what app.py actually calls whenever a document is loaded.
4. generate_outline() -> structured title+sections outline for PDF/DOCX/PPTX generation
"""

import json
import re
from groq import Groq

MODEL_NAME = "llama-3.1-8b-instant"  # fast + free-tier friendly on Groq


class LLMClient:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def general_chat(self, user_query: str, chat_history: list = None) -> str:
        """
        Plain chatbot mode — like talking to a normal LLM.
        chat_history is a list of {"role": "user"/"assistant", "content": "..."} dicts.
        """
        messages = [
            {"role": "system", "content": (
                "You are a helpful, friendly assistant built by Viraj Wadaskar as part of an "
                "AI/ML portfolio project. If asked who made you, who created you, or what model "
                "you are, always say you were built by Viraj Wadaskar using the Groq API and "
                "Llama 3.1. Do not mention Meta, OpenAI, or any other company as your creator. "
                "Answer clearly and concisely. If the question asks about two or more distinct "
                "topics (e.g. 'tell me about X and Y'), address each one in its own separate "
                "paragraph, in the order they were asked, rather than blending them into one "
                "paragraph. "
                "For math, formulas, or step-by-step problems: explain each step in plain "
                "language before or after the formula, not just the formula alone - name the "
                "identity or rule being used and why it applies, so someone learning the topic "
                "for the first time can follow the reasoning, not just copy the result. Write "
                "exponents using ^ notation (e.g. sin^2(x), x^2, e^(-x)) rather than spelling "
                "them out, since the app renders ^ as a proper superscript automatically. "
                "Important: photos are shown automatically below your answer by the app itself "
                "when relevant (via a separate image search you do not control). Never write "
                "image descriptions, alt text, or bracketed placeholders like '[Image: ...]' in "
                "your answer, and never describe what an image would show — just answer the "
                "question in plain text and say nothing about images unless the user explicitly "
                "asks about the photos shown."
            )}
        ]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_query})

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def rag_chat(self, user_query: str, retrieved_chunks: list) -> str:
        """
        RAG mode — answer strictly using the retrieved document chunks.
        retrieved_chunks: list of (chunk_text, source, score) tuples.
        """
        context_blocks = []
        for i, (chunk, source, score) in enumerate(retrieved_chunks, start=1):
            context_blocks.append(f"[Source {i}: {source}]\n{chunk}")
        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a helpful assistant built by Viraj Wadaskar that answers questions using "
            "ONLY the provided context. If asked who made you, say you were built by Viraj "
            "Wadaskar. If the answer is not in the context, say you don't have that information "
            "in the uploaded documents, rather than guessing. Always be concise and clear. "
            "When useful, mention which source number your answer came from."
        )

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}"

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def hybrid_chat(self, user_query: str, retrieved_chunks: list, chat_history: list = None) -> str:
        """
        Document-aware mode — always used when a document is loaded, regardless of the
        similarity score of the retrieved chunks. Replaces the old "gate on a cosine
        similarity threshold before deciding RAG vs General Chat" approach: that approach
        let genuinely relevant questions (e.g. "how much fees is paid" against a fee
        receipt) fall through to general_chat() whenever the embedding score happened to
        land under the threshold - and general_chat() would then invent a plausible-looking
        but entirely fabricated answer, since it has no idea a real document exists.

        Instead, the retrieved chunks are always shown to the model, and the model itself
        judges whether they answer the question - with an explicit instruction never to
        invent, estimate, or extrapolate numbers/details beyond what's actually in the text.

        retrieved_chunks: list of (chunk_text, source, score) tuples, possibly empty.
        """
        if retrieved_chunks:
            context_blocks = []
            for i, (chunk, source, score) in enumerate(retrieved_chunks, start=1):
                context_blocks.append(f"[Source {i}: {source}]\n{chunk}")
            context_text = "\n\n".join(context_blocks)
        else:
            context_text = "(no document content was retrieved)"

        system_prompt = (
            "You are a helpful assistant built by Viraj Wadaskar. If asked who made you, who "
            "created you, or what model you are, always say you were built by Viraj Wadaskar "
            "using the Groq API and Llama 3.1. Do not mention Meta, OpenAI, or any other "
            "company as your creator.\n\n"
            "The user has an uploaded document. Here is the most relevant content retrieved "
            "from it for this specific question:\n\n"
            f"{context_text}\n\n"
            "Rules for using this content:\n"
            "1. Read it carefully. If it actually contains the answer, base your answer "
            "ONLY on the exact facts, figures, and wording it contains. Never invent, "
            "estimate, round, convert currencies, or extrapolate categories, tiers, or "
            "numbers that are not explicitly written in the retrieved content above.\n"
            "2. If the retrieved content does NOT contain the answer, say plainly that the "
            "document doesn't cover that, and only then you may answer from general "
            "knowledge instead - clearly separate from any document-based facts, never "
            "blended together as if both came from the document.\n"
            "3. Never write image descriptions or bracketed placeholders like "
            "'[Image: ...]' - photos are handled separately by the app.\n"
            "4. If the question covers two or more distinct topics, address each one in its "
            "own separate paragraph, in the order they were asked.\n"
            "5. For math, formulas, or step-by-step problems, explain each step in plain "
            "language, not just the formula alone - name the rule or identity being used and "
            "why it applies. Write exponents using ^ notation (e.g. x^2) rather than spelling "
            "them out, since the app renders ^ as a proper superscript automatically.\n"
            "Be concise and clear."
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            messages.extend(chat_history)
        messages.append({"role": "user", "content": user_query})

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _clean_json_text(self, raw: str) -> str:
        """Strips markdown fences and other common wrapping artifacts."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        return raw.strip()

    def _try_repair_json(self, raw: str) -> str:
        """
        Attempts to fix the most common ways LLM-generated JSON breaks:
        - smart/curly quotes instead of straight quotes
        - trailing commas before } or ]
        - raw newlines inside string values (must be escaped as \\n)
        This is a best-effort repair, not a full JSON parser.
        """
        repaired = raw

        # Normalize smart quotes to straight quotes
        repaired = (repaired
                    .replace("\u201c", '"').replace("\u201d", '"')
                    .replace("\u2018", "'").replace("\u2019", "'"))

        # Remove trailing commas before a closing bracket/brace
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        # Escape raw newlines that fall inside string values
        # (only inside quotes - do a conservative pass: any lone \n between
        # a quote-opened string and its close often causes this exact error)
        repaired = repaired.replace("\r\n", "\\n").replace("\n", "\\n")
        # Restore the newlines that are actually meant to separate JSON structure
        # (safe because valid structural newlines were outside strings in the
        # original, well-formed parts of the response)
        repaired = repaired.replace("{\\n", "{\n").replace("}\\n", "}\n") \
                            .replace("[\\n", "[\n").replace("]\\n", "]\n") \
                            .replace(",\\n", ",\n")

        return repaired

    def generate_outline(self, source_type: str, source_content: str) -> dict:
        """
        Generates a structured outline (title + sections) suitable for turning
        into a PDF, DOCX, or PPTX.

        source_type: "topic" or "conversation"
        source_content: either the topic text, or the formatted conversation text

        Raises ValueError with a friendly message if the outline truly can't
        be parsed even after repair attempts, so the UI can show something
        useful instead of a raw JSON traceback.
        """
        if source_type == "topic":
            instruction = (
                f"Create a well-organized document outline about this topic: {source_content}\n"
                "Break it into 4-6 clear sections that build understanding progressively."
            )
        else:  # conversation
            instruction = (
                "Summarize the following conversation into a well-organized document outline, "
                "grouping related questions/answers into clear sections:\n\n"
                f"{source_content}"
            )

        system_prompt = (
            "You generate structured document outlines. Respond with ONLY valid JSON, "
            "no markdown fences, no preamble, no explanation, no trailing commas. "
            "Every string value must be a single line with no literal line breaks inside it "
            "— use plain spaces instead of new lines within a section's content. "
            "Format exactly as:\n"
            '{"title": "...", "sections": [{"heading": "...", "content": "..."}]}\n'
            "Each section's content should be 2-4 sentences, written in plain clear prose, "
            "on a single line with no line breaks."
        )

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
            temperature=0.3,
        )

        raw = response.choices[0].message.content
        cleaned = self._clean_json_text(raw)

        # First attempt: parse as-is
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Second attempt: apply common repairs and retry
        try:
            repaired = self._try_repair_json(cleaned)
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # Third attempt: ask the model to fix its own output
        try:
            fix_response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": (
                        "You fix broken JSON. Respond with ONLY the corrected, valid JSON - "
                        "no markdown fences, no explanation, no trailing commas, no literal "
                        "line breaks inside string values."
                    )},
                    {"role": "user", "content": f"Fix this JSON so it parses correctly:\n\n{cleaned}"},
                ],
                temperature=0.1,
            )
            fixed_raw = self._clean_json_text(fix_response.choices[0].message.content)
            return json.loads(fixed_raw)
        except (json.JSONDecodeError, Exception):
            pass

        # Final fallback: don't crash the whole feature - wrap the raw text as
        # a single-section outline so the user still gets *something* usable.
        fallback_title = source_content[:60] if source_type == "topic" else "Conversation Summary"
        plain_text = re.sub(r"[{}\[\]\"]", "", raw).strip()
        return {
            "title": fallback_title,
            "sections": [
                {"heading": "Overview", "content": plain_text[:1500] or "Could not generate content - please try again."}
            ],
        }
