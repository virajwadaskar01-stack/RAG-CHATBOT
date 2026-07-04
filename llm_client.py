"""
llm_client.py
--------------
Wraps calls to Groq's API (free, fast inference for open-source LLMs like Llama).
Two modes:
1. general_chat()  -> normal chatbot answer, no document context
2. rag_chat()       -> answer grounded in retrieved document chunks, with sources
"""

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
                "Answer clearly and concisely."
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
            temperature=0.3,  # lower temperature = more grounded, less "creative"
        )
        return response.choices[0].message.content

    def generate_outline(self, source_type: str, source_content: str) -> dict:
        """
        Generates a structured outline (title + sections) suitable for turning
        into a PDF, DOCX, or PPTX.

        source_type: "topic" or "conversation"
        source_content: either the topic text, or the formatted conversation text
        """
        import json

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
            "no markdown fences, no preamble, no explanation. Format exactly as:\n"
            '{"title": "...", "sections": [{"heading": "...", "content": "..."}]}\n'
            "Each section's content should be 2-4 sentences, written in plain clear prose."
        )

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
            ],
            temperature=0.4,
        )

        raw = response.choices[0].message.content.strip()
        # Defensive cleanup in case the model wraps JSON in markdown fences anyway
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return json.loads(raw)
