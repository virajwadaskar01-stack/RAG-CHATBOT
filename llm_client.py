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
