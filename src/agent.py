"""Runs the RAG pipeline: retrieve context, then call Gemini with a grounding prompt."""

from llama_index.llms.google_genai import GoogleGenAI

from config import GOOGLE_API_KEY

llm = GoogleGenAI(model="gemini-3.5-flash", api_key=GOOGLE_API_KEY)

# TODO: implement retrieval call, grounding prompt construction, and Gemini API call.
