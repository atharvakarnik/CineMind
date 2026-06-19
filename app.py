"""Streamlit chat UI for CineMind."""

from __future__ import annotations

import streamlit as st

from src.agent import answer
from src.retriever import _get_index

st.set_page_config(page_title="CineMind — Movie QA", page_icon="🎬")


@st.cache_resource(show_spinner="Loading embedding model and vector index...")
def _warm_up() -> None:
    """Load the embedding model + Chroma index once per process; answer() reuses it after this."""
    _get_index()


_warm_up()

st.title("CineMind — Movie QA")
st.caption(
    "Ask about a movie's plot, cast, or reviews — answers are grounded in retrieved "
    "context, not the model's memory."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_sources(sources: list[dict[str, str]]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            st.markdown(f"- {source['title']} ({source['content_type']})")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_sources(message.get("sources"))

question = st.chat_input("Ask about a movie's plot, cast, or reviews...")
if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = answer(question)
            except Exception as exc:
                result = {"answer": f"Something went wrong answering that: {exc}", "sources": []}
        st.markdown(result["answer"])
        _render_sources(result["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
