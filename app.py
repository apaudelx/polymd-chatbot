"""Streamlit entrypoint for PolyMD."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from src.app_services import build_system, process_query
from src.config import Config
from src.ui.rendering import render_assistant_content, show_landing_page
from src.ui.styles import get_global_css


st.set_page_config(
    page_title="PolyMD",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(get_global_css(), unsafe_allow_html=True)


@st.cache_resource
def initialize_system(cache_buster: str = "") -> dict[str, Any]:
    """Initialize and cache the RAG system components for Streamlit."""
    try:
        return build_system(cache_buster=cache_buster)
    except Exception as exc:
        st.error(f"Error initializing system: {exc}")
        st.stop()


def initialize_session_state() -> None:
    """Create the session state keys required by the chat workflow."""
    defaults = {
        "messages": [],
        "show_chat": not Config.SHOW_LANDING_PAGE,
        "pending_query": None,
        "pending_assistant_index": None,
        "stored_query": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def enqueue_query(prompt: str) -> None:
    """Add a pending user query and assistant placeholder to session state."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append(
        {"role": "assistant", "content": "_Searching and generating response..._"}
    )
    st.session_state.pending_query = prompt
    st.session_state.pending_assistant_index = len(st.session_state.messages) - 1
    st.rerun()


def build_chat_export() -> str:
    """Serialize the current chat history as pretty JSON."""
    payload = {
        "app": "PolyMD",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "message_count": len(st.session_state.messages),
        "messages": st.session_state.messages,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def render_sidebar() -> None:
    """Render chat controls and quick RDKit examples in the sidebar."""
    with st.sidebar:
        st.title("Controls")
        st.markdown("---")

        st.download_button(
            "Download Chat JSON",
            data=build_chat_export(),
            file_name="ppqs_chat_history.json",
            mime="application/json",
            use_container_width=True,
            disabled=len(st.session_state.messages) == 0,
        )

        if st.button("New Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.session_state.pending_assistant_index = None
            st.rerun()

        if Config.SHOW_LANDING_PAGE and st.button(
            "Back to Home",
            use_container_width=True,
        ):
            st.session_state.show_chat = False
            st.rerun()

        if st.button("Clear Cache", use_container_width=True):
            st.cache_resource.clear()
            st.success("Cache cleared. Please refresh the page.")
            st.stop()

        st.markdown("---")
        st.subheader("RDKit Quick Tools")
        st.caption("These run local RDKit tools via explicit command routing.")

        quick_tools = {
            "Canonicalize CCO": "rdkit: canonicalize CCO",
            "Descriptors for CCO": "rdkit: descriptors CCO",
            "Similarity CCO vs CCN": "rdkit: similarity CCO CCN",
            "Render CCO image": "rdkit: render CCO",
        }
        for label, query in quick_tools.items():
            if st.button(label, use_container_width=True):
                st.session_state.stored_query = query
                st.rerun()

        st.markdown("---")
        st.info(
            "This app queries a database of polymer properties, research papers, "
            "and force fields."
        )


def render_chat_history(has_pending: bool) -> Any:
    """Render stored chat messages and return the placeholder for pending output."""
    pending_slot = None
    pending_index = st.session_state.pending_assistant_index

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            is_pending_assistant = (
                has_pending
                and index == pending_index
                and message["role"] == "assistant"
            )
            if is_pending_assistant:
                pending_slot = st.empty()
                pending_slot.markdown(message["content"])
            elif message["role"] == "assistant":
                render_assistant_content(message["content"])
            else:
                st.markdown(message["content"])

    return pending_slot


def render_chat_shell() -> None:
    """Render the chat title or compact caption above the conversation."""
    if len(st.session_state.messages) == 0:
        st.markdown(
            "<h1 style='text-align: center;'>PolyMD</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align: center; color: #666;'>"
            "Query polymer properties, research papers, and force fields using "
            "natural language</p>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(
            "Query polymer properties, research papers, and force fields using "
            "natural language"
        )


def handle_pending_query(system: dict[str, Any], pending_slot: Any) -> None:
    """Process a pending query and replace the assistant placeholder."""
    query_text = st.session_state.pending_query
    pending_index = st.session_state.pending_assistant_index

    try:
        prior_history = st.session_state.messages[: max(pending_index - 1, 0)]
        response, _stats = process_query(
            query_text,
            system,
            conversation_history=prior_history,
        )
    except Exception as exc:
        response = {
            "answer": f"Error: {exc}",
            "evidence_snippets": [],
            "doi_links": [],
            "confidence": 0.0,
            "abstained": True,
            "abstention_reason": "Runtime error while processing query.",
        }

    if isinstance(pending_index, int) and 0 <= pending_index < len(
        st.session_state.messages
    ):
        st.session_state.messages[pending_index]["content"] = response
        if pending_slot is not None:
            preview = (
                response.get("answer", "")
                if isinstance(response, dict)
                else str(response)
            )
            pending_slot.markdown(preview)
    else:
        st.session_state.messages.append({"role": "assistant", "content": response})

    st.session_state.pending_query = None
    st.session_state.pending_assistant_index = None
    st.rerun()


def render_chat(system: dict[str, Any]) -> None:
    """Render the full chat UI and process submitted prompts."""
    render_sidebar()
    render_chat_shell()

    has_pending = bool(st.session_state.pending_query)
    pending_slot = render_chat_history(has_pending)

    try:
        prompt = st.chat_input(
            "Ask about polymer properties, papers, or force fields",
            disabled=has_pending,
        )
    except TypeError:
        prompt = st.chat_input("Ask about polymer properties, papers, or force fields")

    if prompt and not has_pending:
        enqueue_query(prompt)

    if has_pending:
        handle_pending_query(system, pending_slot)


def main() -> None:
    """Run the Streamlit application."""
    initialize_session_state()

    if not st.session_state.show_chat:
        show_landing_page()
        return

    system = initialize_system(Config.CACHE_BUSTER)

    if st.session_state.stored_query and not st.session_state.pending_query:
        prompt = st.session_state.stored_query
        st.session_state.stored_query = None
        enqueue_query(prompt)

    render_chat(system)


if __name__ == "__main__":
    main()
