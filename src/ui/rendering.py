"""Streamlit rendering helpers for answers, tools, and the landing page."""

from __future__ import annotations

import base64
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable

import streamlit as st


def _image_to_base64(image_file: Path) -> str:
    """Read an image file and return a base64 string for inline rendering."""
    image_bytes = image_file.read_bytes()
    return base64.b64encode(image_bytes).decode("ascii")


def _render_rdkit_image(content: Dict[str, Any]) -> None:
    """Render an RDKit thumbnail with a full-size CSS modal."""
    answer = str(content.get("answer", "")).strip()
    image_path = str(content.get("image_path", "")).strip()
    image_width = int(content.get("image_width", 180))

    st.markdown(answer if answer else "RDKit render completed.")
    if not image_path:
        return

    image_file = Path(image_path)
    if not image_file.exists():
        st.caption(f"Rendered image path not found: {image_path}")
        return

    image_b64 = _image_to_base64(image_file)
    modal_id = f"rdkit-modal-{uuid.uuid4().hex}"

    # Streamlit does not provide a native image modal, so this small HTML block
    # uses an anchor target and CSS from `src.ui.styles`.
    st.markdown(
        f"""
        <div class="rdkit-thumb-wrap">
          <a href="#{modal_id}" aria-label="Open full-size image">
            <img class="rdkit-thumb" style="width: {image_width}px;"
                 src="data:image/png;base64,{image_b64}"
                 alt="RDKit render thumbnail" />
          </a>
        </div>

        <div id="{modal_id}" class="rdkit-modal">
          <a href="#" class="rdkit-modal-overlay" aria-label="Close image"></a>
          <div class="rdkit-modal-content">
            <div class="rdkit-modal-card">
              <a href="#" class="rdkit-modal-close" aria-label="Close image">&times;</a>
              <img class="rdkit-modal-image"
                   src="data:image/png;base64,{image_b64}"
                   alt="RDKit full-size render" />
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_markdown_list(title: str, items: Iterable[str], ordered: bool = False) -> None:
    """Render a titled expander containing a list of Markdown strings."""
    item_list = [str(item).strip() for item in items if str(item).strip()]
    if not item_list:
        return

    with st.expander(title):
        for index, item in enumerate(item_list, start=1):
            prefix = f"{index}." if ordered else "-"
            st.markdown(f"{prefix} {item}")


def _render_structured_answer(content: Dict[str, Any]) -> None:
    """Render the standard structured RAG response payload."""
    answer = str(content.get("answer", "")).strip()
    evidence = content.get("evidence_snippets", [])
    doi_links = content.get("doi_links", [])
    confidence = float(content.get("confidence", 0.0))
    abstained = bool(content.get("abstained", False))
    abstention_reason = str(content.get("abstention_reason", "")).strip()

    st.markdown(answer if answer else "No answer generated.")

    retrieval_notes = str(content.get("retrieval_notes", "")).strip()
    if retrieval_notes:
        st.caption("Comparison note")
        st.info(retrieval_notes)

    resolved_query = str(content.get("resolved_query", "")).strip()
    if resolved_query:
        st.caption(f"Resolved follow-up query: {resolved_query}")

    st.caption(f"Confidence: {confidence:.2f} | Abstained: {'Yes' if abstained else 'No'}")

    if abstained and abstention_reason:
        st.info(f"Abstention reason: {abstention_reason}")

    _render_markdown_list("Evidence Snippets", evidence, ordered=True)
    _render_markdown_list("Citations (DOI Links)", doi_links)


def render_assistant_content(content: Any) -> None:
    """Render assistant content from text, RAG JSON, or local tool payloads."""
    if not isinstance(content, dict):
        st.markdown(str(content))
        return

    if content.get("tool") == "rdkit_render":
        _render_rdkit_image(content)
        return

    _render_structured_answer(content)


def _render_database_contents() -> None:
    """Render the landing page database summary cards."""
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="font-size: 2rem; font-weight: 600; color: #2c3e50;">
                Database Contents
            </h2>
            <p style="color: #666; font-size: 1.05rem; margin-top: 0.5rem;">
                Comprehensive polymer data from scientific literature
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3, gap="medium")
    cards = [
        (
            col1,
            "Property Data",
            "Density, glass transition temperature, thermal conductivity, "
            "diffusion coefficients, viscosity, mechanical properties, and more",
        ),
        (
            col2,
            "Research Papers",
            "198+ curated scientific papers with full abstracts, citations, "
            "and DOI links",
        ),
        (
            col3,
            "Force Fields",
            "Computational chemistry parameters for polymer molecular dynamics "
            "simulations",
        ),
    ]

    for column, title, body in cards:
        with column:
            st.markdown(
                f"""
                <div class="feature-section" style="min-height: 300px;">
                    <h3 class="feature-title" style="text-align: center;">{title}</h3>
                    <p style="color: #495057; line-height: 1.8; margin-top: 1rem;
                              text-align: center;">
                        {body}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_landing_page() -> None:
    """Render the landing page and capture the first query if provided."""
    st.markdown(
        """
        <div class="landing-container">
            <h1 class="landing-title">PolyMD</h1>
            <p class="landing-subtitle">
                Query comprehensive polymer property data, research papers, and
                force field information using natural language. Get instant
                answers backed by scientific sources.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Search Query",
        placeholder="Ask about polymer properties, papers, or force fields",
        key="landing_query_input",
        label_visibility="collapsed",
    )

    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Searching", key="start_search", use_container_width=False):
            st.session_state.show_chat = True
            if query:
                st.session_state.stored_query = query
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    _render_database_contents()
