"""CSS helpers for the Streamlit interface."""

from __future__ import annotations


def get_global_css() -> str:
    """Return the application-wide CSS injected into Streamlit."""
    return """
<style>
    .stApp {
        background-color: white !important;
    }

    .main .block-container {
        background-color: white !important;
        color: black !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: black !important;
    }

    .stButton>button {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
    }

    .stButton>button:hover {
        background-color: #f5f5f5 !important;
        color: black !important;
        border: 1px solid #ccc !important;
    }

    .stChatInputContainer {
        background-color: white !important;
    }

    input[type="text"] {
        background-color: white !important;
        color: black !important;
        border: 1px solid #ccc !important;
    }

    input[type="text"]::placeholder {
        text-align: center !important;
        color: #999 !important;
        font-size: 1rem !important;
    }

    [data-testid="stChatMessage"] {
        width: 100% !important;
        display: flex !important;
        align-items: flex-start !important;
        gap: 0.5rem !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        justify-content: flex-start !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        gap: 0.5rem !important;
        align-items: center !important;
    }

    .stChatMessage {
        padding: 0 !important;
        margin-bottom: 1rem !important;
        background: transparent !important;
        border: none !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: none !important;
        display: inline-block !important;
        width: fit-content !important;
        min-width: 0 !important;
        max-width: 70% !important;
        padding: 0 !important;
        margin: 0 !important;
        text-align: right !important;
        overflow-wrap: anywhere !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        max-width: 85% !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        margin: 0.25rem 0 !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ol,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul {
        margin: 0.25rem 0 !important;
        padding-left: 1.15rem !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        margin: 0.15rem 0 !important;
    }

    [data-testid="stChatMessage"] a {
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }

    [data-testid="chatAvatarIcon-user"],
    [data-testid="chatAvatarIcon-assistant"] {
        flex-shrink: 0 !important;
        background-color: #e9ecef !important;
        color: #495057 !important;
    }

    .streamlit-expanderHeader {
        background-color: #f8f9fa !important;
        color: black !important;
        border: 1px solid #dee2e6 !important;
    }

    .stCaption {
        color: #666 !important;
    }

    hr {
        border-color: #dee2e6 !important;
    }

    .landing-container {
        text-align: center;
        padding: 2rem 1rem;
        max-width: 800px;
        margin: 0 auto;
    }

    .landing-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #1a1a1a !important;
    }

    .landing-subtitle {
        font-size: 1.2rem;
        color: #666 !important;
        margin-bottom: 2rem;
        line-height: 1.6;
    }

    .feature-section {
        margin: 2rem 0;
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }

    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2c3e50 !important;
    }

    .stTextInput small,
    .stTextInput [data-testid="InputInstructions"] {
        display: none !important;
    }

    .stTextInput input[type="text"] {
        font-size: 1.1rem !important;
        padding: 1.2rem 2rem !important;
        height: auto !important;
        min-height: 60px !important;
        border-radius: 50px !important;
        border: 2px solid #d1d5db !important;
        text-align: center !important;
        outline: none !important;
    }

    .stTextInput input[type="text"]:focus {
        border-color: #9ca3af !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        outline: none !important;
    }

    div[data-testid="column"] .stButton>button {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #1a1a1a !important;
        border: 2px solid #d1d5db !important;
        padding: 1rem 3rem !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
        width: auto !important;
        min-width: 250px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="column"] .stButton>button:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        border: 2px solid #9ca3af !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12) !important;
    }

    .rdkit-thumb-wrap {
        text-align: center;
        margin-top: 0.5rem;
    }

    .rdkit-thumb {
        max-width: 100%;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        cursor: pointer;
    }

    .rdkit-modal {
        display: none;
        position: fixed;
        inset: 0;
        z-index: 2000;
    }

    .rdkit-modal:target {
        display: block;
    }

    .rdkit-modal-overlay {
        position: absolute;
        inset: 0;
        background: rgba(0, 0, 0, 0.65);
    }

    .rdkit-modal-content {
        position: relative;
        z-index: 2001;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1.25rem;
        box-sizing: border-box;
    }

    .rdkit-modal-card {
        position: relative;
        display: inline-flex;
        justify-content: center;
        align-items: center;
    }

    .rdkit-modal-image {
        max-width: min(78vw, 920px);
        max-height: 78vh;
        border-radius: 10px;
        background: #fff;
        padding: 8px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    .rdkit-modal-close {
        position: absolute;
        top: -12px;
        right: -12px;
        z-index: 2002;
        font-size: 30px;
        line-height: 1;
        color: #111;
        text-decoration: none;
        background: #fff;
        border: 1px solid #d1d5db;
        border-radius: 999px;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
</style>
"""
