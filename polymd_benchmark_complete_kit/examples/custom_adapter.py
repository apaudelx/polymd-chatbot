"""
Edit this adapter to call your actual PolyMD chatbot directly from Python.

The benchmark harness calls:
    answer_question(question: str, history: list[dict]) -> dict | str

Return a dict when possible:
{
  "answer": "...",
  "retrieved_chunk_ids": ["..."],
  "retrieved_dois": ["https://doi.org/..."],
  "abstained": false
}
"""

def answer_question(question, history):
    """Placeholder adapter for connecting another chatbot implementation."""
    # TODO: Replace this with your app's real call.
    #
    # Example idea if your app has a service function:
    #
    # from src.app_services import build_rag_system, process_query
    # system = build_rag_system()
    # result = process_query(system, question, chat_history=history)
    # return {
    #     "answer": result.answer,
    #     "retrieved_chunk_ids": [e.chunk_id for e in result.evidence],
    #     "retrieved_dois": [e.doi for e in result.evidence],
    #     "abstained": result.abstained,
    # }
    return {
        "answer": (
            "Adapter not implemented. Replace examples/custom_adapter.py with "
            "your chatbot call."
        ),
        "retrieved_chunk_ids": [],
        "retrieved_dois": [],
        "abstained": True,
    }
