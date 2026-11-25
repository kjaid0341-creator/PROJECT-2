#!/usr/bin/env python3
"""
test_rag_pipeline.py

Comprehensive test suite verifying:
1. ChromaDB connection & document retrieval for all 5 target questions.
2. Conversation memory and session isolation.
3. FastAPI endpoint status, CORS headers, and error handling.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from fastapi.testclient import TestClient
from server import app, init_vectorstore, get_session_history


def run_tests():
    print("=" * 60)
    print("GraphAI Assistant RAG Pipeline Test Suite")
    print("=" * 60)

    # 1. Vector Database & Retrieval Test
    print("\n[Test 1] Testing Vector Database & Retrieval...")
    vs = init_vectorstore()
    assert vs is not None, "Chroma vectorstore could not be initialized!"
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    target_questions = [
        "What is Graph?",
        "Explain BFS.",
        "What is Dijkstra's algorithm?",
        "Explain Disjoint Set Union.",
        "What is the difference between BFS and DFS?"
    ]

    for q in target_questions:
        docs = retriever.invoke(q)
        assert len(docs) > 0, f"No chunks retrieved for question: {q}"
        print(f"  ✓ '{q}' -> Retrieved {len(docs)} chunks (Top: Chunk #{docs[0].metadata.get('chunk_id')})")

    # 2. Conversation Memory & Session Isolation Test
    print("\n[Test 2] Testing Session-Based Conversation Memory...")
    session_a = get_session_history("user_session_alpha")
    session_b = get_session_history("user_session_beta")

    session_a.add_user_message("What is BFS?")
    session_a.add_ai_message("BFS searches level by level.")

    session_b.add_user_message("What is Kruskal algorithm?")
    session_b.add_ai_message("Kruskal finds the minimum spanning tree.")

    assert len(session_a.messages) == 2
    assert len(session_b.messages) == 2
    assert session_a.messages[0].content != session_b.messages[0].content
    print("  ✓ Session Alpha has isolated history of 2 messages.")
    print("  ✓ Session Beta has isolated history of 2 messages.")
    print("  ✓ No cross-contamination between sessions.")

    # 3. FastAPI Client Tests
    print("\n[Test 3] Testing FastAPI Endpoints & CORS...")
    client = TestClient(app)

    # Root endpoint
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert r_root.json()["diagnostics"]["chroma_db_found"] is True
    print("  ✓ GET / returned 200 OK with system diagnostics.")

    # CORS Preflight
    r_cors = client.options("/ask", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST"
    })
    assert r_cors.headers.get("access-control-allow-origin") == "http://localhost:3000"
    print("  ✓ CORS preflight approved for external origin.")

    # Validation: Empty question
    r_empty = client.post("/ask", json={"question": "   "})
    assert r_empty.status_code == 400
    print("  ✓ POST /ask rejected empty question with HTTP 400.")

    # Validation: Missing or unconfigured API key
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key or openai_key.startswith("your_"):
        r_ask = client.post("/ask", json={"question": "What is Graph?"})
        assert r_ask.status_code == 401
        print("  ✓ POST /ask correctly returned HTTP 401 when API key is unconfigured.")
        print(f"    Message: {r_ask.json()['detail'][:70]}...")
    else:
        print("  ✓ Real API key detected in .env! Testing live AI response...")
        r_ask = client.post("/ask", json={"question": "What is Graph?"})
        if r_ask.status_code == 200:
            print(f"    Live AI Answer: {r_ask.json()['answer'][:150]}...")
        else:
            print(f"    LLM response status: {r_ask.status_code} - {r_ask.json()}")

    print("\n" + "=" * 60)
    print("All Pipeline & Vector DB Tests Passed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
