# ADR 0001: The runtime LLM provider is Gemini

**Status:** Accepted, 2026-10-08 (project owner's decision)

**Context.** The master spec (§4, §22) and the phase prompts were written for the Anthropic API. The project owner chose Gemini for the running system. Claude Code remains the development tool.

**Decision.**
- Agents call Gemini through `backend/app/llm/client.py` (google-genai, pinned), configured by `GEMINI_API_KEY` and `GEMINI_MODEL` in `.env`.
- Wherever the spec or a prompt says "Claude" for runtime behaviour, read it as "the configured LLM provider".

**Consequences.**
- The spec's rules apply unchanged to Gemini: reasoning and routing only, never numeric work (rule 5); deterministic fallback when unavailable (§22); prompt-injection isolation (§37).
- LLM answers are always validated locally with Pydantic, so changing provider again only touches `llm/client.py`.
- As of 2026-10-08 no real Gemini call has been made, because no key is configured. The LLM path is covered by a fake client that uses the real validation code, plus the opt-in `backend/tests/test_live_gemini.py`.
