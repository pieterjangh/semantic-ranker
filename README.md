# Semantic Similarity

Compare one reference string with up to ten candidate strings using three OpenAI embedding models (`text-embedding-3-small`, `text-embedding-3-large`, and `text-embedding-ada-002`) and an LLM taxonomy reranker.

Change the reranker model with `LLM_MODEL` in `app.py`.

## Setup

1. Create a Python virtual environment.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies with `pip install -r requirements.txt`.

3. Start the app with `streamlit run app.py`.

4. Open the local Streamlit URL in the browser.

5. Paste an OpenAI API key into the app.

The API key is used only for the current request. It is not saved to disk.
