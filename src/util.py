from google import genai
import spacy
import time
import json
import re

nlp = spacy.load("en_core_web_lg")
client = genai.Client(api_key="ADD_YOUR_KEY")
generation_config={"response_mime_type": "application/json"}


def call_llm(prompt, max_retries=3, delay=60, model="gemini-2.0-flash", temperature=0.1):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=generation_config)
            return response
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay * (attempt + 1))
    return None


def llm_answer_with_context(context, query):
    prompt = (
        f"Context: {context}\n"
        f"Query: {query}\n\n"
        "Instructions:\n"
        "- Answer the query using **only the answer (no extra words)**.\n"
        "- If the answer is not present in the context, return an empty string.\n"
        "- Follow the output format below strictly.\n\n"
        "Output format:\n"
        "{\"answer\": \"<your answer here>\"}"
    )

    response = call_llm(prompt)
    
    if response and hasattr(response, 'text'):
        try:
            result = json.loads(response.text.strip())
            return result.get("answer", "")
        except json.JSONDecodeError:
            print("Failed to parse LLM response as JSON:", response.text.strip())
            return ""
    
    print("LLM call failed or returned no text. Returning empty answer.")
    return ""
    
def clean(text):
    if not text:
        return ""

    text = text.encode("ascii", errors="ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
    ]

    return " ".join(tokens)

