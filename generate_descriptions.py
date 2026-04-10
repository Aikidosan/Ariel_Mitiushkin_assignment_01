import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
BASE_URL       = "https://api.tokenfactory.nebius.com/v1/"
MODEL          = "google/gemma-2-9b-it-fast"    # or meta-llama/Meta-Llama-3.1-8B-Instruct-fast
SYSTEM_PROMPT  = (
    "You are an assistant that writes clear, engaging product descriptions. "
    "Given the product name, key attributes, material, and warranty information, "
    "produce a persuasive description that is exactly 50‑90 words long. "
    "Focus on benefits, use vivid language, and highlight what makes the product valuable to the customer. "
    "Do not mention the word count or any instructions—just output the description."
)

client = OpenAI(base_url=BASE_URL, api_key=NEBIUS_API_KEY)

def generate_description(user_prompt: str) -> dict:
    """
    Calls the model, measures end‑to‑end latency, and returns:
    {
        "generated_description": str,
        "latency_ms": float,
        "input_tokens": int,
        "output_tokens": int
    }
    """
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=150,          # enough for a 90‑word description
    )

    latency_ms = (time.time() - start) * 1000

    # Extract the fields the assignment asks for
    generated_description = response.choices[0].message.content.strip()
    input_tokens          = response.usage.prompt_tokens      # tokens sent (system + user)
    output_tokens         = response.usage.completion_tokens  # tokens received from the model

    return {
        "generated_description": generated_description,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("Assignment_01_product_dataset.csv")

    records = []
    for _, row in df.iterrows():
        user_prompt = (
            f"Name: {row['product_name']}\n"
            f"Attributes: {row['Product_attribute_list']}\n"
            f"Material: {row['material']}\n"
            f"Warranty: {row['warranty']}"
        )
        record = generate_description(user_prompt)
        records.append(record)

    metrics_df = pd.DataFrame(records)
    out_df     = pd.concat([df, metrics_df], axis=1)

    for col in ["persuasiveness", "clarity", "word_count_adherence", "grounding"]:
        out_df[col] = ""
    out_df["final_score"] = ""

    out_df.to_excel("assignment_01.xlsx", index=False, engine="openpyxl")
    print(f"Saved {len(out_df)} rows to assignment_01.xlsx")
