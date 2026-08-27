from dotenv import load_dotenv
import os
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
from langchain_core.prompts import ChatPromptTemplate

import json

with open("agris_filtered.json", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total records: {len(data)}")
print(f"Records with null subject: {sum(1 for r in data if r['subject'] is None)}")
print(f"Records with empty/short abstract (<50 chars): {sum(1 for r in data if len(r['abstract']) < 50)}")

# Rough language check — flag likely non-English entries for awareness
non_ascii_heavy = sum(1 for r in data if sum(not c.isascii() for c in r['title']) > len(r['title']) * 0.3)
print(f"Records with heavy non-ASCII in title (possible non-English): {non_ascii_heavy}")


if __name__ == "__main__":
    if(os.environ.get("GOOGLE_API_KEY")):
        print("Key found")
    else:
        print("key not found")
        