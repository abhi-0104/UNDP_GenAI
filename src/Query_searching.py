from LLM_integration import qa_chain  # assuming same folder
import json

# === 1. Research Questions ===
research_questions = [
    "According to reports how many childrens worldwide are stunted and what is projected future value?"
]

# === 2. Ask and Collect Answers ===
responses = []

for i, question in enumerate(research_questions, start=1):
    print(f"\n🔍 Asking Question {i}:\n{question}\n")
    answer = qa_chain.run(question)
    print(f"📘 Answer {i}:\n{answer}\n")
    responses.append({
        "question": question,
        "answer": answer
    })

# === 3. Save to JSON (optional) ===
with open("llm_responses.json", "w", encoding="utf-8") as f:
    json.dump(responses, f, indent=2)

print("\n✅ All answers saved to llm_responses.json")
