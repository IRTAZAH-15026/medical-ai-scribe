import json
import os
from groq import Groq
from dotenv import load_dotenv

# 1. Load the API key from the .env file
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Load the Knowledge Graph we built in Phase 1
with open("medical_graph.json", "r") as f:
    graph = json.load(f)

# 3. Convert the Graph into a clean, readable text format for the LLM
# This is the "Retrieval" part of Graph-RAG!
graph_text = "PATIENT CONTEXT:\n"
for node in graph["nodes"]:
    graph_text += f"- {node['type']}: {node['label']} (ID: {node['id']})\n"

graph_text += "\nRELATIONSHIPS:\n"
for edge in graph["edges"]:
    graph_text += f"- {edge['from']} is connected to {edge['to']} via {edge['relationship']}\n"

print("=== FEEDING THIS GRAPH TO THE AI ===")
print(graph_text)
print("="*50)

# 4. Define the 3 Unique Prompts
prompts = {
    "SOAP Note": """You are an expert medical scribe. Based ONLY on the provided Patient Context and Relationships, generate a professional SOAP (Subjective, Objective, Assessment, Plan) note. Use strict medical terminology. Do not hallucinate any symptoms or durations not explicitly stated in the relationships.""",
    
    "Prior Authorization": """You are a medical billing specialist. Based ONLY on the provided Patient Context and Relationships, draft a formal 'Letter of Medical Necessity' to an insurance company. Clearly state the patient's symptoms, duration, and vital measurements to justify the need for an evaluation and treatment. Be persuasive and professional.""",
    
    "Patient Summary": """You are a compassionate healthcare provider. Based ONLY on the provided Patient Context and Relationships, write a simple, easy-to-understand summary for the patient. Use a 5th-grade reading level. Be empathetic, avoid medical jargon, and clearly list what they should do next."""
}

# 5. Generate the Reports
reports = {}

for report_type, prompt in prompts.items():
    print(f"\n🤖 Generating {report_type}... (This takes ~1 second with Groq)")
    
    # Combine the system prompt with our graph data
    full_prompt = f"{prompt}\n\nHere is the strict data you must use:\n{graph_text}"
    
    # Call the Groq API (using Llama-3-8b, which is incredibly fast and free)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.1 # Low temperature = less creativity, more strict adherence to facts
    )
    
    # Save the result
    reports[report_type] = response.choices[0].message.content

# 6. Save all reports to a single file
with open("final_reports.txt", "w", encoding="utf-8") as f:
    for report_type, content in reports.items():
        f.write(f"\n{'='*60}\n")
        f.write(f"📄 {report_type.upper()}\n")
        f.write(f"{'='*60}\n")
        f.write(content)
        f.write("\n")

print("\n" + "="*60)
print("✅ SUCCESS! All 3 reports have been generated.")
print("📂 Check the 'final_reports.txt' file in your folder!")
print("="*60)