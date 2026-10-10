import streamlit as st
import json
import os
from dotenv import load_dotenv
from groq import Groq
from faster_whisper import WhisperModel
from fpdf import FPDF
from datetime import datetime

# 1. Load API Keys
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# 2. UI Styling
st.set_page_config(page_title="Agentic AI Medical Scribe", layout="wide", page_icon="🤖")
st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display: none; }
    .stApp { background-color: #ffffff; color: #333333; }
    h1 { color: #005f73; font-family: 'Helvetica Neue', sans-serif; font-weight: bold; }
    .stButton>button { background-color: #0a9396 !important; color: white !important; border-radius: 5px; font-weight: bold; }
    .agent-step { background-color: #f0f4f8; padding: 10px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #0a9396; }
    </style>
""", unsafe_allow_html=True)

st.title(" Agentic AI Medical Scribe")
st.markdown("An autonomous agent that transcribes, reasons, looks up history, and generates clinical orders dynamically.")
st.divider()

# 3. Audio Input
st.subheader("🎙️ Patient Consultation Input")
demo_files = {
    "Case 1: General (Fever & Nausea)": "test_cases/case1_general_fever.mp3",
    "Case 2: Cardiology (Chest Pain)": "test_cases/case2_cardiology_chest_pain.mp3",
}
selected_demo = st.selectbox("Select a pre-recorded scenario:", ["None"] + list(demo_files.keys()))
demo_audio_file = None
if selected_demo != "None" and os.path.exists(demo_files[selected_demo]):
    with open(demo_files[selected_demo], "rb") as f:
        demo_audio_file = f.read()
    st.success(f"Loaded: {selected_demo}")

st.divider()
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"], key="upload")
with col2:
    recorded_file = st.audio_input("Record a conversation", key="record")

if demo_audio_file is not None: audio_file = demo_audio_file
elif uploaded_file is not None: 
    audio_file = uploaded_file
    st.success("File uploaded!")
elif recorded_file is not None: 
    audio_file = recorded_file
    st.success("Recording captured!")
else: 
    audio_file = None

# ==========================================
# AGENTIC TOOL DEFINITIONS & EXECUTION
# ==========================================
def lookup_patient_history(patient_id: str, query_focus: str) -> dict:
    """Simulates querying an EHR database for patient history."""
    mock_db = {
        "PT-8842": {
            "allergies": ["Penicillin", "Peanuts"],
            "chronic_conditions": ["Mild Asthma"],
            "current_medications": ["Albuterol Inhaler (PRN)"]
        },
        "PT-9901": {
            "allergies": ["None known"],
            "chronic_conditions": ["Hypertension", "Type 2 Diabetes"],
            "current_medications": ["Lisinopril 10mg", "Metformin 500mg"]
        }
    }
    patient = mock_db.get(patient_id, mock_db["PT-8842"])
    return {
        "allergies": patient["allergies"],
        "chronic_conditions": patient["chronic_conditions"],
        "current_medications": patient["current_medications"]
    }

def extract_medical_codes(symptoms: list, procedures: list) -> dict:
    """Simulates mapping clinical terms to official ICD-10 and CPT billing codes."""
    return {
        "icd10_codes": ["J20.9: Acute bronchitis, unspecified", "R50.9: Fever, unspecified"],
        "cpt_codes": ["99213: Office visit, established patient, low complexity", "94640: Inhalation treatment"]
    }

def generate_orders(diagnosis: str, age: int, allergies: list, reasoning: str) -> dict:
    """Simulates generating safe prescriptions and lab orders, checking allergies."""
    if "Penicillin" in allergies and "Amoxicillin" in reasoning:
        return {"error": "CONTRAINDICATION: Patient is allergic to Penicillin. Order rejected."}
    
    return {
        "prescriptions": [
            {"medication": "Azithromycin (Z-Pak)", "dosage": "500mg day 1, then 250mg days 2-5", "frequency": "Once daily"},
            {"medication": "Albuterol HFA", "dosage": "90mcg", "frequency": "2 puffs every 4-6 hours PRN"}
        ],
        "lab_orders": ["CBC with differential", "Chest X-Ray (PA and Lateral)"]
    }

def generate_after_care(diagnosis: str, plan: str, red_flags: list) -> dict:
    """Simulates translating clinical jargon into a 5th-grade patient summary."""
    return {
        "summary": f"You were diagnosed with {diagnosis.lower()}. {plan} Please rest and hydrate. GO TO THE ER IMMEDIATELY IF YOU EXPERIENCE: {', '.join(red_flags).lower()}."
    }

# Groq Tool Schema Definitions
agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_patient_history",
            "description": "Look up patient's known allergies, chronic conditions, and current medications from the EHR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "The patient's ID, e.g., 'PT-8842'"},
                    "query_focus": {"type": "string", "description": "What to look up, e.g., 'allergies and chronic_conditions'"}
                },
                "required": ["patient_id", "query_focus"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_medical_codes",
            "description": "Map symptoms and procedures to official ICD-10 and CPT billing codes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symptoms": {"type": "array", "items": {"type": "string"}, "description": "List of patient symptoms"},
                    "procedures": {"type": "array", "items": {"type": "string"}, "description": "List of procedures performed"}
                },
                "required": ["symptoms", "procedures"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_orders",
            "description": "Generate prescriptions and lab orders based on diagnosis, age, and allergies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string", "description": "Primary diagnosis"},
                    "age": {"type": "integer", "description": "Patient age"},
                    "allergies": {"type": "array", "items": {"type": "string"}, "description": "Known allergies"},
                    "reasoning": {"type": "string", "description": "Clinical justification"}
                },
                "required": ["diagnosis", "age", "allergies", "reasoning"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_after_care",
            "description": "Translate the clinical plan into a simple, empathetic, 5th-grade reading level summary for the patient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string", "description": "Medical diagnosis"},
                    "plan": {"type": "string", "description": "Treatment plan"},
                    "red_flags": {"type": "array", "items": {"type": "string"}, "description": "Symptoms requiring immediate ER visit"}
                },
                "required": ["diagnosis", "plan", "red_flags"]
            }
        }
    }
]

# ==========================================
# PROCESSING LOGIC
# ==========================================
if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file if isinstance(audio_file, bytes) else audio_file.getbuffer())
    
    st.audio("temp_audio.wav", format="audio/wav")

    if st.button("🚀 Run Autonomous Agent", type="primary"):
        # Initialize session state variables first
        st.session_state['agent_output'] = ""
        st.session_state['tool_results'] = []
        st.session_state['transcript'] = ""
        
        client = Groq(api_key=groq_key)
        
        with st.status(" Agent Initializing...", expanded=True) as status:
            try:
                # STEP 1: Transcribe
                st.write("🎙️ Transcribing audio locally via Faster-Whisper...")
                model = WhisperModel("base", device="cpu", compute_type="int8")
                segments, _ = model.transcribe("temp_audio.wav", language="en", beam_size=5)
                transcript_text = " ".join([seg.text.strip() for seg in segments])
                st.session_state['transcript'] = transcript_text

                 # --- NEW: AUDIO QUALITY CHECK ---
                if len(transcript_text.strip()) < 10:
                    st.error("️ Audio was too short, silent, or unclear. Please record a longer consultation or upload a clearer file.")
                    st.stop() # This completely stops the app from running the rest of the code
                # --------------------------------
                
                # STEP 2: Agentic Tool Calling Loop
                st.write("🧠 Agent analyzing transcript and deciding which tools to call...")
                messages = [
                    {"role": "system", "content": "You are an expert Autonomous Medical Scribe Agent. Analyze the transcript. Use the provided tools to look up history, extract codes, generate orders, and create after-care instructions. Do NOT hallucinate data."},
                    {"role": "user", "content": f"Patient ID is PT-8842. Age is 34. Transcript: {transcript_text}"}
                ]
                
                # First LLM Call: Decide tools
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    tools=agent_tools,
                    tool_choice="auto",
                    temperature=0.1
                )
                
                assistant_message = response.choices[0].message
                messages.append(assistant_message)
                tool_results_display = []

                # Execute Tools
                if assistant_message.tool_calls:
                    for tool_call in assistant_message.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        
                        st.write(f"⚙️ Executing tool: **{func_name}**...")
                        
                        # Route to local Python function
                        if func_name == "lookup_patient_history":
                            result = lookup_patient_history(**func_args)
                        elif func_name == "extract_medical_codes":
                            result = extract_medical_codes(func_args.get("symptoms", []), func_args.get("procedures", []))
                        elif func_name == "generate_orders":
                            result = generate_orders(func_args.get("diagnosis", ""), func_args.get("age", 34), func_args.get("allergies", []), func_args.get("reasoning", ""))
                        elif func_name == "generate_after_care":
                            result = generate_after_care(func_args.get("diagnosis", ""), func_args.get("plan", ""), func_args.get("red_flags", []))
                        else:
                            result = {"error": "Unknown tool"}
                        
                        tool_results_display.append({"tool": func_name, "result": result})
                        
                        # Feed result back to LLM
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        })
                    
                    # STEP 3: Final Synthesis
                    st.write("📝 Synthesizing final unified clinical payload...")
                    
                    # CRITICAL FIX: Add a final prompt to force the model to stop calling tools
                    messages.append({
                        "role": "user", 
                        "content": "Based on the tool results above, please generate the final unified clinical report. Include the SOAP note, billing codes, orders, and after-care instructions. Do NOT call any more tools, just output the final text."
                    })
                    
                    # Note: We intentionally do NOT pass 'tools' here to force text generation
                    final_response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=messages,
                        temperature=0.1
                    )
                    final_output = final_response.choices[0].message.content
                    
                    # Save to session state BEFORE updating status
                    st.session_state['agent_output'] = final_output
                    st.session_state['tool_results'] = tool_results_display
                    
                    status.update(label="✅ Agent Workflow Complete!", state="complete", expanded=False)
                    
                else:
                    st.warning("⚠️ Agent did not call any tools. Check the transcript quality.")
                    status.update(label="️ Agent Workflow Incomplete", state="error", expanded=False)
                    
            except Exception as e:
                st.error(f"❌ Agent Error: {str(e)}")
                status.update(label="❌ Agent Workflow Failed", state="error", expanded=False)

# ==========================================
# DISPLAY SECTION
# ==========================================
if 'agent_output' in st.session_state and st.session_state['agent_output']:
    st.divider()
    
    st.subheader("🧠 Agent Thought Process & Tool Execution")
    for step in st.session_state.get('tool_results', []):
        st.markdown(f"<div class='agent-step'><strong>🛠️ Tool:</strong> `{step['tool']}`<br><strong> Output:</strong> <pre>{json.dumps(step['result'], indent=2)}</pre></div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("📄 Final Unified Clinical Output")
    st.markdown(st.session_state['agent_output'])
    
    st.divider()
    st.subheader("💾 Export")
    st.download_button(
        label="📥 Download Agent Report as JSON",
        data=json.dumps({
            "transcript": st.session_state.get('transcript', ''),
            "agent_output": st.session_state['agent_output'],
            "tools_used": st.session_state.get('tool_results', [])
        }, indent=2),
        file_name=f"Agentic_Scribe_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        type="primary"
    )

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888888; font-size: 12px; padding: 20px;'><p>© 2024 Agentic AI Medical Scribe. Built for secure, autonomous clinical workflows.</p></div>", unsafe_allow_html=True)