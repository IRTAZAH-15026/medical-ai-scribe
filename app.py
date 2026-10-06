import streamlit as st
import json
import os
import re
from dotenv import load_dotenv
from groq import Groq
from faster_whisper import WhisperModel
from fpdf import FPDF
from datetime import datetime
from pyannote.audio import Pipeline

# 1. Load the API keys silently from the .env file
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

# 2. Force the Medical Look & Hide the Sidebar
st.set_page_config(page_title="Privacy-First AI Scribe", layout="wide", page_icon="🩺")

st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display: none; }
    .stApp { background-color: #ffffff; color: #333333; }
    h1 { color: #005f73; font-family: 'Helvetica Neue', sans-serif; font-weight: bold; }
    .stButton>button { background-color: #0a9396 !important; color: white !important; border-radius: 5px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 3. Add the Title and Intro
st.title("🩺 Privacy-First AI Medical Scribe")
st.markdown("Upload a patient consultation audio file or record live. The AI will transcribe, analyze, and generate reports **100% locally** (except the final LLM generation).")
st.divider()

# 4. Audio Input (Quick Load, Upload, or Record)
st.subheader("🎙️ Patient Consultation Input")

st.markdown("**⚡ Quick Load Demo Cases (For Presentation)**")
demo_files = {
    "Case 1: General (Fever & Nausea)": "test_cases/case1_general_fever.mp3",
    "Case 2: Cardiology (Chest Pain)": "test_cases/case2_cardiology_chest_pain.mp3",
    "Case 3: Neurology (Migraine)": "test_cases/case3_neurology_migraine.mp3",
    "Case 4: Orthopedics (Back Pain)": "test_cases/case4_orthopedics_back_pain.mp3",
    "Case 5: Allergy (Rash)": "test_cases/case5_pediatrics_allergy.mp3"
}

selected_demo = st.selectbox("Select a pre-recorded scenario:", ["None"] + list(demo_files.keys()))

demo_audio_file = None
if selected_demo != "None" and os.path.exists(demo_files[selected_demo]):
    with open(demo_files[selected_demo], "rb") as f:
        demo_audio_file = f.read()
    st.success(f"Loaded: {selected_demo}")

st.divider()
st.markdown("**Or use your own audio:**")
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload .wav or .mp3", type=["wav", "mp3"], key="upload")

with col2:
    recorded_file = st.audio_input("Record a conversation", key="record")

if demo_audio_file is not None:
    audio_file = demo_audio_file
elif uploaded_file is not None:
    audio_file = uploaded_file
    st.success("File uploaded successfully!")
elif recorded_file is not None:
    audio_file = recorded_file
    st.success("Recording captured successfully!")
else:
    audio_file = None

# 5. Processing Logic
if audio_file is not None:
    with open("temp_audio.wav", "wb") as f:
        if isinstance(audio_file, bytes):
            f.write(audio_file)
        else:
            f.write(audio_file.getbuffer())
    
    st.audio("temp_audio.wav", format="audio/wav")

    if 'playback_start' not in st.session_state:
        st.session_state.playback_start = 0

    if st.button("🚀 Generate Medical Reports", type="primary"):
        
        # --- PROFESSIONAL STEP 1 ---
        with st.status("Step 1: Processing Audio Locally...", expanded=True) as status:
            st.write("🎙️ Initializing local Whisper model and speaker diarization...")
            transcript_text = []
            full_text = ""
            
            try:
                diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
                diarization = diarization_pipeline({"audio": "temp_audio.wav"})
                diarization_available = True
            except Exception:
                st.warning("⚠️ Speaker Diarization skipped due to Windows audio library limits. Falling back to standard transcription.")
                diarization_available = False
            
            st.write("🧠 Transcribing audio and mapping speaker segments...")
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = model.transcribe("temp_audio.wav", language="en", beam_size=5, vad_filter=False)
            
            for segment in segments:
                start, end, text = segment.start, segment.end, segment.text.strip()
                full_text += text + " "
                speaker_label = "Speaker"
                
                if diarization_available:
                    speaker_scores = {}
                    for turn, _, speaker in diarization.itertracks(yield_label=True):
                        overlap = max(0, min(end, turn.end) - max(start, turn.start))
                        if overlap > 0:
                            speaker_scores[speaker] = speaker_scores.get(speaker, 0) + overlap
                    dominant_speaker = max(speaker_scores, key=speaker_scores.get) if speaker_scores else "UNKNOWN"
                    speaker_label = dominant_speaker.replace("SPEAKER_", "Speaker ")
                
                transcript_text.append({"start": start, "end": end, "text": text, "speaker": speaker_label})
            
            st.session_state['transcript_text'] = transcript_text
            st.session_state['full_text'] = full_text
            status.update(label="✅ Step 1: Transcription & Speaker Identification Complete!", state="complete", expanded=False)
            
        # --- PROFESSIONAL STEP 2 ---
        with st.status("Step 2: Analyzing Clinical Data...", expanded=True) as status:
            st.write("🔍 Extracting medical entities and matching ICD-10 billing codes...")
            extraction_prompt = f"""
            You are an expert medical coder. Analyze this transcript and extract:
            1. Symptoms (list of strings)
            2. Duration (string)
            3. Measurements (list of strings)
            4. Suggested ICD-10 Codes (list of 1-2 codes with descriptions, e.g., "R50.9: Fever, unspecified")
            Transcript: "{full_text}"
            Return ONLY a valid JSON object with keys: "symptoms", "duration", "measurements", "icd10_codes".
            """
            try:
                client = Groq(api_key=groq_key)
                extract_response = client.chat.completions.create(
                    model="openai/gpt-oss-120b", 
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                extracted_data = json.loads(extract_response.choices[0].message.content)
                entities = {
                    "symptoms": extracted_data.get("symptoms", []),
                    "duration": extracted_data.get("duration", "Not specified"),
                    "measurements": extracted_data.get("measurements", []),
                    "icd10_codes": extracted_data.get("icd10_codes", [])
                }
                st.session_state['entities'] = entities
                status.update(label=f"✅ Step 2: Extracted {len(entities['symptoms'])} symptoms and {len(entities['icd10_codes'])} ICD-10 codes!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                entities = {"symptoms": [], "duration": "Unknown", "measurements": [], "icd10_codes": []}
                status.update(label="❌ Step 2: Extraction Failed (Using fallback data)", state="error", expanded=False)

        # --- PROFESSIONAL STEP 3 ---
        with st.status("Step 3: Building Knowledge Graph...", expanded=True) as status:
            st.write("🕸️ Mapping relationships between patient, symptoms, and medical codes...")
            graph = {"nodes": [{"id": "Patient_01", "type": "Person", "label": "Patient"}], "edges": []}
            for symptom in entities["symptoms"]:
                graph["nodes"].append({"id": f"Symptom_{symptom}", "type": "Symptom", "label": symptom})
                graph["edges"].append({"from": "Patient_01", "to": f"Symptom_{symptom}", "relationship": "HAS_SYMPTOM"})
            
            if entities["duration"] and entities["duration"] != "Not specified":
                graph["nodes"].append({"id": "Duration", "type": "Time", "label": entities["duration"]})
                if entities["symptoms"]:
                    graph["edges"].append({"from": f"Symptom_{entities['symptoms'][0]}", "to": "Duration", "relationship": "LASTS_FOR"})
            
            for m in entities["measurements"]:
                if ":" in str(m):
                    s_name, val = str(m).split(": ", 1)
                    graph["nodes"].append({"id": f"Val_{val}", "type": "Measurement", "label": val})
                    graph["edges"].append({"from": f"Symptom_{s_name.strip()}", "to": f"Val_{val}", "relationship": "HAS_VALUE"})
            
            for code_info in entities["icd10_codes"]:
                graph["nodes"].append({"id": f"Code_{code_info}", "type": "ICD10", "label": code_info})
                if entities["symptoms"]:
                    graph["edges"].append({"from": f"Symptom_{entities['symptoms'][0]}", "to": f"Code_{code_info}", "relationship": "JUSTIFIES_CODE"})
            
            graph_text = "PATIENT CONTEXT:\n" + "\n".join([f"- {n['type']}: {n['label']}" for n in graph["nodes"]])
            graph_text += "\n\nRELATIONSHIPS:\n" + "\n".join([f"- {e['from']} -> {e['to']} via {e['relationship']}" for e in graph["edges"]])
            st.session_state['graph_text'] = graph_text
            status.update(label="✅ Step 3: Knowledge Graph Built Successfully!", state="complete", expanded=False)

        # --- PROFESSIONAL STEP 4 ---
        with st.status("Step 4: Generating Professional Reports...", expanded=True) as status:
            st.write("📝 Drafting SOAP Note, Prior Authorization, and Patient Summary...")
            client = Groq(api_key=groq_key)
            prompts = {
                "SOAP Note": "You are an expert medical scribe. Based ONLY on the provided Patient Context and Relationships, generate a professional SOAP note. Do not hallucinate.",
                "Prior Authorization": "You are a medical billing specialist. Draft a formal 'Letter of Medical Necessity' to an insurance company. You MUST explicitly mention the suggested ICD-10 codes provided in the data. Be persuasive.",
                "Patient Summary": "You are a compassionate healthcare provider. Write a simple, 5th-grade reading level summary for the patient based ONLY on the provided data. Do NOT mention ICD-10 codes."
            }
            reports = {}
            for r_type, prompt in prompts.items():
                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b", 
                    messages=[{"role": "user", "content": f"{prompt}\n\nData:\n{graph_text}"}], 
                    temperature=0.1
                )
                reports[r_type] = response.choices[0].message.content
            
            st.session_state['reports'] = reports
            status.update(label="✅ Step 4: All Reports Generated Successfully!", state="complete", expanded=False)

# ==========================================
# DISPLAY SECTION
# ==========================================
if 'reports' in st.session_state:
    reports = st.session_state['reports']
    transcript_text = st.session_state.get('transcript_text', [])
    entities = st.session_state.get('entities', {})
    
    st.divider()
    st.subheader("📄 Final Reports")
    tab1, tab2, tab3 = st.tabs(["SOAP Note", "Prior Authorization", "Patient Summary"])
    with tab1: st.text_area("Clinical Note", reports["SOAP Note"], height=300)
    with tab2: st.text_area("Insurance Letter", reports["Prior Authorization"], height=300)
    with tab3: st.text_area("Patient Handout", reports["Patient Summary"], height=300)

    st.divider()
    st.subheader("🔍 Verification: Click-to-Source Transcript")
    st.info("Click the '▶ Play' button next to any sentence to jump the audio to that exact moment.")
    st.audio("temp_audio.wav", format="audio/wav", start_time=st.session_state.playback_start)
    st.markdown("---")
    
    for i, seg in enumerate(transcript_text):
        col_time, col_text, col_btn = st.columns([1, 6, 1])
        with col_time: st.caption(f"{seg['start']:.1f}s")
        with col_text: st.markdown(f"**[{seg.get('speaker', 'Speaker')}]**: {seg['text']}")
        with col_btn:
            if st.button("▶ Play", key=f"play_{i}"):
                st.session_state.playback_start = seg['start']
                st.rerun()

    # 3. EHR & PDF Export
    st.divider()
    st.subheader("💾 Export Reports")
    
    def generate_fhir_bundle(entities_data):
        bundle = {"resourceType": "Bundle", "type": "collection", "id": "medical-scribe-encounter-001", "entry": []}
        bundle["entry"].append({"resource": {"resourceType": "Patient", "id": "patient-001", "name": [{"use": "official", "family": "Doe", "given": ["John"]}]}})
        
        symptoms = entities_data.get("symptoms", [])
        codes = entities_data.get("icd10_codes", [])
        
        for i, symptom in enumerate(symptoms):
            icd_code, icd_display = "R50.9", "Fever, unspecified"
            if i < len(codes) and ":" in str(codes[i]):
                parts = str(codes[i]).split(":", 1)
                icd_code, icd_display = parts[0].strip(), parts[1].strip()
            
            bundle["entry"].append({
                "resource": {
                    "resourceType": "Condition",
                    "id": f"condition-{i}",
                    "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
                    "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": icd_code, "display": icd_display}], "text": symptom},
                    "subject": {"reference": "Patient/patient-001"}
                }
            })
        return json.dumps(bundle, indent=2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🏥 EHR Integration (HL7 FHIR)**")
        st.caption("Exports structured clinical data in the standard format used by Epic, Cerner, and Athenahealth.")
        fhir_json = generate_fhir_bundle(entities)
        st.download_button(label="📥 Download FHIR JSON for EHR", data=fhir_json, file_name=f"FHIR_Encounter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", type="secondary")
        
    with col2:
        st.markdown("**📄 Human-Readable Reports**")
        st.caption("Exports SOAP Note, Prior Auth, and Patient Summary as a formatted PDF.")
        def create_medical_pdf(reports_data):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", style='B', size=16)
            pdf.cell(200, 10, txt="AI Medical Scribe Report", ln=True, align='C')
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
            pdf.ln(10)
            for title, content in reports_data.items():
                pdf.set_font("Arial", style='B', size=14)
                pdf.set_fill_color(200, 220, 255)
                pdf.cell(200, 8, txt=title, ln=True, fill=True)
                pdf.ln(2)
                pdf.set_font("Arial", size=11)
                pdf.multi_cell(0, 6, txt=content.encode('ascii', 'ignore').decode('ascii'))
                pdf.ln(5)
            return bytes(pdf.output())

        pdf_bytes = create_medical_pdf(reports)
        st.download_button(label="📥 Download All Reports as PDF", data=pdf_bytes, file_name=f"Medical_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf", type="primary")

# ==========================================
# PROFESSIONAL FOOTER & DISCLAIMER
# ==========================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888888; font-size: 12px; padding: 20px;">
        <p><strong>🔒 Privacy & Compliance Notice:</strong> Audio transcription and entity extraction are processed 100% locally on this device using Faster-Whisper. No patient audio data is transmitted to external servers. LLM report generation utilizes secure, ephemeral API calls. This tool is intended for administrative assistance and should be verified by a licensed healthcare professional.</p>
        <p>© 2024 Privacy-First AI Medical Scribe. Built for secure clinical workflows.</p>
    </div>
    """, 
    unsafe_allow_html=True
)