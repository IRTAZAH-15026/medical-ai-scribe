# 🤖 Agentic AI Medical Scribe

[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-All_Rights_Reserved-red.svg)](LICENSE)

An autonomous, privacy-first AI medical agent that transcribes patient consultations, dynamically queries electronic health records (EHR), extracts billing codes, and generates safe, allergy-checked clinical orders—all while keeping sensitive audio data 100% on the user's device.

## 🏗️ Agentic System Architecture

Unlike traditional linear pipelines, this application uses a **ReAct (Reason + Act) Agentic Loop**. The LLM dynamically decides which tools to call based on the clinical context, ensuring high accuracy and preventing hallucinations.

```text
[Patient Audio] 
      │
      ▼
[Local Device] ───(Faster-Whisper)───> [Raw Transcript]
      │                                       │
      ▼                                       ▼
[Agentic LLM Core] <──(Tool Calls)──> [Mock EHR / Code DB]
      │   ├─ lookup_patient_history (Checks allergies & history)
      │   ├─ extract_medical_codes (Maps to ICD-10/CPT)
      │   ├─ generate_orders (Enforces allergy guardrails)
      │   └─ generate_after_care (Translates to 5th-grade level)
      ▼
[Unified JSON Payload: SOAP Note, Billing Codes, Lab Orders, Patient Summary]
      │
      ▼
[HL7 FHIR Export / PDF Report]