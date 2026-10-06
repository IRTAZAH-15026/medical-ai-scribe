# 🩺 Privacy-First AI Medical Scribe

[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)

An enterprise-grade, privacy-first AI medical scribe that transcribes patient consultations, extracts clinical entities, suggests ICD-10 billing codes, and generates comprehensive medical reports—all while keeping sensitive audio data 100% on the user's device.

## 🏗️ System Architecture

```text
[Patient Audio] 
      │
      ▼
[Local Device] ───(Faster-Whisper)───> [Raw Transcript + Timestamps]
      │                                       │
      │                               (Pyannote Diarization)
      ▼                                       ▼
[Secure LLM API] <──(JSON Extraction)── [Symptoms, Duration, ICD-10 Codes]
      │                                       │
      ▼                                       ▼
[Knowledge Graph] ───────────────────> [Context-Aware Prompting]
      │
      ▼
[Outputs: SOAP Note, Prior Auth Letter, Patient Summary, HL7 FHIR JSON]