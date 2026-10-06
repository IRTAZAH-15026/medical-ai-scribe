from faster_whisper import WhisperModel

print(" Loading Local AI Model (Privacy-First)...")
# "base" is fast and accurate. "small" is better for medical jargon but slower.
model = WhisperModel("base", device="cpu", compute_type="int8")

audio_file = "fixed_audio.wav"
print(f"🎧 Processing {audio_file} locally... (No data leaves your computer)")

# transcribe with VAD (Voice Activity Detection) to filter out silence
segments, info = model.transcribe(audio_file, language="en", beam_size=5, vad_filter=False)

print(f"Detected language: {info.language} (probability {info.language_probability:.2f})")
print("\n=== LOCAL TRANSCRIPT ===")

transcript_text = []

for segment in segments:
    # Format the timestamp
    start_time = f"[{segment.start:.1f}s -> {segment.end:.1f}s]"
    text = segment.text.strip()
    
    print(f"{start_time} {text}")
    transcript_text.append({
        "start": segment.start,
        "end": segment.end,
        "text": text
    })

# Save to JSON for your next steps
import json
with open("local_transcript.json", "w") as f:
    json.dump(transcript_text, f, indent=2)

print("\n✅ SUCCESS! Saved to local_transcript.json")