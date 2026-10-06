import json
import re

# Load the transcript
with open("transcript.json", "r") as f:
    segments = json.load(f)

# Combine all text
full_text = " ".join([seg["text"] for seg in segments])
print("=== FULL TEXT ===")
print(full_text)
print()

# 1. Extract Symptoms
symptoms_pattern = r"\b(headache|migraine|fever|nausea|vomiting|pain|cough|fatigue|dizziness)\b"
symptoms = re.findall(symptoms_pattern, full_text, re.IGNORECASE)
# Remove duplicates and keep original case
symptoms = list(set([s.capitalize() for s in symptoms]))

# 2. Extract Duration (Handles both "3 days" and "three days")
# This looks for digits OR number words followed by time units
duration_pattern = r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|hour|hours|month|months)\b"
durations = re.findall(duration_pattern, full_text, re.IGNORECASE)
# Format them nicely (e.g., "Three days")
durations = [f"{d[0].capitalize()} {d[1]}" for d in durations]

# 3. Extract Measurements (Looks for numbers near keywords)
# This is a bit looser: it looks for a number within 5 words of "fever" or "temperature"
measurements = []
# Find all numbers in the text
numbers = re.finditer(r"\b(\d+\.?\d*)\b", full_text)
for num_match in numbers:
    # Get the text around the number (10 characters before and after)
    start = max(0, num_match.start() - 15)
    end = min(len(full_text), num_match.end() + 15)
    context = full_text[start:end].lower()
    
    # If "fever" or "temperature" is nearby, it's a measurement
    if "fever" in context or "temperature" in context:
        measurements.append(f"Fever: {num_match.group()}")

# Remove duplicates
measurements = list(set(measurements))

# Package everything
entities = {
    "symptoms": symptoms,
    "duration": durations,
    "measurements": measurements
}

# Save entities
with open("entities.json", "w") as f:
    json.dump(entities, f, indent=2)

print("=== EXTRACTED MEDICAL ENTITIES ===")
print(json.dumps(entities, indent=2))
print()
print("✅ Saved to entities.json")