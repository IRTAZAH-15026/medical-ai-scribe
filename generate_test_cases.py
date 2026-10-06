import asyncio
import edge_tts
import os

# Create a folder for our test cases
os.makedirs("test_cases", exist_ok=True)

# Define the 5 Medical Scenarios
scenarios = [
    {
        "filename": "case1_general_fever.mp3",
        "title": "Case 1: General Practice (Fever & Nausea)",
        "conversation": [
            ("Doctor", "Good morning. What brings you in to see me today?"),
            ("Patient", "Hi doctor. I've been feeling terrible for the last three days. I have a really bad fever, it's been around 102, and I'm constantly nauseous."),
            ("Doctor", "I'm sorry to hear that. Have you been able to keep any fluids down? Any vomiting?"),
            ("Patient", "No vomiting, just severe nausea. And I have a slight headache too.")
        ]
    },
    {
        "filename": "case2_cardiology_chest_pain.mp3",
        "title": "Case 2: Cardiology (Chest Pain)",
        "conversation": [
            ("Doctor", "Hello. I see you're here for some chest discomfort. Can you describe the pain?"),
            ("Patient", "Yes. It started yesterday. It's a sharp pain in the center of my chest, and sometimes it radiates down my left arm."),
            ("Doctor", "Does it happen when you exert yourself, or is it constant?"),
            ("Patient", "It happens when I walk up the stairs. I also get very short of breath and feel dizzy.")
        ]
    },
    {
        "filename": "case3_neurology_migraine.mp3",
        "title": "Case 3: Neurology (Severe Migraine)",
        "conversation": [
            ("Doctor", "Tell me about the headaches you've been experiencing."),
            ("Patient", "It's not just a headache. It's a severe migraine on the right side of my head. I've had it for about 5 hours now."),
            ("Doctor", "Are you experiencing any visual changes or sensitivity to light?"),
            ("Patient", "Yes, the light is bothering me a lot today. And I saw some flashing lights before the pain started. My blood pressure was 140 over 90 this morning.")
        ]
    },
    {
        "filename": "case4_orthopedics_back_pain.mp3",
        "title": "Case 4: Orthopedics (Lower Back Pain)",
        "conversation": [
            ("Doctor", "What seems to be the problem with your back?"),
            ("Patient", "I was lifting heavy boxes at work two days ago, and I felt a sharp pull in my lower back. The pain is about a 7 out of 10."),
            ("Doctor", "Does the pain stay in your back, or does it travel anywhere?"),
            ("Patient", "It shoots down my right leg, and my foot feels a bit numb sometimes.")
        ]
    },
    {
        "filename": "case5_pediatrics_allergy.mp3",
        "title": "Case 5: Allergy (Skin Rash)",
        "conversation": [
            ("Doctor", "Hi there. What's going on with your skin today?"),
            ("Patient", "I woke up this morning with this really itchy red rash all over my arms and chest. It started about 12 hours ago."),
            ("Doctor", "Did you eat anything unusual or try any new soaps recently?"),
            ("Patient", "I ate some peanuts at a party last night. I've never had a reaction before, but this is really swollen and itchy.")
        ]
    }
]

# Define Voices (Doctor = Male, Patient = Female)
DOCTOR_VOICE = "en-US-GuyNeural"
PATIENT_VOICE = "en-US-JennyNeural"

async def generate_audio(scenario):
    print(f"🎙️ Generating: {scenario['title']}...")
    
    # We will generate short clips for each line and combine them using pydub
    from pydub import AudioSegment
    
    full_audio = AudioSegment.empty()
    
    for speaker, text in scenario['conversation']:
        voice = DOCTOR_VOICE if speaker == "Doctor" else PATIENT_VOICE
        temp_file = f"temp_{speaker}.mp3"
        
        # Generate speech
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_file)
        
        # Add to full audio with a small pause
        clip = AudioSegment.from_mp3(temp_file)
        silence = AudioSegment.silent(duration=800) # 0.8s pause between speakers
        full_audio += clip + silence
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    # Export final file
    full_audio.export(f"test_cases/{scenario['filename']}", format="mp3")
    print(f"✅ Saved to test_cases/{scenario['filename']}")

async def main():
    print("🚀 Starting Test Case Generation...")
    for scenario in scenarios:
        await generate_audio(scenario)
    print("\n🎉 All 5 test cases generated successfully in the 'test_cases' folder!")

if __name__ == "__main__":
    asyncio.run(main())