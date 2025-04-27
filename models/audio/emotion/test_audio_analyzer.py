import os
from audio_processor import SimpleAudioAnalyzer

# Initialize the audio analyzer
print("Initializing SimpleAudioAnalyzer...")
analyzer = SimpleAudioAnalyzer.get_instance()

# Find a test file
test_file = None
for root, dirs, files in os.walk("data"):
    for file in files:
        if file.endswith(".wav"):
            test_file = os.path.join(root, file)
            print(f"Found audio file: {test_file}")
            break
    if test_file:
        break

if not test_file:
    print("No test file found!")
    exit(1)

# Test analysis
print("\nAnalyzing audio...")
result = analyzer.analyze_audio(test_file)

print("\nResults:")
print(f"Detected emotion: {result['detected_emotion']}")
print("Probabilities:")
for emotion, prob in result.get('probabilities', {}).items():
    print(f"  {emotion}: {prob:.4f}")

print("\nTest completed successfully!") 