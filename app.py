from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import speech_recognition as sr
import random
app = Flask(__name__)
CORS(app)
# Load trained model
vectorizer, classifier = pickle.load(open("model.pkl", "rb"))
# Sample Questions Pool
questions_pool = [
    "How does Newton’s First Law of Motion apply to everyday life?",
    "Why is the Mona Lisa considered a masterpiece in art history?",
    "How did the discovery of gravity change scientific understanding?",
    "Why is the speed of light considered a universal constant?",
    "What impact did the invention of the telephone have on communication?",
    "Why is water's freezing point significant in scientific studies?",
    "How did World War II shape the modern world?",
    "Why are primary colors important in color theory?",
    "How does photosynthesis contribute to the Earth's ecosystem?",
    "What role does language play in shaping cultural identity?"
]
feedback_templates = {
    "A": "Excellent answer! You explained the concept clearly.",
    "B": "Good effort! Try to add more details for a perfect answer.",
    "C": "You're on the right track, but your answer needs more depth.",
    "D": "Keep trying! Review the topic and attempt again.",
}
def recognize_speech():
    """Captures speech and converts it to text"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand."
    except sr.RequestError:
        return "Speech recognition failed. Check your internet."

@app.route('/random-question', methods=['GET'])
def get_random_question():
    """API to send a random question"""
    question = random.choice(questions_pool)
    return jsonify({"question": question})

@app.route('/speech', methods=['GET'])
def speech_to_text():
    """API to capture speech and return text"""
    text = recognize_speech()
    return jsonify({"text": text})

@app.route('/grade', methods=['POST'])
def grade_answer():
    """API to grade student answers and provide feedback."""
    data = request.json
    student_answer = data.get("answer", "")

    # Convert input to TF-IDF and predict grade
    input_tfidf = vectorizer.transform([student_answer])
    predicted_grade = classifier.predict(input_tfidf)[0]

    # Generate feedback
    feedback = feedback_templates.get(predicted_grade, "Great attempt!")

    return jsonify({"grade": predicted_grade, "feedback": feedback})

if __name__ == '__main__':
    app.run(debug=True)
