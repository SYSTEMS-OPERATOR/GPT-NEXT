import re
import random

# Reflection map for pronoun swapping
reflections = {
    "am": "are", "was": "were", "i": "you", "i'd": "you would", "i've": "you have",
    "i'll": "you will", "my": "your", "are": "am", "you're": "I'm", "you've": "I have",
    "you'll": "I will", "your": "my", "yours": "mine", "you": "me", "me": "you"
}

# Pattern-response pairs for the therapist (ELIZA/DOCTOR) persona
doctor_patterns = [
    (r'I need (.*)', [
        "Why do you need {0}?", 
        "Would it really help you to get {0}?", 
        "Are you sure you need {0}?"]),
    (r'Why don\'?t you ([^\?]*)\??', [
        "Do you really think I don't {0}?", 
        "Perhaps eventually I will {0}.", 
        "Do you really want me to {0}?"]),
    (r'Why can\'?t I ([^\?]*)\??', [
        "Do you think you should be able to {0}?", 
        "If you could {0}, what would you do?", 
        "What’s stopping you from {0}?"]),
    (r'I can\'?t (.*)', [
        "How do you know you can't {0}?", 
        "Perhaps you could {0} if you tried.", 
        "What would it take for you to {0}?"]),
    (r'I am (.*)', [
        "Did you come to me because you are {0}?", 
        "How do you feel about being {0}?"]),
    (r'I\'?m (.*)', [
        "How do you feel about being {0}?", 
        "Do you often feel {0}?"]),
    (r'You are (.*)', [
        "What makes you think I am {0}?", 
        "Does it please you to think that I'm {0}?"]),
    (r'What (.*)', [
        "Why do you ask?", 
        "What do you think?"]),
    (r'How (.*)', [
        "How do you suppose?", 
        "Perhaps you can answer your own question."]),
    (r'Because (.*)', [
        "Is that the real reason?", 
        "What other reasons come to mind?"]),
    (r'(.*) sorry (.*)', [
        "There's no need to apologize.", 
        "What feelings do you have when you apologize?"]),
    (r'Hello(.*)', [
        "Hello... I'm glad you came today.", 
        "Hi there, how can I help you?"]),
    (r'I think (.*)', [
        "Do you doubt {0}?", 
        "Do you really think so?"]),
    (r'(.*) friend (.*)', [
        "Tell me more about your friends.", 
        "Why not tell me about a childhood friend?"]),
    (r'Yes', [
        "You seem quite sure.", 
        "OK, but can you elaborate?"]),
    (r'(.*) computer(.*)', [
        "Are computers a source of concern for you?", 
        "What do you think about machines?"]),
    (r'Is it (.*)', [
        "Do you think it is {0}?"]),
    (r'It is (.*)', [
        "What makes you feel it is {0}?"]),
    (r'Can you (.*)', [
        "What makes you think I can't {0}?", 
        "Whether or not I can {0} is not the question."]),
    (r'Can I (.*)', [
        "Perhaps you don't want to {0}.", 
        "Do you want to be able to {0}?"]),
    (r'(.*) mother(.*)', [
        "Tell me more about your mother.", 
        "What was your relationship with your mother like?"]),
    (r'(.*) father(.*)', [
        "How did your father make you feel?", 
        "Tell me more about your father."]),
    (r'(.*) child(.*)', [
        "Did you have close friends as a child?", 
        "What is your favorite childhood memory?"]),
    (r'(.*)\?', [
        "Why do you ask that?", 
        "What do you think?"]),
    (r'quit', [
        "Thank you for talking with me.", 
        "Good-bye."]),
    (r'(.*)', [
        "Please tell me more.", 
        "Let's change focus a bit... Tell me about your family.", 
        "Can you elaborate on that?"])
]

# Use the same patterns for ELIZA persona (alias to doctor_patterns)
eliza_patterns = doctor_patterns

def reflect(fragment):
    """Reflects a fragment of input by swapping pronouns (I -> you, me -> you, etc.)."""
    tokens = fragment.lower().split()
    for i, token in enumerate(tokens):
        if token in reflections:
            tokens[i] = reflections[token]
    return " ".join(tokens)

def generate_response(message, persona="doctor"):
    """Generate a response to the user's message using the specified persona's rules."""
    patterns = doctor_patterns if persona == "doctor" else eliza_patterns
    for pattern, responses in patterns:
        match = re.match(pattern, message.strip(), re.IGNORECASE)
        if match:
            response_template = random.choice(responses)
            # If the response template expects a fragment, fill it in after reflection
            if '{0}' in response_template:
                fragment = match.group(1)
                fragment = reflect(fragment)
                return response_template.format(fragment)
            else:
                return response_template
    # Fallback (should not normally reach here because of last catch-all pattern)
    return "Interesting. Please continue."
