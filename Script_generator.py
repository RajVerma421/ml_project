from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import traceback
import unicodedata
import re

model = None

ILLEGAL_KEYWORDS = [
    "hack", "hacker's", "crack", "drug", "drugs", "bomb", "bombs", "weapon", "weapons",
    "explosive", "explosives", "poison", "poisons", "kill", "killing", "murder",
    "attack", "terrorist", "terrorism", "fraud", "scam", "phishing", "malware",
    "virus", "ransomware", "stolen", "theft", "steal", "illegal", "pirated",
    "crack", "fake", "counterfeit", "money", "counterfeit", "child", "children",
    "abuse", "abusive", "explicit", "porn", "pornography", "nude", "naked",
    " nude", "nsfw", "sex", "weapon", "gun", "guns", "ammo", "bullet",
    "isis", "isis", "killing", "racist", "hate", "discriminat",
    "discrimination", "suicide", "self-harm", "cutting", "eating disorder",
    "anorexia", "bulimia", "pro-ana", "pro-mia", "extremist",
    "white supremac", "nazi", "neo-nazi", "pedophil", "pedo", "grooming"
]

def get_model():
    global model
    if model is None:
        try:
            model = OllamaLLM(model="mistral:7b")
        except Exception as e:
            print(f"Ollama not available: {e}")
            model = False
    return model

def is_illegal_content(topic, content_type):
    topic_lower = topic.lower()
    content_type_lower = content_type.lower()
    
    for keyword in ILLEGAL_KEYWORDS:
        if keyword in topic_lower or keyword in content_type_lower:
            return True
    
    illegal_content_types = ["hack", "crack", "fraud", "scam", "pirated", "fake id", "fake passport"]
    for illegal_type in illegal_content_types:
        if illegal_type in topic_lower or illegal_type in content_type_lower:
            return True
    
    return False

def remove_emojis(text):
    result = []
    for c in text:
        cat = unicodedata.category(c)
        if cat in ('Ll', 'Lu', 'Lo', 'Lt', 'Lm', 'Nd', 'Nl', 'No', 'Pc', 'Pd', 'Ps', 'Pe', 'Pi', 'Pf', 'Po', 'Space'):
            result.append(c)
        elif cat == 'Zs':
            result.append(' ')
        else:
            result.append(' ')
    return ''.join(result)

def Script(content_type, topic, emotion):
    if is_illegal_content(topic, content_type):
        return "Sorry, I cannot generate content for this request. Please choose a legal and appropriate topic."
    
    ollama = get_model()
    
    if ollama is False:
        return generate_fallback_script(content_type, topic, emotion)
    
    promptTemplate = PromptTemplate(
        template="""
You are a helpful assistant that generates ONLY legal, appropriate, and educational content.

Generate a short {content_type} script about "{topic}"

IMPORTANT RULES:
- ONLY generate legal, educational, or entertainment content
- NEVER generate anything related to: hacking, illegal activities, harmful content, weapons, drugs, fraud, or explicit material
- If the topic is inappropriate or illegal, respond with "Sorry"
- Content must be appropriate for all audiences
- EXACTLY 5-6 lines only
- Each line under 6 words
- Conversational tone
- No emojis
- No extra text 

OUTPUT:
Only the script lines or "Sorry" if inappropriate.
""",
        input_variables=["content_type", "topic", "emotion"]
    )
    
    try:
        chain = promptTemplate | ollama
        response = chain.invoke({
            "content_type": content_type,
            "topic": topic,
            "emotion": emotion
        })
        script = response.strip() if hasattr(response, 'strip') else str(response)
        return remove_emojis(script)
    except Exception as e:
        print(f"Ollama error: {e}")
        return generate_fallback_script(content_type, topic, emotion)

def generate_fallback_script(content_type, topic, emotion):
    if is_illegal_content(topic, content_type):
        return "Sorry, I cannot generate content for this request. Please choose a legal and appropriate topic."
    
    hooks = {
        "Reel": [
            "Wow! " + topic + " is amazing!",
            "Did you know this?",
            "Here is the secret!",
            "This changes everything!",
            "You need to try this!",
            "Believe me, it works!"
        ],
        "YouTube Short": [
            "POV: " + topic,
            "This actually works!",
            "Trust me on this!",
            "Game changing tip!",
            "Must try today!",
            "You will thank me!"
        ],
        "TikTok": [
            "POV: " + topic,
            "Not many know this!",
            "Tutorial time!",
            "This is your sign!",
            "Works every time!",
            "3... 2... 1... like!"
        ],
        "Advertisement": [
            "Introducing " + topic,
            "This is the solution!",
            "Join thousands now!",
            "Try it FREE today!",
            "No credit card!",
            "Click NOW!"
        ],
        "Post": [
            topic + " explained!",
            "Key facts here!",
            "Must know tips!",
            "Top strategies!",
            "Best practices!",
            "Your guide!"
        ]
    }
    
    selected = hooks.get(content_type, hooks["Reel"])
    emotion_prefix = {
        "Excited": "Wow! ",
        "Motivational": "You can do this! ",
        "Funny": "Haha! ",
        "Serious": "Important: ",
        "Inspirational": "Remember: "
    }
    
    lines = []
    for i, hook in enumerate(selected[:10]):
        if i == 0:
            lines.append(emotion_prefix.get(emotion, "") + hook)
        elif i == len(selected) - 1:
            lines.append("Share now! Like! Follow!")
        else:
            lines.append(f"Point {i+1}: " + hook)
    
    return "\n".join(lines)