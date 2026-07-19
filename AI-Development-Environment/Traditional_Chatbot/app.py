import os
import re
import time
import random
import datetime
import gradio as gr
from fastapi import FastAPI
import uvicorn

BOT_NAME = "SmartBot"

WELCOME_MESSAGE = (
    f"Hello! I am {BOT_NAME}, your rule-based personal assistant.\n"
    "Type 'help' to see what I can do."
)

GOODBYE_MESSAGE = (
    "Thank you for using SmartBot.\n"
    "Have a wonderful day!"
)

CAPABILITIES = [
    "Greet users",
    "Introduce myself",
    "Show today's date",
    "Show current time",
    "Tell jokes",
    "Provide motivational quotes",
    "Explain AI",
    "Explain Machine Learning",
    "Explain Deep Learning",
    "Explain Python",
    "Perform basic calculations",
    "Flip a coin",
    "Roll a dice",
    "Handle unknown questions politely"
]

JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "Debugging is like being a detective in a crime movie where you're also the murderer.",
    "Why did Python go to school? To improve its class."
]

QUOTES = [
    "Success is the sum of small efforts repeated every day.",
    "Learning never exhausts the mind.",
    "Stay curious and keep building."
]

KNOWLEDGE = {
    "ai": "Artificial Intelligence enables computers to perform tasks that normally require human intelligence.",
    "machine learning": "Machine Learning is a branch of AI where systems learn from data without being explicitly programmed.",
    "deep learning": "Deep Learning uses artificial neural networks with multiple layers to solve complex problems.",
    "python": "Python is a high-level programming language widely used in AI, automation, and data science."
}

def get_current_date():
    """Returns today's date."""
    return datetime.datetime.now().strftime("%d %B %Y")


def get_current_time():
    """Returns current time."""
    return datetime.datetime.now().strftime("%I:%M:%S %p")


def tell_joke():
    """Returns a random joke."""
    return random.choice(JOKES)


def motivational_quote():
    """Returns a random motivational quote."""
    return random.choice(QUOTES)


def flip_coin():
    """Simulates a coin toss."""
    return random.choice(["Heads", "Tails"])


def roll_dice():
    """Simulates rolling a six-sided dice."""
    return random.randint(1, 6)


def calculator(expression):
    """
    Performs a basic mathematical calculation.
    Example: 10 + 5
    """

    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Invalid mathematical expression."

INTENTS = {

    "greeting": [
        "hi", "hello", "hey",
        "good morning", "good afternoon", "good evening"
    ],

    "help": [
        "help", "capabilities",
        "what can you do", "commands"
    ],

    "date": [
        "date", "today"
    ],

    "time": [
        "time", "clock"
    ],

    "joke": [
        "joke", "funny", "make me laugh"
    ],

    "quote": [
        "quote", "motivate me", "motivation"
    ],

    "ai": [
        "ai", "artificial intelligence"
    ],

    "ml": [
        "machine learning", "ml"
    ],

    "dl": [
        "deep learning"
    ],

    "python": [
        "python"
    ],

    "coin": [
        "coin", "flip coin"
    ],

    "dice": [
        "dice", "roll dice"
    ],

    "bye": [
        "bye", "exit", "quit", "goodbye"
    ]
}

def detect_intent(user_input):

    user_input = user_input.lower().strip()

    # Split the sentence into words
    words = re.findall(r"\b\w+\b", user_input)

    for intent, keywords in INTENTS.items():

        for keyword in keywords:

            # Multi-word keyword
            if " " in keyword:
                if keyword in user_input:
                    return intent

            # Single-word keyword
            else:
                if keyword in words:
                    return intent

    return "unknown"
def display_capabilities():
    print("\nI can perform the following tasks:")
    for i, item in enumerate(CAPABILITIES, start=1):
        print(f"{i}. {item}")
    print()


def chatbot():

    print("=" * 60)
    print(WELCOME_MESSAGE)
    print("=" * 60)

    user_name = input("Before we start, what's your name? ").strip()

    if not user_name:
        user_name = "User"

    print(f"\nNice to meet you, {user_name}! 😊")
    print("You can start chatting now.\n")

    start_time = time.time()
    message_count = 0

    while True:

        user_input = input(f"{user_name}: ")

        message_count += 1

        intent = detect_intent(user_input)

        if intent == "greeting":
            print(f"{BOT_NAME}: Hello {user_name}! How can I help you today?\n")

        elif intent == "help":
            display_capabilities()

        elif intent == "date":
            print(f"{BOT_NAME}: Today's date is {get_current_date()}\n")

        elif intent == "time":
            print(f"{BOT_NAME}: Current time is {get_current_time()}\n")

        elif intent == "joke":
            print(f"{BOT_NAME}: {tell_joke()}\n")

        elif intent == "quote":
            print(f"{BOT_NAME}: {motivational_quote()}\n")

        elif intent == "ai":
            print(f"{BOT_NAME}: {KNOWLEDGE['ai']}\n")

        elif intent == "ml":
            print(f"{BOT_NAME}: {KNOWLEDGE['machine learning']}\n")

        elif intent == "dl":
            print(f"{BOT_NAME}: {KNOWLEDGE['deep learning']}\n")

        elif intent == "python":
            print(f"{BOT_NAME}: {KNOWLEDGE['python']}\n")

        elif intent == "coin":
            print(f"{BOT_NAME}: Coin Toss -> {flip_coin()}\n")

        elif intent == "dice":
            print(f"{BOT_NAME}: Dice Roll -> {roll_dice()}\n")

        elif intent == "bye":

            duration = round(time.time() - start_time, 2)

            print("\n===================================")
            print("Session Summary")
            print("===================================")
            print(f"User Name      : {user_name}")
            print(f"Messages       : {message_count}")
            print(f"Session Time   : {duration} seconds")
            print("===================================")
            print(GOODBYE_MESSAGE)

            break

        else:
            print(f"{BOT_NAME}: Sorry, I couldn't understand that.")
            print("Type 'help' to see what I can do.\n")

def smartbot_response(message, history):

    intent = detect_intent(message)

    if intent == "greeting":
        return f"Hello! I'm {BOT_NAME}. How can I help you today?"

    elif intent == "help":
        return "\n".join(
            [f"{i+1}. {item}" for i, item in enumerate(CAPABILITIES)]
        )

    elif intent == "date":
        return f"Today's date is {get_current_date()}"

    elif intent == "time":
        return f"Current time is {get_current_time()}"

    elif intent == "joke":
        return tell_joke()

    elif intent == "quote":
        return motivational_quote()

    elif intent == "ai":
        return KNOWLEDGE["ai"]

    elif intent == "ml":
        return KNOWLEDGE["machine learning"]

    elif intent == "dl":
        return KNOWLEDGE["deep learning"]

    elif intent == "python":
        return KNOWLEDGE["python"]

    elif intent == "coin":
        return f"Coin Toss → {flip_coin()}"

    elif intent == "dice":
        return f"Dice Roll → {roll_dice()}"

    elif intent == "bye":
        return "Goodbye! 👋 Have a wonderful day."

    else:
        return "Sorry, I couldn't understand that. Type 'help' to see my capabilities."



demo = gr.ChatInterface(
    fn=smartbot_response,
    title="🤖 SmartBot",
    description="Traditional Rule-Based Personal Assistant"
)

app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )