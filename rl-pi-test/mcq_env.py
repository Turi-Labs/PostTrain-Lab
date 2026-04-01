"""
Multiple Choice Question (MCQ) Environment for Prime Intellect Environments Hub.

This environment teaches LLMs to answer multiple-choice questions with single-letter
answers in XML format. The model learns to extract the correct option letter from
a set of choices.

Example usage:
    >>> env = load_environment()
    >>> dataset = env.get_dataset()
    >>> print(dataset[0])
"""

from __future__ import annotations

import os
from typing import Any

import verifiers as vf
from datasets import Dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Monkey-patch Dataset to add repeat method that verifiers expects
def _dataset_repeat(self, n: int) -> Dataset:
    """Repeat each example in the dataset n times."""
    repeated_data = []
    for item in self:
        for _ in range(n):
            repeated_data.append(dict(item))
    return Dataset.from_list(repeated_data)

if not hasattr(Dataset, 'repeat'):
    Dataset.repeat = _dataset_repeat

# Sample dataset of 25 multiple-choice questions
dataset = [
    {
        "id": 1,
        "question": "What is the capital of France?",
        "options": {
            "A": "London",
            "B": "Berlin",
            "C": "Paris",
            "D": "Madrid"
        },
        "correctAnswer": "C",
        "explanation": "Paris is the capital and largest city of France."
    },
    {
        "id": 2,
        "question": "Which planet is known as the Red Planet?",
        "options": {
            "A": "Venus",
            "B": "Mars",
            "C": "Jupiter",
            "D": "Saturn"
        },
        "correctAnswer": "B",
        "explanation": "Mars is called the Red Planet due to iron oxide (rust) on its surface."
    },
    {
        "id": 3,
        "question": "What is the largest ocean on Earth?",
        "options": {
            "A": "Atlantic Ocean",
            "B": "Indian Ocean",
            "C": "Arctic Ocean",
            "D": "Pacific Ocean"
        },
        "correctAnswer": "D",
        "explanation": "The Pacific Ocean covers approximately 63 million square miles."
    },
    {
        "id": 4,
        "question": "Who wrote 'Romeo and Juliet'?",
        "options": {
            "A": "Charles Dickens",
            "B": "William Shakespeare",
            "C": "Jane Austen",
            "D": "Mark Twain"
        },
        "correctAnswer": "B",
        "explanation": "William Shakespeare wrote this famous tragedy in the 1590s."
    },
    {
        "id": 5,
        "question": "What is the chemical symbol for gold?",
        "options": {
            "A": "Go",
            "B": "Gd",
            "C": "Au",
            "D": "Ag"
        },
        "correctAnswer": "C",
        "explanation": "Au comes from the Latin word 'aurum' meaning gold."
    },
    {
        "id": 6,
        "question": "What is the smallest prime number?",
        "options": {
            "A": "0",
            "B": "1",
            "C": "2",
            "D": "3"
        },
        "correctAnswer": "C",
        "explanation": "2 is the smallest and only even prime number."
    },
    {
        "id": 7,
        "question": "Which gas makes up approximately 78% of Earth's atmosphere?",
        "options": {
            "A": "Oxygen",
            "B": "Carbon Dioxide",
            "C": "Nitrogen",
            "D": "Argon"
        },
        "correctAnswer": "C",
        "explanation": "Nitrogen makes up about 78% of Earth's atmosphere by volume."
    },
    {
        "id": 8,
        "question": "What is the hardest natural substance on Earth?",
        "options": {
            "A": "Gold",
            "B": "Iron",
            "C": "Diamond",
            "D": "Quartz"
        },
        "correctAnswer": "C",
        "explanation": "Diamond is the hardest natural substance, rating 10 on the Mohs scale."
    },
    {
        "id": 9,
        "question": "Which continent is the largest by land area?",
        "options": {
            "A": "Africa",
            "B": "Asia",
            "C": "North America",
            "D": "Europe"
        },
        "correctAnswer": "B",
        "explanation": "Asia is the largest continent, covering about 30% of Earth's land area."
    },
    {
        "id": 10,
        "question": "What is the speed of light in a vacuum?",
        "options": {
            "A": "300,000 km/s",
            "B": "299,792,458 m/s",
            "C": "186,000 miles/s",
            "D": "All of the above"
        },
        "correctAnswer": "D",
        "explanation": "All options represent the same speed of light, just in different units."
    },
    {
        "id": 11,
        "question": "Who painted the Mona Lisa?",
        "options": {
            "A": "Vincent van Gogh",
            "B": "Pablo Picasso",
            "C": "Leonardo da Vinci",
            "D": "Michelangelo"
        },
        "correctAnswer": "C",
        "explanation": "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519."
    },
    {
        "id": 12,
        "question": "What is the largest mammal in the world?",
        "options": {
            "A": "African Elephant",
            "B": "Blue Whale",
            "C": "Giraffe",
            "D": "Hippopotamus"
        },
        "correctAnswer": "B",
        "explanation": "The Blue Whale is the largest mammal and largest animal ever known to exist."
    },
    {
        "id": 13,
        "question": "Which element has the atomic number 1?",
        "options": {
            "A": "Helium",
            "B": "Hydrogen",
            "C": "Lithium",
            "D": "Carbon"
        },
        "correctAnswer": "B",
        "explanation": "Hydrogen has atomic number 1, meaning it has one proton in its nucleus."
    },
    {
        "id": 14,
        "question": "What is the capital of Australia?",
        "options": {
            "A": "Sydney",
            "B": "Melbourne",
            "C": "Canberra",
            "D": "Perth"
        },
        "correctAnswer": "C",
        "explanation": "Canberra is the capital city of Australia, not Sydney or Melbourne."
    },
    {
        "id": 15,
        "question": "How many sides does a hexagon have?",
        "options": {
            "A": "5",
            "B": "6",
            "C": "7",
            "D": "8"
        },
        "correctAnswer": "B",
        "explanation": "A hexagon is a polygon with six sides and six angles."
    },
    {
        "id": 16,
        "question": "Which organ in the human body produces insulin?",
        "options": {
            "A": "Liver",
            "B": "Kidney",
            "C": "Pancreas",
            "D": "Heart"
        },
        "correctAnswer": "C",
        "explanation": "The pancreas produces insulin, which regulates blood sugar levels."
    },
    {
        "id": 17,
        "question": "What is the longest river in the world?",
        "options": {
            "A": "Amazon River",
            "B": "Nile River",
            "C": "Mississippi River",
            "D": "Yangtze River"
        },
        "correctAnswer": "B",
        "explanation": "The Nile River is generally considered the longest river at about 6,650 km."
    },
    {
        "id": 18,
        "question": "Which programming language was created by Guido van Rossum?",
        "options": {
            "A": "Java",
            "B": "Python",
            "C": "C++",
            "D": "JavaScript"
        },
        "correctAnswer": "B",
        "explanation": "Python was created by Guido van Rossum and first released in 1991."
    },
    {
        "id": 19,
        "question": "What is the freezing point of water in Celsius?",
        "options": {
            "A": "32°C",
            "B": "100°C",
            "C": "0°C",
            "D": "-273°C"
        },
        "correctAnswer": "C",
        "explanation": "Water freezes at 0°C (32°F) under standard atmospheric pressure."
    },
    {
        "id": 20,
        "question": "Which country has the most time zones?",
        "options": {
            "A": "United States",
            "B": "Russia",
            "C": "China",
            "D": "France"
        },
        "correctAnswer": "D",
        "explanation": "France has 12 time zones due to its overseas territories, more than any other country."
    },
    {
        "id": 21,
        "question": "What is the square root of 144?",
        "options": {
            "A": "11",
            "B": "12",
            "C": "13",
            "D": "14"
        },
        "correctAnswer": "B",
        "explanation": "The square root of 144 is 12, since 12 × 12 = 144."
    },
    {
        "id": 22,
        "question": "Which vitamin is produced when skin is exposed to sunlight?",
        "options": {
            "A": "Vitamin A",
            "B": "Vitamin B",
            "C": "Vitamin C",
            "D": "Vitamin D"
        },
        "correctAnswer": "D",
        "explanation": "Vitamin D is synthesized in the skin when exposed to UVB radiation from sunlight."
    },
    {
        "id": 23,
        "question": "What is the currency of Japan?",
        "options": {
            "A": "Yuan",
            "B": "Won",
            "C": "Yen",
            "D": "Rupee"
        },
        "correctAnswer": "C",
        "explanation": "The Japanese Yen is the official currency of Japan."
    },
    {
        "id": 24,
        "question": "Which scientist developed the theory of relativity?",
        "options": {
            "A": "Isaac Newton",
            "B": "Albert Einstein",
            "C": "Galileo Galilei",
            "D": "Stephen Hawking"
        },
        "correctAnswer": "B",
        "explanation": "Albert Einstein developed both special and general theories of relativity."
    },
    {
        "id": 25,
        "question": "How many bones are in an adult human body?",
        "options": {
            "A": "196",
            "B": "206",
            "C": "216",
            "D": "226"
        },
        "correctAnswer": "B",
        "explanation": "An adult human body typically has 206 bones."
    }
]


def build_dataset() -> Dataset:
    """
    Build a HuggingFace Dataset from the MCQ questions.
    
    Returns:
        Dataset with columns: question, answer, explanation, task, info
    """
    data = []
    for item in dataset:
        formatted_question = f"{item['question']}\n\n"
        for key, value in item['options'].items():
            formatted_question += f"{key}. {value}\n"
        
        data.append({
            "question": formatted_question,
            "answer": item["correctAnswer"],
            "explanation": item["explanation"],
            "task": "multiple-choice-qa",  # Required by VerifiersRLDatasetBuilder
            "info": {},  # Optional metadata
        })
    return Dataset.from_list(data)


def evaluate_answer(completion: str, answer: str, parser: vf.XMLParser) -> float:
    """
    Evaluate the model's answer against the ground truth.
    
    Args:
        completion: The model's generated completion text
        answer: The correct answer (single letter A-D)
        parser: XML parser to extract answer from completion
    
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    response = parser.parse_answer(completion)
    return 1.0 if response.strip() == answer.strip() else 0.0


system_prompt = """
You are a helpful assistant that answers multiple choice questions.

You will be given a question and options. Respond with the correct option letter only out of 4 Letters, in XML format:

<answer>option</answer>
"""


def load_environment(**kwargs) -> vf.Environment:
    """
    Load the MCQ environment.
    
    This function is the entry point for the Verifiers framework. It creates
    a single-turn environment where the model receives a question and must
    respond with the correct answer in XML format.
    
    Args:
        **kwargs: Additional configuration arguments (currently unused)
    
    Returns:
        A configured vf.Environment instance
    """
    dataset = build_dataset()
    parser = vf.XMLParser(["answer"], answer_field="answer")

    rubric = vf.Rubric(
    funcs=[evaluate_answer],
    weights=[1.0],
    parser=parser
    )
    
    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric
    )


if __name__ == "__main__":
    """
    Test the environment locally with OpenAI API.
    Requires OPENAI_API_KEY in .env file.
    """
    import asyncio
    from openai import AsyncOpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    env = load_environment()
    client = AsyncOpenAI(api_key=api_key)
    
    print("Testing MCQ Environment")
    print("=" * 60)
    
    async def test_env():
        results = env.evaluate_sync(
            client=client,
            model="gpt-4o-mini",
            num_examples=5,
            rollouts_per_example=2,
            max_concurrent=4,
            sampling_args={
                "max_tokens": 50,
                "temperature": 0.7,
            }
        )
        
        outputs = results["outputs"]
        rewards = [o["reward"] for o in outputs]
        
        print(f"\nResults:")
        print(f"Average Reward: {sum(rewards) / len(rewards):.3f}")
        print(f"Total Evaluations: {len(rewards)}")
        
        # Show a sample
        o = outputs[0]
        print("\n" + "=" * 60)
        print("Sample Question:")
        if o.get("prompt"):
            print(vf.utils.message_utils.messages_to_printable(o["prompt"]))
        print("\nSample Response:")
        if o.get("completion"):
            print(vf.utils.message_utils.messages_to_printable(o["completion"]))
        print(f"\nReward: {o['reward']}")
    
    asyncio.run(test_env())
