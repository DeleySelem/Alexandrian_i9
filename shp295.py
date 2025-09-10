import os
import re
import json
import time
import random
import numpy as np
import requests
#import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime
from colorama import Fore, Style, init
import OmniSourceNLP2 as omni
import ResponseLogic4 as er2
import ResponseLogic2 as er
from OmniSourceNLP4 import NeuroLinguisticProcessor
#from ResponseLogic2 import get_enhanced_response
#from OmniSourceNLP3 EnhancedNeuroMorphicProcessor
import difflib
from difflib import SequenceMatcher
import sys
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp

# Initialize colorama with custom colors
init(autoreset=True)
Fore.BLUE_DARK = "\033[34m"

stop_words_loader = NeuroLinguisticProcessor()
STOP_WORDS = stop_words_loader._load_stop_words()  # Words to exclude from premonition and spectrum
HEXAGRAMS = [
    ("䷀", "The Creative"), ("䷁", "The Receptive"), ("䷂", "Difficulty at the Beginning"),
    ("䷃", "Youthful Folly"), ("䷄", "Waiting"), ("䷅", "Conflict"),
    ("䷆", "Army"), ("䷇", "Holding Together"), ("䷈", "Small Taming"),
    ("䷉", "Treading"), ("䷊", "Peace"), ("䷋", "Standstill"),
    ("䷌", "Fellowship"), ("䷍", "Great Possession"), ("䷎", "Modesty"),
    ("䷏", "Enthusiasm"), ("䷐", "Following"), ("䷑", "Work on the Decayed"),
    ("䷒", "Approach"), ("䷓", "Contemplation"), ("䷔", "Biting Through"),
    ("䷕", "Grace"), ("䷖", "Splitting Apart"), ("䷗", "Return"),
    ("䷘", "Innocence"), ("䷙", "Great Taming"), ("䷚", "Nourishment"),
    ("䷛", "Great Excess"), ("䷜", "Water"), ("䷝", "Fire"),
    ("䷞", "Clinging Fire"), ("䷟", "Lake"), ("䷠", "Mountain"),
    ("䷡", "Thunder"), ("䷢", "Wind"), ("䷣", "Water over Fire"),
    ("䷤", "Fire over Water"), ("䷥", "Abundance"), ("䷦", "Traveling"),
    ("䷧", "Wandering"), ("䷨", "Pushing Upward"), ("䷩", "Darkening of the Light"),
    ("䷪", "Family"), ("䷫", "Opposition"), ("䷬", "Obstruction"),
    ("䷭", "Deliverance"), ("䷮", "Decrease"), ("䷯", "Increase"),
    ("䷰", "Breakthrough"), ("䷱", "Coming to Meet"), ("䷲", "Gathering"),
    ("䷳", "Pressing Onward"), ("䷴", "Well"), ("䷵", "Revolution"),
    ("䷶", "Cauldron"), ("䷷", "Shock"), ("䷸", "Gentle"),
    ("䷹", "Joyous"), ("䷺", "Dispersing"), ("䷻", "Limiting"),
    ("䷼", "Inner Truth"), ("䷽", "Small Excess"), ("䷾", "After Completion"),
    ("䷿", "Before Completion"),
]



class ResponseParameters:
    def __init__(self):
        self.primary_sentences = 1
        self.secondary_sentences = 1
        self.randomization = 50
        self.similarity_threshold = 0.3333
        self.multifractal_weight = 0.9725
        self.last_feedback = None
        self.relevance_threshold = 1  # New parameter
        self.relevance_count = 0
    def adjust_for_relevance(self):
        if self.relevance_count >= 5:
            self.relevance_threshold = 5
        elif self.relevance_count >= 4:
            self.relevance_threshold = 4
        elif self.relevance_count >= 3:
            self.relevance_threshold = 3
        elif self.relevance_count >= 2:
            self.relevance_threshold = 2
        elif self.relevance_count >= 1:
            self.relevance_threshold = 1
    def adjust_parameters(self, hex_grid):
        green_count = sum(1 for row in hex_grid.rows for h in row if h["color"] == "green")
        red_count = sum(1 for row in hex_grid.rows for h in row if h["color"] == "red")
        self.randomization = max(0, min(100, self.randomization + (green_count - red_count)))
        if random.random() < 0.2:
            self.primary_sentences += random.randint(-1, 1)
            self.secondary_sentences += random.randint(-1, 1)

class FeedbackAnalyzer:
    def __init__(self):
        self.feedback_data = []
        self.load_feedback()
        self.min_similarity = 0.3333
    
    def load_feedback(self):
        if os.path.exists("feedback.json"):
            with open("feedback.json", "r") as f:
                try:
                    self.feedback_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"{Fore.YELLOW}Malformed feedback.json detected. Initializing as an empty list.")
                    self.feedback_data = []
    
    def find_most_similar(self, query):
        """Finds the most similar feedback entry to the given query."""
        best_match = None
        highest_sim = 0
        for entry in self.feedback_data:
            if 'original_input' in entry:
                sim = difflib.SequenceMatcher(None, query, entry['original_input']).ratio()
                if sim > highest_sim:
                    highest_sim = sim
                    best_match = entry
        return best_match, highest_sim


def autocorrect_json_file(file_path):
    """Attempts to autocorrect a malformed JSON file."""
    try:
        # Attempt to load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)  # This will raise JSONDecodeError if malformed
        return data  # If successful, return the loaded data

    except json.JSONDecodeError:
        print(f"{Fore.YELLOW}Malformed JSON detected in '{file_path}'. Attempting to autocorrect...")

        # Attempt to autocorrect the file
        corrected_data = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    # Try to parse each line as a JSON object
                    corrected_data.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip lines that cannot be parsed
                    continue

        # If correction was successful, save the corrected data
        if corrected_data:
            with open(file_path, 'w') as f:
                json.dump(corrected_data, f, indent=4)
            print(f"{Fore.GREEN}Autocorrection successful. '{file_path}' has been fixed.")
            return corrected_data

        # If the file is still malformed, replace it with an empty file
        else:
            print(f"{Fore.RED}Autocorrection failed. Replacing '{file_path}' with a new empty file.")
            with open(file_path, 'w') as f:
                json.dump([], f, indent=4)  # Replace with an empty list
            return []
class NeuroMorphicProcessor:
    def __init__(self):
        self.feedback_data = []
        self.feedback_file = 'feedback.json'
        self.min_similarity = 0.3333
        self.response_length_modifier = 0.3333
        self.response_history = {}
        self.multifractal_params = {}
    
    def get_applicable_feedback(self, user_input):
        feedback_data = autocorrect_json_file(self.feedback_file)
        applicable = []
        for entry in feedback_data:
            sim = difflib.SequenceMatcher(None, user_input, entry['original_input']).ratio()
            if sim >= 0.6666:
                applicable.append(entry)
                
                # Count "more relevant" feedbacks
                if entry['feedback_type'] == "relevance_up":
                    self.relevance_count = sum(1 for e in feedback_data 
                                           if e['feedback_type'] == "relevance_up")
        return applicable
  
    def save_feedback(self, user_input, ai_response, feedback, enhancement):
        feedback_data = autocorrect_json_file(self.feedback_file)
        timestamp = datetime.now().isoformat()
        
        new_entry = {
            'original_input': user_input,  # Store original input
            'response': ai_response,
            'feedback': feedback,
            'enhancement': enhancement,
            'timestamp': timestamp,
            'feedback_type': self._classify_feedback(enhancement)
        }
        feedback_data.append(new_entry)
        with open(self.feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=4)
    @staticmethod
    def save_to_inputs_json(user_input, user_response):
        """Saves user input and response to 'inputs.json'."""
        inputs_file = 'inputs.json'
        inputs_data = autocorrect_json_file(inputs_file)
    
        # Ensure inputs_data is a list
        if not isinstance(inputs_data, list):
            print(f"{Fore.YELLOW}Warning: Malformed 'inputs.json'. Resetting to an empty list.")
            inputs_data = []

        # Append the new user input and response
        inputs_data.append({
            'input': user_input,
            'response': user_response
        })

        # Save back to the file
        with open(inputs_file, 'w') as f:
            json.dump(inputs_data, f, indent=4)
    
    def get_enhanced_responses(self, user_input):
        # Check inputs.json first
        if os.path.exists("inputs.json"):
            try:
                with open("inputs.json", "r") as f:
                    inputs_data = json.load(f)
                    if not isinstance(inputs_data, list):
                        print(f"{Fore.YELLOW}Warning: Malformed 'inputs.json'. Resetting to an empty list.")
                        inputs_data = []
                    for entry in inputs_data:
                        if isinstance(entry, dict) and entry.get("input", "").strip().lower() == user_input.strip().lower():
                            return f"{Style.BRIGHT}{Fore.GREEN} {entry.get('response', '').strip()}"
            except Exception:
                pass  # Silent error handling

        try:
            with open(self.feedback_file, 'r') as f:
                feedback_data = json.load(f)
        except Exception as e:
            print(f"{Fore.RED}Error loading feedback: {e}")
            return None

        best_match = None
        highest_sim = 0
        for entry in feedback_data:
            if isinstance(entry, dict) and "input" in entry:
                sim = difflib.SequenceMatcher(None, user_input.lower(), entry['input'].lower()).ratio()
                if sim > self.min_similarity and sim > highest_sim:
                    highest_sim = sim
                    best_match = entry
        return best_match.get('response') if best_match else None


    def calculate_multifractal_spectrum(self):
        """Calculates the multifractal spectrum from 'conversation.log'."""
        if not os.path.exists('conversation.log'):
            return {}

        with open('conversation.log', 'r') as f:
            text = f.read().lower()

        words = [word for word in re.findall(r'\w+', text) if word not in STOP_WORDS]
        total_words = len(words)
        word_counts = Counter(words)

        q_values = np.linspace(-5, 5, 21)
        tau = []
        for q in q_values:
            if q != 1:
                zq = sum(np.power(count / total_words, q) for count in word_counts.values())
                tau.append(np.log(zq) / np.log(5))
            else:
                entropy = sum(-(count / total_words) * np.log(count / total_words) for count in word_counts.values())
                tau.append(entropy / np.log(5))

        self.multifractal_params = {'q': q_values, 'tau': tau}
        return self.multifractal_params
    def _classify_feedback(self, enhancement):
        enhancement = enhancement.lower()
        if any(w in enhancement for w in ["shorter", "less sentences"]):
            return "length_adjustment"
        elif "more random" in enhancement:
            return "randomness_up"
        elif "less random" in enhancement:
            return "randomness_down"
        elif "more relevant" in enhancement:
            return "relevance_up"
        return "general"

class HexagramGrid:
    def __init__(self):
        self.rows = []
        self.color_map = {"red": Fore.RED, "yellow": Fore.YELLOW, "green": Fore.GREEN}
        self.word_bank = []
        self.response_params = ResponseParameters()
        self.init_word_bank()
        self.init_grid()

    def init_word_bank(self):
        """Initializes the word bank from 'conversation.log'."""
        if os.path.exists('conversation.log'):
            with open('conversation.log', 'r') as f:
                text = f.read().lower()
                self.word_bank = [
                    word for word in re.findall(r'\w+', text) if word not in STOP_WORDS
                ]

    def init_grid(self):
        """Initializes the hexagram grid."""
        hexagrams = random.sample(HEXAGRAMS, 64)
        for i in range(8):
            row = []
            for j in range(8):
                symbol, name = hexagrams[i * 8 + j]
                row.append({
                    "symbol": symbol,
                    "name": name,
                    "color": random.choice(list(self.color_map.keys())),
                    "position": (i, j),
                    "lines": [self.create_line() for _ in range(6)]
                })
            self.rows.append(row)

    def create_line(self):
        """Creates a line with a random state and word."""
        return {
            'state': 'closed' if random.random() < 0.5 else 'open',
            'word': random.choice(self.word_bank) if self.word_bank else ""
        }

    def display(self):
        """Displays the hexagram grid."""
        for row in self.rows:
            line = []
            for hexagram in row:
                color = self.color_map[hexagram["color"]]
                symbol = hexagram["symbol"]
                lines = ''.join(['-' if l['state'] == 'closed' else '○' for l in hexagram['lines']])
                line.append(f"{color}{symbol} {lines}{Style.RESET_ALL}")
            print("  ".join(line))
def calculate_y_probability(spectrum, hex_grid):
    """Calculates the probability based on the multifractal spectrum and hex_grid."""
    green_count = sum(1 for row in hex_grid.rows for h in row if h["color"] == "green")
    red_count = sum(1 for row in hex_grid.rows for h in row if h["color"] == "red")
    return green_count / (green_count + red_count + 1e-8)

def generate_future_sentences(word_freq):
    """Generate future sentences based on word frequency."""
    return [f"I sense that the word '{word}' will be significant." for word, _ in word_freq]

def process_conversation_log():
    """Processes the conversation log to perform multifractal analysis and generate premonitions."""
    if not os.path.exists('conversation.log'):
        print(f"{Fore.YELLOW}No conversation.log file found. Premonition system disabled.")
        return []

    try:
        with open('conversation.log', 'r') as f:
            text = f.read().lower()
        words = [word for word in re.findall(r'\w+', text) if word not in STOP_WORDS]
        word_freq = Counter(words)

        # Reverse time by mirroring the log timestamps
        reversed_words = list(reversed(words))

        # Predict three "future" words
        most_common_future_words = word_freq.most_common(3)
        future_words = [word for word, _ in most_common_future_words]

        print(f"\n{Fore.CYAN}Premonition System Active:")
        print(f"{Fore.GREEN}Predicted future words: {', '.join(future_words)}")
        return future_words
    except Exception as e:
        print(f"Error processing conversation.log: {e}")
        return []

def generate_future_sentences2(word_freq):
    """Generate future sentences based on word frequency."""
    return [f"I sense that the word '{word}' will be significant." for word, _ in word_freq]

def process_conversation_log2():
    """Processes the conversation log to perform multifractal analysis and generate premonitions."""
    if not os.path.exists('conversation.log'):
        print(f"{Fore.YELLOW}No conversation.log file found. Premonition system disabled.")
        return []

    try:
        with open('conversation.log', 'r') as f:
            text = f.read().lower()
        words = [word for word in re.findall(r'\w+', text) if word not in STOP_WORDS]
        word_freq = Counter(words)

        # Reverse time by mirroring the log timestamps
        reversed_words = list(reversed(words))
        

        # Predict three "future" words
        most_common_future_words = word_freq.most_common(3)
        future_words = [word for word, _ in most_common_future_words]

        print(f"\n{Fore.CYAN}Premonition System Active:")
        print(f"{Fore.GREEN}Predicted future words: {', '.join(future_words)}")
        return future_words
    except Exception as e:
        print(f"Error processing conversation.log: {e}")
        return []

def fetch_wikipedia_sentences(word):
    """Fetches sentences from Wikipedia."""
    try:
        url = f"https://en.wikipedia.org/wiki/{word.capitalize()}"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            sentences = [
                re.sub(r"\s+:", " ", p.text.strip())
                for p in soup.find_all("p")
            ]
            # Filter out unwanted patterns
            filtered_sentences = [
                s for s in sentences
                if word.lower() in s.lower() and not re.search(r"[\[\]]|\[\d+\]:", s)
            ]
            return filtered_sentences[:2]
    except Exception as e:
        print(f"Error fetching Wikipedia for {word}: {e}")
        return []
        
def filter_relevant_sentences(response, subject_words, similarity_threshold=0.73):
    """
    Filters sentences in the response to only include those that have words with a similarity above the threshold.

    Parameters:
        response (str): The full response text to be filtered.
        subject_words (list): The list of subject words to check against.
        similarity_threshold (float): The minimum similarity required to keep a sentence.

    Returns:
        str: The filtered response containing only relevant sentences.
    """
    # Split response into sentences using punctuation as delimiters
    sentences = re.split(r'(?<=[.!?])\s+', response)

    # Define a list to hold relevant sentences
    relevant_sentences = []

    # Check each sentence for words similar to the subject words
    for sentence in sentences:
        words_in_sentence = sentence.split()
        for word_in_sentence in words_in_sentence:
            for subject_word in subject_words:
                # Calculate similarity between the words
                similarity = SequenceMatcher(None, word_in_sentence.lower(), subject_word.lower()).ratio()
                if similarity >= similarity_threshold:
                    relevant_sentences.append(sentence)
                    break  # No need to check other words if this one matches
            else:
                continue
            break  # Stop checking the sentence if a match is found

    # Join relevant sentences back into a single response
    return ' '.join(relevant_sentences)


async def fetch_wikipedia_sentences_async(word):
    """Asynchronously fetch sentences from Wikipedia."""
    url = f"https://en.wikipedia.org/wiki/{word.capitalize()}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    sentences = [
                        re.sub(r"\[\d+\]", "", p.text.strip())  # Remove numeric brackets like [1], [2]
                        for p in soup.find_all("p")
                    ]
                    # Filter out unwanted patterns
                    filtered_sentences = [
                        s for s in sentences
                        if word.lower() in s.lower() and not re.search(r"\[|\d+:", s)
                    ]
                    return filtered_sentences[:2]
    except Exception as e:
        print(f"Error fetching Wikipedia for {word}: {e}")
        return []

async def fetch_all_sentences_async(words):
    """Fetch Wikipedia sentences for all words asynchronously."""
    tasks = [fetch_wikipedia_sentences_async(word) for word in words]
    results = await asyncio.gather(*tasks)
    all_sentences = []
    for result in results:
        if result:
            all_sentences.extend(result)
    return all_sentences
def generate_response(user_input, nlp_processor, neuromorph, hex_grid, resources):
    """
    Generates a response based on user input, including:
      - Help command
      - Checking inputs.json for predefined responses
      - Printing and plotting multifractal spectrum
      - Fallback for cases where no predefined response is found
    """
    params = hex_grid.response_params
    feedback_analyzer = FeedbackAnalyzer()

    # Handle "help" or "commands" command first
    if user_input.lower() in ["help", "commands"]:
        return """\n==================COMMANDS LIST===================
help/commands: this help screen
print multifractal spectrum: prints out the multifractal spectrum in numbers.
plot multifractal spectrum: spawns the multifractal spectrum plot.
plot multifractal spectrum log: spawns the logarithmic multifractal spectrum plot.

Teach mode trigger words:
"Respond:" a trigger word to the feedback to teach the AI to respond with user-taught responses.
           The sentence after "Respond:" is taught to the AI and will override every other res-
           ponding mechanisms like: [Respond: <user input>]\n"""

    # Handle "print multifractal spectrum" command
    if user_input.lower() == "print multifractal spectrum":
        spectrum = neuromorph.calculate_multifractal_spectrum()
        print(f"Multifractal Spectrum:\nQ Values: {spectrum['q']}\nTau Values: {spectrum['tau']}")
        return "Multifractal spectrum printed in console."

    # Handle "plot multifractal spectrum" command
    if user_input.lower() == "plot multifractal spectrum":
        spectrum = neuromorph.calculate_multifractal_spectrum()
        plt.plot(spectrum['q'], spectrum['tau'])
        plt.xlabel('q')
        plt.ylabel('τ(q)')
        plt.title('Multifractal Spectrum (Excluding Stop Words)')
        plt.show()
        return "Multifractal spectrum plot displayed."

    # Handle "plot multifractal spectrum log" command
    if user_input.lower() == "plot multifractal spectrum log":
        spectrum = neuromorph.calculate_multifractal_spectrum()
        plt.figure(figsize=(8, 6))
        plt.plot(spectrum['q'], spectrum['tau'], marker='o', label='Multifractal Spectrum')
        plt.xscale('log')
        plt.xlabel('q (log scale)', fontsize=12)
        plt.ylabel('τ(q)', fontsize=12)
        plt.title('Multifractal Spectrum (Logarithmic q-axis)', fontsize=14)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return "Multifractal spectrum plot with logarithmic q-axis displayed."

    # Check `inputs.json` for predefined responses
    if os.path.exists("inputs.json"):
        try:
            with open("inputs.json", "r") as f:
                inputs_data = json.load(f)

                # Validate that inputs_data is a list
                if not isinstance(inputs_data, list):
                    print(f"{Fore.YELLOW}Warning: Malformed 'inputs.json'. Resetting to an empty list.")
                    inputs_data = []

                # Normalize user input
                normalized_input = user_input.strip().lower()

                # Iterate through the list of dictionaries
                for entry in inputs_data:
                    if isinstance(entry, dict) and "input" in entry and "response" in entry:
                        # Normalize the stored input for comparison
                        stored_input = entry["input"].strip().lower()
                        if normalized_input == stored_input:
                            # Fetch the response and print it in bold bright green font
                            response = entry["response"].strip()
                            print(f"{Style.BRIGHT}{Fore.GREEN}Response:")
                            for word in response.split():
                                print(f"{Style.BRIGHT}{Fore.GREEN}{word}", end=" ", flush=True)
                                time.sleep(0.01)
                            print("\n")
                            return response  # Return the response directly

        except json.JSONDecodeError:
            print(f"{Fore.RED}Error: 'inputs.json' is not a valid JSON file.")
        except Exception as e:
            print(f"{Fore.RED}Error reading 'inputs.json': {e}")

    # Extract subject words from user input
    words = user_input.split()
    stop_words = nlp_processor.stop_words
    non_stop_words = [word for word in words if word.lower() not in stop_words]

    # Optimize Wikipedia fetching with asyncio
    all_sentences = []
    try:
        if os.path.exists("inputs.json"):
            with open("inputs.json") as f:
                kb = json.load(f)
                subject_entries = [e for e in kb if e.get("input", "").lower() == subject_word.lower()]
                if subject_entries:
                    all_sentences = subject_entries[0].get("response", "").split(". ")
    except Exception as e:
        pass
    try:
        all_sentences = asyncio.run(fetch_all_sentences_async(non_stop_words))
    except RuntimeError as e:
        print(f"Error with asyncio runtime: {e}")

    # Fallback to enhanced_response_generation if needed
    if len(all_sentences) < len(words) and non_stop_words:
        fallback_response = er.enhanced_response_generation(non_stop_words)
        if fallback_response:
            sentences = re.split(r'(?<=[.!?]) +', fallback_response)
            all_sentences.extend(sentences)

    # Filter and score sentences
    scored_sentences = []
    for sentence in all_sentences:
        subject_word_count = sum(1 for word in words if word.lower() in sentence.lower())
        if subject_word_count < params.relevance_threshold:
            continue
        sentence = sentence.strip()
        if not sentence or sentence.endswith(":"):
            continue
        words_in_sentence = sentence.split()
        half_index = len(words_in_sentence) // 2
        first_half = " ".join(words_in_sentence[:half_index])
        starts_with_subject = any(sentence.lower().startswith(word.lower()) for word in words)
        subject_in_first_half = any(word.lower() in first_half.lower() for word in words)
        score = subject_word_count + (2 if starts_with_subject else 0) + (1 if subject_in_first_half else 0)
        scored_sentences.append((score, sentence))

    # Sort sentences by score and select top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    top_sentences = [sentence for _, sentence in scored_sentences[:len(words)]]

    # Handle empty responses
    if not top_sentences:
        return "I couldn't find relevant information."

    # Build the final response
    combined_response = " ".join(top_sentences)

    # Typewriter effect for response
    print(f"\n{Style.BRIGHT}{Fore.GREEN}Response:")
    for word in combined_response.split():
        print(f"{Style.BRIGHT}{Fore.GREEN}{word}", end=" ", flush=True)
        time.sleep(0.01)
    print("\n")

    return combined_response
def start_user_mode():
    """Starts the main interactive user mode with enhanced feedback and response logic."""
    nlp_processor = omni.NeuroLinguisticProcessor()
    neuromorph = NeuroMorphicProcessor()  # Initialize neuromorph here
    resources = [
        ("https://en.wikipedia.org/api/rest_v1/page/summary/{}", omni.OmniSourceResponseEngine._parse_wikipedia),
        ("https://api.dictionaryapi.dev/api/v2/entries/en/{}", omni.OmniSourceResponseEngine._parse_dictionary)
    ]

    # Create required files if missing
    for file in ["inputs.json", "feedback.json", "conversation.log"]:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                json.dump([], f, indent=4)  # Create an empty JSON array
            print(f"Created missing file: {file}")

    print(f"{Fore.RED}========={Fore.WHITE} A L E X A N D R I A N {Fore.RED}=========")
    print(f"{Fore.GREEN} ------< {Fore.WHITE}I N T E L L I G E N C E{Fore.GREEN} >------")
    print(f"{Fore.GREEN}             >>>{Fore.BLUE}i6 v2.9.5{Fore.GREEN}<<<        ")
    print("\nType your query below to begin. Type 'help' for commands and 'exit' to quit.\n")

    while True:
        try:
            # Display the hexagram grid
            print("\nHexagram Mind Simulation:\n")
            hex_grid = HexagramGrid()
            hex_grid.display()

            # Prompt user for input
            raw_input = input(f"\n{Fore.RED}[{Fore.WHITE}<{Fore.RED}]: {Fore.GREEN}")
            user_input = re.sub(r'[?!.,:]', '', raw_input)  # Clean input

            # Handle the "exit" command
            if user_input.lower() == "exit":
                print(f"\n{Fore.CYAN}Goodbye!")
                break

            if not user_input:
                print(f"{Fore.YELLOW}Input cannot be empty. Please try again.")
                continue

            # Generate the response
            response = generate_response(user_input, nlp_processor, neuromorph, hex_grid, resources)

            # Immediately print the response for "help" or "commands"
            if user_input.lower() in ["help", "commands"]:
                print(response)
                continue  # Skip further processing

            # Print the primary response
            #print(f"\n{Style.BRIGHT}{Fore.GREEN}Response: {response}\n")
	
            # Generate premonitions
            future_words = process_conversation_log()

            # Ask user if they are satisfied with the response
            while True:
                satisfied = input(f"{Fore.YELLOW}Are you satisfied with this response? (y/n): ").strip().lower()
                if satisfied in ['y', 'n']:
                    break
                print(f"{Fore.RED}Invalid input. Please enter 'y' for yes or 'n' for no.")

            if satisfied == 'y':
                neuromorph.save_feedback(user_input, response, feedback="satisfied", enhancement="satisfied_response")
                print(f"{Fore.CYAN}Thank you for your feedback!")
            else:
                enhancement = input(f"{Fore.CYAN}What enhancements would you like? Please be specific: ").strip()
                if enhancement:
                    if re.match(r"^(respond|answer):?", enhancement, re.IGNORECASE):
                        user_response = re.sub(r"^(respond|answer):?\s*", "", enhancement, flags=re.IGNORECASE)
                        NeuroMorphicProcessor.save_to_inputs_json(user_input, user_response.strip())
                        print(f"{Fore.GREEN}Your response has been saved and will be prioritized in future responses.")
                    else:
                        neuromorph.save_feedback(user_input, response, feedback="Not satisfied", enhancement=enhancement)
                        print(f"{Fore.GREEN}Your feedback has been recorded and will be used to improve future responses.")
                else:
                    print(f"{Fore.YELLOW}No enhancements provided. We value your feedback!")

            # Log input and response for further processing
            with open('conversation.log', 'a') as f:
                f.write(f"[<]: {user_input}\n")
                f.write(f"[>]: {response}\n")

        except KeyboardInterrupt:
            print("\nSession Ended.")
            break

def adjust_parameters_based_on_input(user_input, hex_grid, neuromorph):
    """Adjusts parameters based on trigger words in user input."""
    params = hex_grid.response_params

    if "more" in user_input or "longer" in user_input:
        params.primary_sentences += 1
        params.secondary_sentences += 1

    if "random" in user_input or "randomness" in user_input:
        params.randomization = min(100, params.randomization + 20)

    if "less" in user_input or "shorter" in user_input:
        params.primary_sentences = max(1, params.primary_sentences - 1)
        params.secondary_sentences = max(1, params.secondary_sentences - 1)



    # Check letter similarity with feedback.json entries
    feedback_analyzer = FeedbackAnalyzer()
    most_similar_entry, similarity = feedback_analyzer.find_most_similar(user_input)
    if similarity > 0.3333:
        params.primary_sentences += 1

    # Add logic for multifractal spectrum and premonition words
    process_conversation_log()
    spectrum = neuromorph.calculate_multifractal_spectrum()
    if params.randomization > 50:
        print(f"{Fore.CYAN}Randomizing sentence selection based on hexagram grid.")
        # Use hexagram grid to influence random selections


def dynamic_adjustment(hex_grid):
    """Adjusts parameters spontaneously based on the hexagram grid."""
    params = hex_grid.response_params
    green_count = sum(1 for row in hex_grid.rows for h in row if h["color"] == "green")
    red_count = sum(1 for row in hex_grid.rows for h in row if h["color"] == "red")

    if green_count > red_count:
        params.primary_sentences += 1
        params.randomization = min(100, params.randomization + 10)
    else:
        params.secondary_sentences = max(0, params.secondary_sentences - 1)
        params.randomization = max(0, params.randomization - 10)
if __name__ == "__main__":
    start_user_mode()
