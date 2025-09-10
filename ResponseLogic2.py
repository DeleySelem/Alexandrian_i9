import requests
import json
import re


def count_subject_words(sentence, stop_words):
    """Counts subject words in a sentence, excluding stop words."""
    words = re.findall(r"\b\w+\b", sentence.lower())
    return len([word for word in words if word not in stop_words])

def clean_response(response, input_words):
    """Cleans up the response by removing unwanted characters and formatting."""
    stop_words = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "he", "him", "his", "she", "her", "it",
        "its", "they", "them", "their", "what", "which", "who", "this",
        "that", "these", "those", "am", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "do", "does", "did", "doing", "a",
        "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against",
        "between", "to", "from", "up", "down", "in", "out", "on", "off",
        "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
        "just", "don", "should", "now", ":", "explain", "I'm", "I", "think", "thinking"
    ])

    sentences = re.split(r"(?<=[.!?]) + ? ", response.strip())
    filtered_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        # Skip sentences that end with a colon or start with ". " or " "
        current_has_input = any(word.lower() in sentence.lower() for word in input_words)
        starts_with_continuation = re.match(r'^(It|This|That|Which|They|He|She|There)', sentence, re.I)
        prev_has_input = ""
        if not current_has_input:
            if prev_has_input and starts_with_continuation:
                filtered_sentences.append(sentence)
            continue
        
        prev_has_input = current_has_input
        if sentence.endswith(":") or sentence.startswith(". ") or sentence.startswith(" "):
            continue

        # Handle sentences with colons
        if ":" in sentence:
            parts = sentence.split(":", 1)
            if len(parts) == 2:
                before_colon, after_colon = parts[0].strip(), parts[1].strip()
                if count_subject_words(after_colon, stop_words) > count_subject_words(before_colon, stop_words):
                    filtered_sentences.append(f"{before_colon}:  {after_colon}")
            else:
                filtered_sentences.append(f"{before_colon}:")
            continue

        # Include only sentences that have at least two subject words if user input has more than one subject word
        subject_word_count = sum(1 for word in input_words if word.lower() in sentence.lower())
        if len(input_words) > 1 and subject_word_count >= 2:
            filtered_sentences.append(sentence)
        elif len(input_words) == 1 and subject_word_count >= 1:
            filtered_sentences.append(sentence)

    return " ".join(filtered_sentences)


def fetch_from_duckduckgo(input_words):
    """Fetches responses from DuckDuckGo API."""
    try:
        ddg_response = requests.get(
            f"https://api.duckduckgo.com/?q={'+'.join(input_words)}&format=json&no_html=1",
            timeout=5
        )
        if ddg_response.status_code == 200:
            data = ddg_response.json()
            if data.get("AbstractText"):
                response = clean_response(data["AbstractText"], input_words)
                print(f"Processing Query [DuckDuckGo]: {response}")
                return response
    except Exception as e:
        print(f"Error fetching from DuckDuckGo: {e}")
    return None


def fetch_from_dictionary_api(word):
    """Fetches word definitions from Dictionary API."""
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data:
                meaning = data[0]['meanings'][0]['definitions'][0]['definition']
                response = clean_response(f"{word} means: {meaning}", [word])
                #print(f"Processing Query:\n> {response}")
                return response
    except Exception as e:
        print(f"Error fetching from Dictionary API for {word}: {e}")
    return None


def fetch_from_additional_apis(input_words):
    """Fetches random content from additional APIs."""
    apis = [
        ("https://api.quotable.io/random", lambda data: f"{data['content']} — {data['author']}"),
        ("https://v2.jokeapi.dev/joke/Any?type=single", lambda data: data['joke']),
        ("https://api.adviceslip.com/advice", lambda data: data['slip']['advice']),
        ("http://numbersapi.com/random/trivia?json", lambda data: data['text']),
        ("https://poetrydb.org/random", lambda data: f"Poem: {data[0]['title']} by {data[0]['author']}\n" + "\n".join(data[0]['lines'])),
        ("https://opentdb.com/api.php?amount=1", lambda data: f"Trivia: {data['results'][0]['question']} Answer: {data['results'][0]['correct_answer']}"),
        ("http://www.boredapi.com/api/activity/", lambda data: f"Activity: {data['activity']}"),
        ("https://uselessfacts.jsph.pl/random.json?language=en", lambda data: data['text']),
        ("https://yesno.wtf/api", lambda data: f"Answer: {data['answer']}")
    ]
    for url, parser in apis:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                result = clean_response(parser(data), input_words)
                if result:
                    print(f"Processing Query:\n>>{result}")
                    return result
        except Exception as e:
            print(f"Error fetching from {url}: {e}")
    return None


def enhanced_response_generation(input_words):
    try:
        with open('inputs.json', 'r') as f:
            inputs_data = json.load(f)['input']
            best_match = None
            highest_score = 0
            
            for saved_input, data in inputs_data.items():
                saved_words = set(re.findall(r'\w+', saved_input.lower()))
                current_words = set(re.findall(r'\w+', ' '.join(input_words).lower()))
                similarity = len(saved_words & current_words) / len(saved_words | current_words)
                
                if similarity > 0.666 and similarity > highest_score:
                    best_match = data['meaning']
                    highest_score = similarity
            
            if best_match:
                return best_match
    except Exception as e:
        pass
    """Generates enhanced responses based on input words."""
    # Try DuckDuckGo API first
    primary_response = fetch_from_duckduckgo(input_words)
    if primary_response:
        return primary_response

    # Try Dictionary API for each word
    for word in input_words:
        definition = fetch_from_dictionary_api(word)
        if definition:
            return definition

    # Try additional APIs for random content
    fallback_response = fetch_from_additional_apis(input_words)
    if fallback_response:
        return fallback_response

    # Default response if all APIs fail
    return "I'm sorry, I couldn't find any relevant information. Please try rephrasing your query."


def generate_secondary_response(response, input_words):
    """Generates a secondary response based on the selected sentence."""
    sentences = re.split(r"(?<=[.!?]) +", response.strip())
    scored_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        # Skip empty sentences
        if not sentence:
            continue

        # Split the sentence into two halves
        words = sentence.split()
        half_index = len(words) // 3
        first_half = " ".join(words[:half_index])

        # Count subject words in the first half
        subject_word_count = sum(1 for word in input_words if word.lower() in first_half.lower())

        # Add sentence and its score if it contains subject words
        if subject_word_count > 0:
            scored_sentences.append((subject_word_count, sentence))

    # Sort sentences by score (descending)
    scored_sentences.sort(reverse=True, key=lambda x: x[0])

    # Return the highest-scoring sentence or an empty string if no valid sentence is found
    return scored_sentences[0][1] if scored_sentences[0] != scored_sentences[1] else ""


# Example Usage
if __name__ == "__main__":
    words = ["example", "test"]
    response = enhanced_response_generation(words)
    print(f"\nPrimary Response: {response}")

    secondary_response = generate_secondary_response(response, words)
    print(f"\nSecondary Response: {secondary_response}")
