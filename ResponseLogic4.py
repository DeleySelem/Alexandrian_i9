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
            "just", "don", "should", "now", ":", "explain", "define", "tell", 
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
            "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
            "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
            "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
            "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
            "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
            "it's", "its", "itself", "just", "let's", "me", "more", "most", "mustn't", "my",
            "myself", "no", "nor", "not", "nothing", "now", "of", "off", "on", "once",
            "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
            "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
            "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
            "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
            "they'll", "they're", "they've", "this", "those", "through", "to", "too",
            "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
            "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
            "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
            "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
            "your", "yours", "yourself", "yourselves", "yeah", "oh", "hey", "oh", "hi",
            "hello", "explain", "define", "tell", "none", "nobody", "nothing", "might",
            "shall", "ought", "I'm", "I", "think", "thinking", "the", "word"
    ])

    sentences = re.split(r"(?<=[.!?]) +", response.strip())
    filtered_sentences = []
    prev_has_input = False  # Initialize prev_has_input to avoid referencing before assignment

    for sentence in sentences:
        sentence = sentence.strip()

        # Skip sentences that end with a colon or start with ". " or " "
        current_has_input = any(word.lower() in sentence.lower() for word in input_words)
        starts_with_continuation = re.match(r'^(It|This|That|Which|They|He|She|There)', sentence, re.I)

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
                    filtered_sentences.append(f"{before_colon}: {after_colon}")
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


# The rest of the code remains unchanged
# (fetch_from_dictionary_api, fetch_from_additional_apis, enhanced_response_generation, etc.)
