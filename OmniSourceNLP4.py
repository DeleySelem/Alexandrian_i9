import requests
import re
import random
import json
from collections import defaultdict, OrderedDict
from urllib.parse import quote
from bs4 import BeautifulSoup
import ResponseLogic4 as er
import difflib


        
class NeuroLinguisticProcessor:
    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.stem_cache = {}
        self.pos_rules = self._load_pos_rules()

    def _load_stop_words(self):
        """Loads a set of stop words to filter out from extracted concepts."""
        return set([
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
            "just", "don", "should", "now", ":", "explain", "define", "I'm", "thinking"
        ])

    def _load_pos_rules(self):
        """Defines basic POS tagging rules based on word suffixes."""
        return {
            "noun_suffixes": ["tion", "ment", "ness", "ity", "ance", "ence", "hood"],
            "verb_suffixes": ["ize", "ate", "ify", "ing", "ed", "en"],
            "adj_suffixes": ["able", "ible", "al", "ant", "ent", "ic", "ical", "ous"],
            "adv_suffixes": ["ly", "ward", "wise"]
        }

    def _stem(self, word):
        """Applies simple stemming rules to reduce words to their base form."""
        if word not in self.stem_cache:
            self.stem_cache[word] = re.sub(r"(ss|i)es$|ed$", "", word.lower())
        return self.stem_cache[word]

    def extract_concepts(self, text):
        """Extracts key concepts from the given text."""
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
        filtered = [t for t in tokens if t not in self.stop_words]
        return list(OrderedDict.fromkeys(filtered))  # Removes duplicates


class OmniSourceResponseEngine:
    def __init__(self):
        self.nlp = NeuroLinguisticProcessor()
        self.resources = [
            ("https://en.wikipedia.org/api/rest_v1/page/summary/{}", self._parse_wikipedia),
            ("https://api.dictionaryapi.dev/api/v2/entries/en/{}", self._parse_dictionary)
        ]
        # Apply multifractal weighting
        if self.multifractal_params:
            content = sorted(all_content, 
                key=lambda x: len(set(re.findall(r'\w+', x)) & set(self.multifractal_params['significant_words']), 
                reverse=True))
        
        return er.clean_response(" ".join(all_content), concepts)
        
    def generate_response(self, query):
        try:
            with open('inputs.json', 'r') as f:
                inputs_data = json.load(f)['input']
                current_query = ' '.join(query) if isinstance(query, list) else query
                best_match = None
                highest_sim = 0
                
                for saved_input in inputs_data:
                    seq = difflib.SequenceMatcher(None, current_query.lower(), saved_input.lower())
                    if seq.ratio() > 0.666 and seq.ratio() > highest_sim:
                        best_match = inputs_data[saved_input]['meaning']
                        highest_sim = seq.ratio()
                
                if best_match:
                    return best_match
        except Exception as e:
            print(f"Error checking input matches: {e}")
        """Generates a response by fetching and processing data from multiple sources."""
        if isinstance(query, list):
            query = " ".join(query)  # Join list elements into a single string

        concepts = self.nlp.extract_concepts(query)
        all_content = []

        for base_url, handler in self.resources:
            for concept in concepts:
                try:
                    url = base_url.format(quote(concept))
                    response = requests.get(url, timeout=3)
                    if response.status_code == 200:
                        all_content.extend(handler(response))
                except Exception as e:
                    print(f"Error fetching data for {concept}: {e}")

        if not all_content:
            return "I couldn't find relevant information for your query."

        return er.clean_response(" ".join(all_content), concepts)

    def _parse_wikipedia(self, response):
        """Parses content from Wikipedia API."""
        try:
            data = response.json()
            return [data.get("extract", "")[:500] + "."]
        except Exception as e:
            print(f"Error parsing Wikipedia response: {e}")
            return []

    def _parse_dictionary(self, response):
        """Parses content from Dictionary API."""
        try:
            data = response.json()
            return [f"Definition: {data[0]['meanings'][0]['definitions'][0]['definition']}"]
        except Exception as e:
            print(f"Error parsing Dictionary response: {e}")
            return []

        # Apply multifractal weighting
        if self.multifractal_params:
            content = sorted(all_content, 
                key=lambda x: len(set(re.findall(r'\w+', x)) & set(self.multifractal_params['significant_words']), 
                reverse=True))
        
        return er.clean_response(" ".join(all_content), concepts)
def enhanced_response_generation(user_input):
    engine = OmniSourceResponseEngine()
    return engine.generate_response(user_input)


# Example Usage
if __name__ == "__main__":
    query = "explain genome integration and viruses"
    print(enhanced_response_generation(query))
