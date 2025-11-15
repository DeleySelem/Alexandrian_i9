import multilingus3 as ml3
import random
import difflib
import json
import os
import re
import time
from collections import defaultdict, Counter
import shp295

class EnhancedLanguageGenerator(ml3.LanguageGenerator):
    def __init__(self):
        super().__init__()
        self.corrections = defaultdict(list)
        self.last_original = None
        self.positive_tokens = set()
        self.negative_tokens = set()
        self.conversation_log = []
        self.token_analysis = defaultdict(lambda: defaultdict(int))
        self.BLUE_BOLD = "\033[1;34m"
        self.BLUE = "\033[34m"
        self.GREEN = "\033[32m"
        self.RED = "\033[31m"
        self.YELLOW = "\033[33m"
        self.RESET = "\033[0m"
        self.MACHINE_PROMPT = "\033[1;31m"
        self.MACHINE_RESPONSE = "\033[1;32m"
        self.question_struct = {
            'words': set(),
            'phrases': set(),
            'patterns': set()
        }
        self.load_resources()
        # Initialize shp295 components
        self.shp_nlp_processor = shp295.NeuroLinguisticProcessor()
        self.shp_neuromorph = shp295.NeuroMorphicProcessor()                            
        self.shp_hex_grid = shp295.HexagramGrid()
        self.shp_resources = [
            "words.json",
            "phrases.json",
            "mind.json",
            "parameters.json",
            "minus.json",
            "plus.json"
        ]

    def load_resources(self):
        """Load all resources from files"""
        try:
            # Load corrections
            if os.path.exists('corrections.json'):
                with open('corrections.json', 'r') as f:
                    data = json.load(f)
                    for orig, corrs in data.items():
                        self.corrections[orig] = [(corr, weight) for corr, weight in corrs]

            # Load positive tokens
            if os.path.exists('plus.json'):
                with open('plus.json', 'r') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.positive_tokens = set(data)
                        else:
                            print(f"{self.MACHINE_RESPONSE}Warning: plus.json has invalid format{self.RESET}")
                    except json.JSONDecodeError:
                        print(f"{self.MACHINE_RESPONSE}Error decoding plus.json{self.RESET}")

            # Load negative tokens
            if os.path.exists('minus.json'):
                with open('minus.json', 'r') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.negative_tokens = set(data)
                        else:
                            print(f"{self.MACHINE_RESPONSE}Warning: minus.json has invalid format{self.RESET}")
                    except json.JSONDecodeError:
                        print(f"{self.MACHINE_RESPONSE}Error decoding minus.json{self.RESET}")

            # Load conversation log
            if os.path.exists('conversation.log'):
                with open('conversation.log', 'r') as f:
                    self.conversation_log = [line.strip() for line in f.readlines()]
                    self.analyze_conversation()

            # Load question structures
            if os.path.exists('qstructdetect.json'):
                with open('qstructdetect.json', 'r') as f:
                    data = json.load(f)
                    self.question_struct = {
                        'words': set(data.get('words', [])),
                        'phrases': set(data.get('phrases', [])),
                        'patterns': set(data.get('patterns', []))
                    }

        except Exception as e:
            print(f"{self.MACHINE_RESPONSE}Error loading resources: {e}{self.RESET}")

    def save_resources(self):
        """Save all resources to files"""
        try:
            # Save corrections
            with open('corrections.json', 'w') as f:
                data = {orig: corrs for orig, corrs in self.corrections.items()}
                json.dump(data, f, indent=2)

            # Save positive tokens
            with open('plus.json', 'w') as f:
                json.dump(list(self.positive_tokens), f, indent=2)

            # Save negative tokens
            with open('minus.json', 'w') as f:
                json.dump(list(self.negative_tokens), f, indent=2)

            # Save conversation log
            with open('conversation.log', 'a') as f:
                for entry in self.conversation_log:
                    f.write(entry + "\n")

            # Save question structures
            with open('qstructdetect.json', 'w') as f:
                json.dump({
                    'words': list(self.question_struct['words']),
                    'phrases': list(self.question_struct['phrases']),
                    'patterns': list(self.question_struct['patterns'])
                }, f, indent=2)

        except Exception as e:
            print(f"{self.MACHINE_RESPONSE}Error saving resources: {e}{self.RESET}")

    def analyze_conversation(self):
        """Perform multifractal analysis on conversation log"""
        if not self.conversation_log:
            return

        text = " ".join(self.conversation_log).lower()

        # Analyze at different linguistic levels
        levels = {
            'letters': r'[a-z]',
            'pairs': r'[a-z]{2}',
            'triplets': r'[a-z]{3}',
            'words': r'\b\w+\b',
            'sentences': r'[^.!?]+'
        }

        for level, pattern in levels.items():
            tokens = re.findall(pattern, text)
            self.token_analysis[level] = Counter(tokens)

    def get_word_sentiment(self, word):
        """Calculate sentiment score for a word (-1 to 1)"""
        word_lower = word.lower()

        # Check exact matches
        if word_lower in self.positive_tokens:
            return 1.0
        if word_lower in self.negative_tokens:
            return -1.0

        # Check token-based sentiment
        pos_score, neg_score = 0, 0
        total_tokens = 0

        # Analyze tokens of different sizes
        for n in range(1, len(word_lower)+1):
            for i in range(len(word_lower) - n + 1):
                token = word_lower[i:i+n]
                total_tokens += 1
                if token in self.positive_tokens:
                    pos_score += 1
                if token in self.negative_tokens:
                    neg_score += 1

        # Calculate overall sentiment
        if total_tokens == 0:
            return 0.0
        return (pos_score - neg_score) / total_tokens

    def typewrite_text(self, text, delay=0.01):
        """Print text with typewriter effect and sentiment coloring"""
        if not text:
            return

        # Ensure proper sentence formatting
        text = text.strip()
        if not text[0].isupper():
            text = text[0].upper() + text[1:]
        if not any(text.endswith(p) for p in ['.', '!', '?']):
            text += '.'

        words = text.split()

        # Print with typewriter effect
        print(f"{self.MACHINE_PROMPT}> ", end='', flush=True)
        for i, word in enumerate(words):
            # Determine word color based on sentiment
            sentiment = self.get_word_sentiment(word)
            if sentiment > 0.3:
                color_code = self.GREEN
            elif sentiment < -0.3:
                color_code = self.RED
            else:
                color_code = self.YELLOW

            # Print word with delay
            end_char = ' ' if i < len(words)-1 else '\n'
            print(f"{color_code}{word}{self.RESET}", end=end_char, flush=True)
            time.sleep(delay)

    def update_sentiment_resources(self, text, is_positive):
        """Update sentiment resources based on user input"""
        words = re.findall(r'\b\w+\b', text.lower())

        for word in words:
            # Add tokens of different sizes
            for n in range(1, len(word)+1):
                for i in range(len(word) - n + 1):
                    token = word[i:i+n]
                    if is_positive:
                        self.positive_tokens.add(token)
                    else:
                        self.negative_tokens.add(token)

        # Save updated resources
        self.save_resources()

    def print_heatmap(self, user_input, responses):
        """Print word heatmap for verbosity level 99"""
        # Collect all words
        all_words = user_input.split()
        for response in responses:
            all_words.extend(response.split())

        # Calculate frequencies
        word_freq = Counter(word.lower() for word in all_words)
        if not word_freq:
            return

        max_freq = max(word_freq.values())

        # Print header
        print(f"{self.BLUE_BOLD}Subconscious Processes: {self.RESET}", end='', flush=True)

        # Print words with color gradient
        for word, freq in word_freq.most_common():
            norm_freq = freq / max_freq

            # Color gradient: dark blue (0) -> blue (0.33) -> light blue (0.66) -> white (1.0)
            if norm_freq < 0.33:
                blue_val = 128 + int(127 * (norm_freq * 3))
                color_code = f"\033[38;2;0;0;{blue_val}m"
            elif norm_freq < 0.66:
                intensity = int(128 * (norm_freq - 0.33) * 3)
                color_code = f"\033[38;2;{intensity};{intensity};255m"
            else:
                intensity = 128 + int(127 * (norm_freq - 0.66) * 3)
                color_code = f"\033[38;2;{intensity};{intensity};255m"

            print(f"{color_code}{word}{self.RESET} ", end='', flush=True)
            time.sleep(0.05)
        print()

    def update_question_structure(self, input_str):
        """Update question structure detection from input"""
        # Clean and tokenize input
        clean_input = re.sub(r'[^\w\s]', '', input_str.lower())
        words = clean_input.split()

        # Add individual words
        self.question_struct['words'].update(words)

        # Add phrases (2-5 word n-grams)
        for n in range(2, 6):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                self.question_struct['phrases'].add(phrase)

        # Add the entire input as a pattern
        self.question_struct['patterns'].add(clean_input)

        # Save updated structure
        self.save_resources()

    def detect_and_reformat_question(self, response):
        """Detect question structures and reformat response"""
        response_lower = response.lower()

        # Check for exact question patterns first (longest patterns first)
        for pattern in sorted(self.question_struct['patterns'], key=len, reverse=True):
            if pattern and pattern in response_lower:
                # Split response at pattern location
                pattern_idx = response_lower.find(pattern)
                if pattern_idx >= 0:
                    before = response[:pattern_idx].strip()
                    question_phrase = response[pattern_idx:pattern_idx+len(pattern)].strip()

                    # Format with dot before and question mark after
                    if before:
                        return f"{before}. {question_phrase}?"
                    return f"{question_phrase}?"

        # Check for question phrases
        for phrase in sorted(self.question_struct['phrases'], key=len, reverse=True):
            if phrase and phrase in response_lower:
                # Split response at phrase location
                phrase_idx = response_lower.find(phrase)
                if phrase_idx >= 0:
                    before = response[:phrase_idx].strip()
                    question_phrase = response[phrase_idx:phrase_idx+len(phrase)].strip()

                    # Format with dot before and question mark after
                    if before:
                        return f"{before}. {question_phrase}?"
                    return f"{question_phrase}?"

        # No question pattern found
        return response

    def correct_word_in_files(self, old_word, new_word):
        """Replace word in all JSON files"""
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]

        for filename in json_files:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)

                # Recursive replacement in JSON structure
                def replace_in_structure(obj):
                    if isinstance(obj, dict):
                        return {replace_in_structure(k): replace_in_structure(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [replace_in_structure(item) for item in obj]
                    elif isinstance(obj, str):
                        return obj.replace(old_word, new_word)
                    return obj

                updated_data = replace_in_structure(data)

                with open(filename, 'w') as f:
                    json.dump(updated_data, f, indent=2)

                print(f"{self.GREEN}Updated {filename}{self.RESET}")

            except Exception as e:
                print(f"{self.RED}Error processing {filename}: {e}{self.RESET}")

    def process_input(self, user_input):
        # Generate shp295 response in background
        shp_response = shp295.generate_response(
            user_input,
            self.shp_nlp_processor,
            self.shp_neuromorph,
            self.shp_hex_grid,
            self.shp_resources
        )

        # Combine shp295 response with user input
        combined_input = f"{shp_response} {user_input}"

        # Handle "correct" command
        if user_input.startswith('correct "') and '" to "' in user_input:
            parts = user_input.split('"')
            if len(parts) >= 5:
                old_word = parts[1]
                new_word = parts[3]
                self.correct_word_in_files(old_word, new_word)
                response = f'Corrected "{old_word}" to "{new_word}" in all resources'
                self.conversation_log.append(f"System: {response}")
                print(f"{self.MACHINE_RESPONSE}{response}{self.RESET}")
                return

        # Process combined input through ML57
        self.universe.process_parameters()
        self.conversation_log.append(f"User: {combined_input}")

        # Handle question responses
        if user_input.endswith('?') or any(word in self.question_struct['words'] for word in user_input.lower().split()):
            # Update question structure if it was a clear question (ended with ?)
            if user_input.endswith('?'):
                clean_input = user_input.rstrip('?').strip()
                self.update_question_structure(clean_input)

            # Update linguistic analysis
            words = re.findall(r'\b\w+\b', user_input.lower())
            for word in words:
                # Update word frequencies
                self.token_analysis['words'][word] += 1
                self.token_analysis['question_words'][word] += 1

            # Automatically detect question words (language-agnostic)
            question_words = []
            if self.token_analysis['question_words']:
                total_questions = sum(self.token_analysis['question_words'].values())
                for word, count in self.token_analysis['question_words'].items():
                    # Calculate distinctiveness: frequency in questions vs general frequency
                    general_freq = self.token_analysis['words'].get(word, 1)
                    distinctiveness = count / (general_freq + 1)  # Avoid division by zero

                    # Detect words that appear primarily in questions
                    if distinctiveness > 0.7 and count > total_questions * 0.1:
                        question_words.append(word)

            # Identify content words from current question
            content_words = []
            for word in words:
                # Skip short words and high-frequency question words
                if len(word) > 2 and word not in question_words:
                    content_words.append(word)

            # Generate response based on detected patterns
            if content_words:
                # Select up to 2 content words to reference
                selected_words = content_words[:2]
                responses = [
                    f"Associating: {', '.join(selected_words)}",
                    f"Connecting question patterns: {', '.join(selected_words)}",
                    f"Detected concepts: {', '.join(selected_words)}",
                    f"?: association with: {', '.join(selected_words)}",
                    f"Pattern match: {', '.join(selected_words)}"
                ]
            elif question_words:
                # Reference detected question words
                responses = [
                    f"Recognized question structure: {question_words[0]}",
                    f"Associating question pattern: {question_words[0]}",
                    f"Detected question word: {question_words[0]}",
                    f"?: pattern: {question_words[0]}",
                    f"Question association: {question_words[0]}"
                ]
            else:
                # Generic response if no patterns detected
                responses = [
                    "Associating question structure.",
                    "Detected question pattern.",
                    "?: connecting concepts.",
                    "Recognizing question format.",
                    "Associating question markers."
                ]

            response = random.choice(responses)
            # Apply question reformatting if needed
            formatted_response = self.detect_and_reformat_question(response)
            self.conversation_log.append(f"System: {formatted_response}")
            if self.universe.parameters["verbosity"] == 99:
                self.typewrite_text(formatted_response)
            else:
                print(f"{self.MACHINE_RESPONSE}{formatted_response}{self.RESET}")
            self.auto_input_counter += 1
            return

        words = user_input.split()
        is_negation = False
        negation_words = ['No', 'no', 'Ei', 'ei', 'Nein', 'nein', 'Non', 'non']

        # Check for negation commands
        if words and words[0] in negation_words:
            is_negation = True
            correction_phrase = ' '.join(words[1:])

            if self.last_original:
                # Update existing or add new correction
                updated = False
                new_corrections = []
                total_weight = 0

                for corr, weight in self.corrections[self.last_original]:
                    if corr == correction_phrase:
                        new_weight = weight + 1
                        updated = True
                    else:
                        new_weight = weight
                    new_corrections.append((corr, new_weight))
                    total_weight += new_weight

                if not updated:
                    new_corrections.append((correction_phrase, 1))
                    total_weight += 1

                # Normalize weights to percentages
                self.corrections[self.last_original] = [
                    (corr, weight/total_weight)
                    for corr, weight in new_corrections
                ]

                # Build response
                response = "Correction stored."

                # Add correction details
                if self.universe.parameters["verbosity"] > 0:
                    response += f"\nOriginal: {self.last_original}"
                    response += f"\nCorrection: {correction_phrase}"

                # Update sentiment resources
                self.update_sentiment_resources(self.last_original, is_positive=False)
                self.update_sentiment_resources(correction_phrase, is_positive=True)

                # Print response
                self.conversation_log.append(f"System: {response}")
                self.typewrite_text(response)

                # Save and update
                self.save_resources()
                self.universe.update_with_sentence(user_input)
                self.auto_input_counter += 1
                return

        # Update language model
        if ' ' in user_input:
            self.universe.update_with_sentence(user_input)
        else:
            self.universe.update_with_word(user_input)

        # Store original phrase for potential correction
        if not is_negation:
            self.last_original = user_input

        # Generate responses
        responses = [self.universe.generate_sentence() for _ in range(37)]
        weights = [1.0] * 37

        # Apply corrections
        matching_corrections = []
        if not is_negation and self.corrections:
            for orig, corrs in self.corrections.items():
                char_sim = self.char_sequence_similarity(user_input, orig)
                pair_sim = self.char_pair_similarity(user_input, orig)
                triplet_sim = self.char_triplet_similarity(user_input, orig)
                word_sim = self.word_sequence_similarity(user_input, orig)
                phrase_sim = self.phrase_similarity(user_input, orig)

                combined_sim = (
                    0.15 * char_sim +
                    0.20 * pair_sim +
                    0.25 * triplet_sim +
                    0.20 * word_sim +
                    0.20 * phrase_sim
                )

                if combined_sim > 0.4:
                    for corr, weight in corrs:
                        candidate_weight = combined_sim * weight * 1000
                        matching_corrections.append((corr, candidate_weight))

                        # Verbose output
                        if self.universe.parameters["verbosity"] > 20 and self.universe.parameters["verbosity"] != 99:
                            details = (
                                f"Match: '{orig}' → '{corr}' "
                                f"(char:{char_sim:.2f} pair:{pair_sim:.2f} "
                                f"triplet:{triplet_sim:.2f} word:{word_sim:.2f} "
                                f"phrase:{phrase_sim:.2f} combined:{combined_sim:.2f} "
                                f"weight:{candidate_weight:.1f})"
                            )
                            print(f"{self.BLUE_BOLD}Subconscious Processes: {self.BLUE}{details}{self.RESET}")

        for corr, weight in matching_corrections:
            responses.append(corr)
            weights.append(weight)

        # Select response
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w/total_weight for w in weights]
            selected_index = random.choices(range(len(responses)), weights=normalized_weights, k=1)[0]
        else:
            selected_index = random.randint(0, len(responses)-1)

        selected_response = responses[selected_index]

        # Apply question reformatting if needed
        formatted_response = self.detect_and_reformat_question(selected_response)

        # Handle verbosity level 99
        if self.universe.parameters["verbosity"] == 99:
            self.print_heatmap(user_input, responses)

        # Handle other verbosity levels
        elif self.universe.parameters["verbosity"] > 30:
            if selected_index >= len(responses) - len(matching_corrections):
                orig = next(orig for orig, corrs in self.corrections.items()
                           if any(corr == selected_response for corr, _ in corrs))
                verbose = f"Selected correction: '{orig}' → '{selected_response}'"
            else:
                verbose = f"Selected generated response: '{selected_response}'"

            print(f"{self.BLUE_BOLD}Subconscious Processes: {self.BLUE}{verbose}{self.RESET}")

        # Print final response with sentiment coloring for verbosity 99
        self.conversation_log.append(f"System: {formatted_response}")
        if self.universe.parameters["verbosity"] == 99:
            self.typewrite_text(formatted_response)
        else:
            print(f"{self.MACHINE_RESPONSE}{shp_response} {formatted_response}{self.RESET}")
        self.auto_input_counter += 1

        # Update analysis and save
        self.analyze_conversation()
        self.save_resources()

    # Similarity methods
    def char_sequence_similarity(self, s1, s2):
        return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def char_pair_similarity(self, s1, s2):
        pairs1 = set(s1[i:i+2] for i in range(len(s1)-1))
        pairs2 = set(s2[i:i+2] for i in range(len(s2)-1))
        return len(pairs1 & pairs2) / max(len(pairs1 | pairs2), 1)

    def char_triplet_similarity(self, s1, s2):
        triplets1 = set(s1[i:i+3] for i in range(len(s1)-2))
        triplets2 = set(s2[i:i+3] for i in range(len(s2)-2))
        return len(triplets1 & triplets2) / max(len(triplets1 | triplets2), 1)

    def word_sequence_similarity(self, s1, s2):
        words1 = s1.lower().split()
        words2 = s2.lower().split()
        return difflib.SequenceMatcher(None, words1, words2).ratio()

    def phrase_similarity(self, s1, s2):
        seq_match = difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        jaccard = len(words1 & words2) / len(words1 | words2) if words1 or words2 else 0.0
        return (seq_match * 0.7) + (jaccard * 0.3)

if __name__ == "__main__":
    generator = EnhancedLanguageGenerator()
    generator.start()