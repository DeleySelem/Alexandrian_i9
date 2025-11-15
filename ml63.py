import multilingus3 as ml3
import random
import difflib
import json
import os
import re
import time
from collections import defaultdict, Counter

class EnhancedLanguageGenerator(ml3.LanguageGenerator):
    def __init__(self):
        super().__init__()
        # Structural learning properties
        self.structures = {}
        self.variations = defaultdict(dict)
        self.connections = defaultdict(lambda: defaultdict(float))
        self.token_to_index = {}
        self.index_to_token = {}
        self.next_index = 1
        self.next_structure_id = 1
        self.word_token_cache = {}

        # Existing properties
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
        self.load_resources()
        self.load_structure_resources()

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

        except Exception as e:
            print(f"{self.MACHINE_RESPONSE}Error loading resources: {e}{self.RESET}")

    def load_structure_resources(self):
        """Load structural resources from files"""
        if os.path.exists('words.json'):
            with open('words.json', 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.token_to_index = data.get("token_to_index", {})
                        self.next_index = data.get("next_index", 1)
                        self.index_to_token = {v: k for k, v in self.token_to_index.items()}
                    else:
                        print("Warning: words.json has invalid format")
                except json.JSONDecodeError:
                    print("Error decoding words.json")

        if os.path.exists('struct.json'):
            with open('struct.json', 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.structures = data.get("structures", {})
                        self.variations = defaultdict(dict, data.get("variations", {}))
                        self.connections = defaultdict(lambda: defaultdict(float),
                                                      {k: dict(v) for k, v in data.get("connections", {}).items()})
                        self.next_structure_id = data.get("next_structure_id", 1)
                    else:
                        print("Warning: struct.json has invalid format")
                except json.JSONDecodeError:
                    print("Error decoding struct.json")

        # Load token cache
        if os.path.exists('token_cache.json'):
            with open('token_cache.json', 'r') as f:
                try:
                    self.word_token_cache = json.load(f)
                except:
                    self.word_token_cache = {}

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

            # Save structural resources
            self.save_structure_resources()

        except Exception as e:
            print(f"{self.MACHINE_RESPONSE}Error saving resources: {e}{self.RESET}")

    def save_structure_resources(self):
        """Save structural resources to files"""
        with open('words.json', 'w') as f:
            json.dump({
                "token_to_index": self.token_to_index,
                "next_index": self.next_index
            }, f, indent=2)

        with open('struct.json', 'w') as f:
            json.dump({
                "structures": self.structures,
                "variations": dict(self.variations),
                "connections": {k: dict(v) for k, v in self.connections.items()},
                "next_structure_id": self.next_structure_id
            }, f, indent=2)

        # Save token cache
        with open('token_cache.json', 'w') as f:
            json.dump(self.word_token_cache, f)

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
        token_set = self.get_token_set(word_lower)
        total_tokens = len(token_set)
        for token in token_set:
            if token in self.positive_tokens:
                pos_score += 1
            if token in self.negative_tokens:
                neg_score += 1

        # Calculate overall sentiment
        if total_tokens == 0:
            return 0.0
        return (pos_score - neg_score) / total_tokens

    def typewrite_text(self, text, delay=0.05):
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
        words = re.findall(r'\b\w+\b', text.lower()) #convert to lowercase

        for word in words:
            # Add tokens of different sizes
            token_set = self.get_token_set(word)
            for token in token_set:
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

    def get_token_set(self, word):
        """Get token set for a word (cached)"""
        if word in self.word_token_cache:
            return set(self.word_token_cache[word])

        tokens = set()
        n_min = 1
        n_max = min(3, len(word))  # Limit n-gram size to 3
        for n in range(n_min, n_max+1):
            for i in range(len(word) - n + 1):
                tokens.add(word[i:i+n])

        self.word_token_cache[word] = list(tokens)
        return tokens

    def token_similarity(self, word1, word2):
        """Calculate token-based similarity between two words"""
        if word1 == word2:
            return 1.0

        tokens1 = self.get_token_set(word1)
        tokens2 = self.get_token_set(word2)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def get_word_index(self, word):
        """Get or create index for a word"""
        if word not in self.token_to_index:
            self.token_to_index[word] = self.next_index
            self.index_to_token[self.next_index] = word
            self.next_index += 1
        return self.token_to_index[word]

    def add_variation(self, word_index, variation):
        """Add a variation for a word"""
        if word_index not in self.variations:
            self.variations[word_index] = {}

        # Find existing variation index or create new
        variation_index = None
        for idx, word in self.variations[word_index].items():
            if word == variation:
                variation_index = idx
                break

        if variation_index is None:
            variation_index = len(self.variations[word_index]) + 1
            self.variations[word_index][variation_index] = variation

        return variation_index

    def analyze_sentence_structure(self, sentence):
        """Analyze and store sentence structure using token-based matching"""
        words = sentence.split()
        structure = []

        # Convert words to index + variation format
        for word in words:
            word_index = self.get_word_index(word)
            variation_index = self.add_variation(word_index, word)
            structure.append((word_index, variation_index))

        # Store structure
        structure_id = f"s{self.next_structure_id}"
        self.structures[structure_id] = structure
        self.next_structure_id += 1

        return structure_id

    def find_similar_structure(self, current_structure_id):
        """Find structurally similar sentences using token similarity"""
        current_structure = self.structures[current_structure_id]
        current_words = [self.index_to_token[word_idx] for word_idx, _ in current_structure]
        matches = []

        for sid, structure in self.structures.items():
            if sid == current_structure_id:
                continue

            if len(structure) != len(current_structure):
                continue

            # Check token similarity for each word
            match = True
            for i in range(len(structure)):
                word_idx, _ = structure[i]
                ref_word = self.index_to_token[word_idx]
                curr_word = current_words[i]

                if self.token_similarity(ref_word, curr_word) < 0.5:
                    match = False
                    break

            if match:
                matches.append(sid)

        return matches

    def generate_structural_response(self, structure_id):
        """Generate response based on structural patterns"""
        if structure_id not in self.structures:
            return None

        structure = self.structures[structure_id]

        # Convert structure to words
        words = []
        for word_index, variation_index in structure:
            if word_index in self.variations and variation_index in self.variations[word_index]:
                words.append(self.variations[word_index][variation_index])
            elif word_index in self.index_to_token:
                words.append(self.index_to_token[word_index])
            else:
                words.append("?")

        return " ".join(words)

    def connect_structures(self, source_id, target_id):
        """Connect two sentence structures with weighted connection"""
        if source_id not in self.connections:
            self.connections[source_id] = defaultdict(float)

        self.connections[source_id][target_id] += 1.0  # Increase connection weight

    def find_dynamic_patterns(self, structure_id):
        """Enhanced pattern detection with mathematical operation awareness"""
        current_structure = self.structures[structure_id]
        current_words = []

        # Safely build current_words with fallback
        for word_idx, _ in current_structure:
            try:
                current_words.append(self.index_to_token[word_idx])
            except KeyError:
                if word_idx in self.variations and self.variations[word_idx]:
                    # Use first variation if available
                    first_var = next(iter(self.variations[word_idx].values()))
                    current_words.append(first_var)
                else:
                    current_words.append(f"?{word_idx}?")

        dynamic_patterns = []

        for sid, structure in self.structures.items():
            if sid == structure_id or len(structure) != len(current_structure):
                continue

            ref_words = []
            for word_idx, _ in structure:
                try:
                    ref_words.append(self.index_to_token[word_idx])
                except KeyError:
                    if word_idx in self.variations and self.variations[word_idx]:
                        first_var = next(iter(self.variations[word_idx].values()))
                        ref_words.append(first_var)
                    else:
                        ref_words.append(f"?{word_idx}?")

            changing_positions = []
            operator_positions = []
            verb_positions = []

            # Identify changing words, operators, and verbs
            for i in range(len(structure)):
                if self.token_similarity(ref_words[i], current_words[i]) < 0.5:
                    changing_positions.append(i)
                elif ref_words[i] in {'plus', 'minus', 'times', 'divided'}:
                    operator_positions.append(i)
                elif ref_words[i] in {'is', 'are', 'equals', '='}:
                    verb_positions.append(i)

            # Calculate pattern strength
            pattern_strength = len(structure) - len(changing_positions)

            # Add extra weight for operator and verb positions
            pattern_strength += len(operator_positions) * 1.5
            pattern_strength += len(verb_positions) * 1.2

            if changing_positions:
                dynamic_patterns.append((sid, changing_positions, operator_positions, verb_positions, pattern_strength))

        # Sort by pattern strength (strongest first)
        dynamic_patterns.sort(key=lambda x: x[4], reverse=True)
        return dynamic_patterns

    def apply_dynamic_pattern(self, current_structure_id, pattern_structure_id, changing_positions, operator_positions, verb_positions):
        """Generate response with enhanced mathematical structure awareness"""
        current_structure = self.structures[current_structure_id]
        pattern_structure = self.structures[pattern_structure_id]
        response_structure = []

        # Apply pattern while preserving changing words
        for i in range(len(current_structure)):
            if i in changing_positions:
                # Use word from current structure
                response_structure.append(current_structure[i])
            else:
                # Use word from pattern structure
                response_structure.append(pattern_structure[i])

        # Convert structure to words
        words = []
        for word_index, variation_index in response_structure:
            if word_index in self.variations and variation_index in self.variations[word_index]:
                words.append(self.variations[word_index][variation_index])
            elif word_index in self.index_to_token:
                words.append(self.index_to_token[word_index])
            else:
                words.append("?")

        # Enhanced mathematical reordering
        if operator_positions and verb_positions:
            # Check for mathematical question pattern
            if (len(words) >= 5 and
                any(w in {'is', 'are'} for w in words) and
                any(op in words for op in {'plus', 'minus', 'times'}) and
                words[-1].endswith('?')):

                # Find key positions
                verb_index = next((i for i, w in enumerate(words) if w in {'is', 'are'}), -1)
                op_index = next((i for i, w in enumerate(words) if w in {'plus', 'minus', 'times'}), -1)

                # Validate positions
                if verb_index > 0 and op_index > 0 and op_index < len(words) - 2:
                    # Extract components based on verb position
                    if verb_index == 0:  # "is X plus Y Z?"
                        num1 = words[1]
                        operator = words[op_index]
                        num2 = words[op_index+1]
                        result = words[-1].rstrip('?')
                    else:  # Other patterns
                        num1 = words[verb_index-1]
                        operator = words[op_index]
                        num2 = words[op_index+1]
                        result = words[-1].rstrip('?')

                    # Reconstruct as statement: num1 operator num2 is result
                    return f"{num1} {operator} {num2} is {result}."

        return " ".join(words)

    def apply_correction_pattern(self, user_input, orig, corr):
        """Apply correction pattern with mathematical awareness"""
        input_words = user_input.split()
        orig_words = orig.split()
        corr_words = corr.split()

        # Check if this is an arithmetic correction
        if (len(orig_words) >= 4 and len(corr_words) >= 4 and
            any(op in orig_words for op in {'plus', 'minus', 'times'}) and
            'is' in corr_words):

            # Try to extract mathematical components from input
            verb_index = next((i for i, w in enumerate(input_words) if w in {'is', 'are'}), -1)
            op_index = next((i for i, w in enumerate(input_words) if w in {'plus', 'minus', 'times'}), -1)

            if verb_index != -1 and op_index != -1 and verb_index < len(input_words)-1:
                # Extract components from input
                num1 = input_words[verb_index+1] if verb_index == 0 else input_words[verb_index-1]
                operator = input_words[op_index]
                num2 = input_words[op_index+1]
                result = input_words[-1].rstrip('?')

                # Form proper response
                return f"{num1} {operator} {num2} is {result}"

        # Fallback to standard pattern application
        pattern_positions = []
        changing_positions = []

        for i in range(min(len(input_words), len(orig_words))):
            if input_words[i] == orig_words[i]:
                pattern_positions.append(i)
            else:
                changing_positions.append(i)

        # Apply pattern: use correction words for pattern positions
        response_words = []
        for i in range(len(corr_words)):
            if i < len(orig_words) and i in pattern_positions and i < len(corr_words):
                response_words.append(corr_words[i])
            elif i < len(input_words) and i in changing_positions:
                response_words.append(input_words[i])
            elif i < len(corr_words):
                response_words.append(corr_words[i])

        return " ".join(response_words)
    def process_input(self, user_input):
        self.universe.process_parameters()

        # Handle negation commands first (no logging)
        words = user_input.split()
        is_negation = False
        negation_words = ['No', 'no', 'Ei', 'ei', 'Nein', 'nein', 'Non', 'non']

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

        # Log non-negation input
        self.conversation_log.append(f"User: {user_input}")

        selected_response = None

        # 1. PRIORITY: Exact match in corrections.json
        if user_input in self.corrections:
            corrs = self.corrections[user_input]
            total_weight = sum(weight for _, weight in corrs)
            if total_weight > 0:
                weights = [weight for _, weight in corrs]
                selected_correction = random.choices(
                    [corr for corr, _ in corrs],
                    weights=weights,
                    k=1
                )[0]
                selected_response = selected_correction

                # Verbose output
                if self.universe.parameters["verbosity"] > 0:
                    print(f"{self.BLUE_BOLD}Subconscious Processes: {self.BLUE}Exact match in corrections.json selected{self.RESET}")

        # 2. Pattern-based correction: Match with changing words
        if selected_response is None:
            best_pattern = None
            max_pattern_words = 0

            for orig, corrs in self.corrections.items():
                if not corrs:
                    continue

                # Get best correction for this original
                best_corr, best_weight = max(corrs, key=lambda x: x[1])

                # Check if structure matches with changing words
                orig_words = orig.split()
                input_words = user_input.split()
                corr_words = best_corr.split()

                # Must have same number of words
                if len(orig_words) != len(input_words):
                    continue

                # Must have at least 2 pattern words (same words)
                pattern_positions = []
                changing_positions = []

                for i in range(len(orig_words)):
                    if orig_words[i] == input_words[i]:
                        pattern_positions.append(i)
                    else:
                        changing_positions.append(i)

                # Require at least 2 pattern words and at least 1 changing word
                if len(pattern_positions) >= 2 and len(changing_positions) >= 1:
                    if len(pattern_positions) > max_pattern_words:
                        max_pattern_words = len(pattern_positions)
                        best_pattern = (orig, best_corr, pattern_positions, changing_positions)

            # Apply best pattern found
            if best_pattern:
                orig, corr, pattern_positions, changing_positions = best_pattern
                selected_response = self.apply_correction_pattern(user_input, orig, corr)

                # Verbose output
                if self.universe.parameters["verbosity"] > 0:
                    pattern_str = ", ".join([orig.split()[i] for i in pattern_positions])
                    changing_str = ", ".join([user_input.split()[i] for i in changing_positions])
                    print(f"{self.BLUE_BOLD}Subconscious Processes: {self.BLUE}Pattern match: '{pattern_str}' with changes: '{changing_str}'{self.RESET}")

        # 3. Analyze sentence structure if not handled yet
        if selected_response is None:
            structure_id = self.analyze_sentence_structure(user_input)

            # Update language model with user input
            if ' ' in user_input:
                self.universe.update_with_sentence(user_input)
            else:
                self.universe.update_with_word(user_input)

            # Store original phrase for potential correction
            self.last_original = user_input

            # Try dynamic patterns
            dynamic_patterns = self.find_dynamic_patterns(structure_id)
            for pattern_id, changing_positions, operator_positions, verb_positions, pattern_strength in dynamic_patterns:
                response = self.apply_dynamic_pattern(structure_id, pattern_id, changing_positions, operator_positions, verb_positions)
                if response and response.strip().lower() != user_input.strip().lower():
                    selected_response = response
                    if self.universe.parameters["verbosity"] > 0:
                        print(f"{self.BLUE_BOLD}Subconscious Processes: {self.BLUE}Applied dynamic pattern (strength: {pattern_strength:.1f}){self.RESET}")
                    break

            # Then try structural patterns
            if selected_response is None:
                similar_structures = self.find_similar_structure(structure_id)
                for sid in similar_structures:
                    response = self.generate_structural_response(sid)
                    if response and response.strip().lower() != user_input.strip().lower():
                        selected_response = response
                        if self.universe.parameters["verbosity"] > 0:
                            print(f"{self.BLUE_BOLD}Subconscious Processes: {self.BLUE}Applied structural pattern{self.RESET}")
                        break

            # Fall back to base model if no pattern found
            if selected_response is None:
                # Generate responses
                responses = [self.universe.generate_sentence() for _ in range(37)]
                weights = [1.0] * 37

                # Apply corrections
                matching_corrections = []
                if self.corrections:
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
                                        f"Match:'{orig}' → '{corr}' "
                                        f"char:'{char_sim:.2f}' pair:'{pair_sim:.2f}' "
                                        f"triplet:'{triplet_sim:.2f}' word:'{word_sim:.2f}' "
                                        f"phrase:'{phrase_sim:.2f}' combined:'{combined_sim:.2f}' "
                                        f"weight:'{candidate_weight:.1f}')"
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

            # Analyze response structure and connect
            response_structure_id = self.analyze_sentence_structure(selected_response)
            self.connect_structures(structure_id, response_structure_id)

            # Update language model with response
            if ' ' in selected_response:
                self.universe.update_with_sentence(selected_response)
            else:
                self.universe.update_with_word(selected_response)

        # Print final response
        self.conversation_log.append(f"System: {selected_response}")
        self.typewrite_text(selected_response)
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
        return (seq_match * 0.5) + (jaccard * 0.5)

if __name__ == "__main__":
    generator = EnhancedLanguageGenerator()
    generator.start()