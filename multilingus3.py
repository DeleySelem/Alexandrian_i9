import random
import math
import string
import json
from collections import defaultdict
import re
import os
import sys
from datetime import datetime

class LinguisticUniverse:
    def __init__(self):
        # Linguistic model components
        self.letter_freq = defaultdict(int)
        self.word_freq = defaultdict(int)
        self.pos_letter_freq = defaultdict(lambda: defaultdict(int))
        self.word_lengths = []
        self.sentence_lengths = []
        self.word_transitions = defaultdict(lambda: defaultdict(int))
        self.sentence_starters = defaultdict(int)
        self.sentence_enders = defaultdict(int)
        self.language_history = []
        self.last_word = None
        
        # Logging system
        self.word_counter = 1
        self.phrase_counter = 1
        self.word_occurrences = []
        self.phrase_occurrences = []
        self.word_connections = defaultdict(set)
        
        # Parameter system
        self.parameters = {
            "imitation": 100.0,   # How much to use user-inputted words
            "creativity": 0.0,    # How much to generate novel words
            "verbosity": 20.0,    # Amount of processing details to show
            "randomization": 0.0, # Randomization level for other parameters
            "recursiveness": 5.0, # Depth of recursive processing
            "length": 50.0        # Response length control
        }
        self.feedback_log = []   # Stores 1 for positive, 0 for negative feedback
        self.success_threshold = 0.6  # Target success rate
        self.learning_rate = 0.1  # How quickly parameters adjust
        
        # Load existing data
        self._load_data()
        
    def _load_data(self):
        """Load existing data from JSON files if available"""
        try:
            if os.path.exists('words.json'):
                with open('words.json', 'r') as f:
                    self.word_occurrences = json.load(f)
                print("Loaded words.json")
                
            if os.path.exists('phrases.json'):
                with open('phrases.json', 'r') as f:
                    self.phrase_occurrences = json.load(f)
                print("Loaded phrases.json")
                
            if os.path.exists('mind.json'):
                with open('mind.json', 'r') as f:
                    mind_data = json.load(f)
                    for entry in mind_data:
                        idx = entry['word_index']
                        if 'connect_indexes' in entry and entry['connect_indexes']:
                            connections = [int(x) for x in entry['connect_indexes']]
                            self.word_connections[idx] = set(connections)
                print("Loaded mind.json")
            
            # Set counters to max index + 1
            if self.word_occurrences:
                self.word_counter = max(entry['word_index'] for entry in self.word_occurrences) + 1
            if self.phrase_occurrences:
                self.phrase_counter = max(entry['phrase_index'] for entry in self.phrase_occurrences) + 1
            
            # Rebuild frequency tables
            self._rebuild_frequencies()
            print("Rebuilt language model from existing data")
            
            # Load parameters if available
            if os.path.exists('parameters.json'):
                with open('parameters.json', 'r') as f:
                    self.parameters = json.load(f)
                print("Loaded parameters from parameters.json")
                
        except Exception as e:
            print(f"Error loading data: {e}, starting fresh")

    def deduplicate_words(self):
        """Combine duplicate word entries and update connections"""
        word_map = {}
        new_word_occurrences = []
        
        # Group words by cleaned form
        for entry in self.word_occurrences:
            key = (entry['raw_word'], entry['clean_word'])
            if key not in word_map:
                word_map[key] = {
                    'word_index': entry['word_index'],
                    'raw_word': entry['raw_word'],
                    'clean_word': entry['clean_word'],
                    'phrase_indexes': set(),
                    'connect_indexes': set(),
                    'popularity-%': entry.get('popularity-%', 0)
                }
                new_word_occurrences.append(word_map[key])
            
            # Merge data from duplicates
            if 'phrase_indexes' in entry:
                if isinstance(entry['phrase_indexes'], list):
                    word_map[key]['phrase_indexes'] |= set(entry['phrase_indexes'])
                else:
                    word_map[key]['phrase_indexes'].add(entry['phrase_indexes'])
            elif 'phrase_index' in entry:
                word_map[key]['phrase_indexes'].add(entry['phrase_index'])
            
            if 'connect_indexes' in entry:
                if isinstance(entry['connect_indexes'], list):
                    word_map[key]['connect_indexes'] |= set(entry['connect_indexes'])
                else:
                    word_map[key]['connect_indexes'].add(entry['connect_indexes'])
        
        # Update word occurrences
        self.word_occurrences = []
        for word in new_word_occurrences:
            # Convert sets to lists for JSON serialization
            word['phrase_indexes'] = list(word['phrase_indexes'])
            word['connect_indexes'] = list(word['connect_indexes'])
            self.word_occurrences.append(word)
        
        # Update phrase references
        for phrase in self.phrase_occurrences:
            if 'word_indices' in phrase:
                # Create mapping from old indices to new deduplicated indices
                index_map = {}
                for word in self.word_occurrences:
                    index_map[word['word_index']] = word['word_index']
                
                # Update phrase word indices
                new_indices = []
                for idx in phrase['word_indices']:
                    if idx in index_map:
                        new_indices.append(index_map[idx])
                phrase['word_indices'] = new_indices

    def deduplicate_phrases(self):
        """Combine duplicate phrase entries"""
        phrase_map = {}
        new_phrase_occurrences = []
        
        for phrase in self.phrase_occurrences:
            text = phrase['phrase']
            if text not in phrase_map:
                phrase_map[text] = {
                    'phrase_index': phrase['phrase_index'],
                    'phrase': text,
                    'word_indices': set(phrase.get('word_indices', []))
                }
                new_phrase_occurrences.append(phrase_map[text])
            else:
                # Merge word indices from duplicates
                phrase_map[text]['word_indices'] |= set(phrase.get('word_indices', []))
        
        # Convert sets to lists
        for phrase in new_phrase_occurrences:
            phrase['word_indices'] = list(phrase['word_indices'])
        
        self.phrase_occurrences = new_phrase_occurrences

    def _rebuild_frequencies(self):
        """Deduplicate and rebuild frequency tables"""
        # Clear existing data
        self.letter_freq.clear()
        self.word_freq.clear()
        self.pos_letter_freq.clear()
        self.word_lengths = []
        self.sentence_lengths = []
        self.word_transitions.clear()
        self.sentence_starters.clear()
        self.sentence_enders.clear()
        self.language_history = []
        
        # Rebuild word frequencies and language history
        word_index_map = {}
        for entry in self.word_occurrences:
            word = entry['clean_word']
            # Calculate frequency based on number of phrase appearances
            count = len(entry.get('phrase_indexes', [1]))
            self.word_freq[word] = count
            self.word_lengths.append(len(word))
            # Add to language history once per occurrence
            self.language_history.extend([word] * count)
            word_index_map[entry['word_index']] = word
            
            # Rebuild letter frequencies
            for pos, char in enumerate(word):
                self.letter_freq[char] += count
                self.pos_letter_freq[pos][char] += count
        
        # Rebuild phrase frequencies
        for phrase in self.phrase_occurrences:
            words = [w for w in phrase['phrase'].split() if w]
            if len(words) > 1:
                cleaned_words = [self.clean_word(w) for w in words]
                cleaned_words = [w for w in cleaned_words if w]
                
                if cleaned_words:
                    self.sentence_lengths.append(len(cleaned_words))
                    self.sentence_starters[cleaned_words[0]] += 1
                    self.sentence_enders[cleaned_words[-1]] += 1
                    
                    # Rebuild word transitions
                    for i in range(len(cleaned_words) - 1):
                        self.word_transitions[cleaned_words[i]][cleaned_words[i+1]] += 1
        
        # Set last word if available
        if self.language_history:
            self.last_word = self.language_history[-1]

    def clean_word(self, word):
        """Remove non-letter characters with deduplication"""
        cleaned = re.sub(r'[^a-zA-Z]', '', word)
        return cleaned.lower() if cleaned else ""

    def _log_utterance(self, utterance):
        """Log utterance with deduplication"""
        # Check for duplicate phrase
        for phrase in self.phrase_occurrences:
            if phrase['phrase'] == utterance:
                # Update existing phrase
                self.phrase_counter = max(self.phrase_counter, phrase['phrase_index'])
                return
        
        # Create new phrase entry
        self.phrase_counter += 1
        phrase_index = self.phrase_counter
        phrase_entry = {
            "phrase_index": phrase_index,
            "phrase": utterance,
            "word_indices": []
        }
        self.phrase_occurrences.append(phrase_entry)
        
        # Process words
        words = utterance.split()
        current_phrase_indices = []
        
        for raw_word in words:
            cleaned_word = self.clean_word(raw_word)
            
            if not cleaned_word:
                continue
                
            # Check for duplicate word
            word_exists = False
            for entry in self.word_occurrences:
                if entry['clean_word'] == cleaned_word and entry['raw_word'] == raw_word:
                    # Update existing word
                    entry['popularity-%'] = (self.word_freq.get(cleaned_word, 0) / 
                                            self.word_counter * 100)
                    # Add new phrase index
                    if 'phrase_indexes' in entry:
                        entry['phrase_indexes'].append(phrase_index)
                    else:
                        # Convert old single index to list
                        old_index = entry.get('phrase_index', phrase_index)
                        entry['phrase_indexes'] = [old_index]
                        if 'phrase_index' in entry:
                            del entry['phrase_index']
                    current_phrase_indices.append(entry['word_index'])
                    word_exists = True
                    break
            
            if not word_exists:
                # Create new word entry
                popularity = (self.word_freq.get(cleaned_word, 0) / 
                             self.word_counter * 100) if self.word_counter > 1 else 0.0
                
                word_entry = {
                    "word_index": self.word_counter,
                    "raw_word": raw_word,
                    "clean_word": cleaned_word,
                    "connect_indexes": [],
                    "popularity-%": popularity,
                    "phrase_indexes": [phrase_index]
                }
                self.word_occurrences.append(word_entry)
                current_phrase_indices.append(self.word_counter)
                self.word_counter += 1
        
        # Add connections only for new words
        for i, idx in enumerate(current_phrase_indices):
            for j in range(i+1, len(current_phrase_indices)):
                other_idx = current_phrase_indices[j]
                self.word_connections[idx].add(other_idx)
                self.word_connections[other_idx].add(idx)
        
        # Store word indices
        phrase_entry["word_indices"] = current_phrase_indices
    
    def save_logs(self):
        """Save deduplicated logs to JSON files"""
        # Apply deduplication before saving
        self.deduplicate_words()
        self.deduplicate_phrases()
        
        # Create deduplicated mind data
        mind_data = []
        seen_words = set()
        
        for idx, connections in self.word_connections.items():
            if idx not in seen_words:
                word_entry = next((w for w in self.word_occurrences 
                                  if w["word_index"] == idx), None)
                if word_entry:
                    mind_data.append({
                        "word_index": idx,
                        "raw_word": word_entry["raw_word"],
                        "clean_word": word_entry["clean_word"],
                        "connect_indexes": list(connections)
                    })
                    seen_words.add(idx)
        
        # Save files
        with open('mind.json', 'w') as f:
            json.dump(mind_data, f, indent=2)
        
        with open('words.json', 'w') as f:
            json.dump(self.word_occurrences, f, indent=2)
        
        with open('phrases.json', 'w') as f:
            json.dump(self.phrase_occurrences, f, indent=2)
            
        with open('parameters.json', 'w') as f:
            json.dump(self.parameters, f, indent=2)
        
        print("Saved deduplicated language data")

    def update_with_word(self, raw_word):
        """Update language model with a new word"""
        word = self.clean_word(raw_word)
        if not word:
            return
            
        # Update frequencies
        self.word_freq[word] += 1
        self.word_lengths.append(len(word))
        self.language_history.append(word)
        self.last_word = word
        
        # Update letter frequencies
        for pos, char in enumerate(word):
            self.letter_freq[char] += 1
            self.pos_letter_freq[pos][char] += 1
        
        # Log original input
        self._log_utterance(raw_word)
    
    def update_with_sentence(self, sentence):
        """Update language model with a new sentence"""
        words = sentence.split()
        cleaned_words = []
        
        # Process each word
        for raw_word in words:
            word = self.clean_word(raw_word)
            if word:
                self.word_freq[word] += 1
                self.word_lengths.append(len(word))
                self.language_history.append(word)
                cleaned_words.append(word)
                
                # Update letter frequencies
                for pos, char in enumerate(word):
                    self.letter_freq[char] += 1
                    self.pos_letter_freq[pos][char] += 1
        
        # Update sentence-level stats
        if cleaned_words:
            self.sentence_lengths.append(len(cleaned_words))
            self.sentence_starters[cleaned_words[0]] += 1
            self.sentence_enders[cleaned_words[-1]] += 1
            self.last_word = cleaned_words[-1]
            
            # Update word transitions
            for i in range(len(cleaned_words) - 1):
                self.word_transitions[cleaned_words[i]][cleaned_words[i+1]] += 1
        
        # Log original sentence
        self._log_utterance(sentence)
    
    def generate_letter(self, position=None):
        """Generate a letter based on frequency patterns"""
        if position is not None and position in self.pos_letter_freq and self.pos_letter_freq[position]:
            choices, weights = zip(*self.pos_letter_freq[position].items())
        elif self.letter_freq:
            choices, weights = zip(*self.letter_freq.items())
        else:
            return random.choice(string.ascii_lowercase)
        
        total = sum(weights)
        rand_val = random.uniform(0, total)
        cumulative = 0
        for choice, weight in zip(choices, weights):
            cumulative += weight
            if rand_val <= cumulative:
                return choice
        return choices[0]
    
    def predict_word_length(self):
        """Predict next word length using fractal patterns"""
        if not self.word_lengths:
            return random.randint(3, 8)
        
        if len(self.word_lengths) > 2:
            diffs = [abs(self.word_lengths[i] - self.word_lengths[i-1]) 
                    for i in range(1, len(self.word_lengths))]
            mean_diff = sum(diffs) / len(diffs)
            if len(diffs) > 1:
                std_dev = math.sqrt(sum((d - mean_diff)**2 for d in diffs) / (len(diffs) - 1))
                fd = mean_diff / std_dev if std_dev > 0 else 1.0
                randomness = min(1.0, fd / 3.0)
            else:
                randomness = 1.0
        else:
            randomness = 1.0
        
        if random.random() < randomness:
            return random.choice(self.word_lengths)
        return max(3, min(10, round(sum(self.word_lengths) / len(self.word_lengths))))
    
    def generate_word(self):
        """Generate a new word based on language patterns"""
        length = int(self.predict_word_length())
        candidate = ''.join(self.generate_letter(i) for i in range(length))
        
        # Enhance with matching logged words
        matches = self.find_matching_words(candidate)
        
        # Apply creativity parameter - if creativity is 0, use only existing words
        if self.parameters["creativity"] == 0:
            # When creativity is 0, choose from existing words only
            if not self.word_occurrences:
                return candidate
                
            # Find best matching existing word
            best_match = max(matches, key=lambda x: x[4]) if matches else None
            return best_match[0] if best_match else candidate
        elif random.random() < (self.parameters["creativity"] / 100.0):
            # High creativity - use raw candidate
            if self.parameters["verbosity"] > 40:
                print(f"High creativity: Using raw word '{candidate}'")
            return candidate
        else:
            # Low creativity - enhance with existing patterns
            enhanced = self.enhance_candidate(candidate, matches)
            if self.parameters["verbosity"] > 40:
                print(f"Low creativity: Enhanced '{candidate}' to '{enhanced}'")
            return enhanced
    
    def find_matching_words(self, candidate):
        """Find best matching words from logs based on letter patterns"""
        if not self.word_occurrences:
            return []
        
        best_matches = []
        max_score = -1
        
        for entry in self.word_occurrences:
            word = entry["clean_word"]
            
            # Calculate same_letters_number
            min_len = min(len(candidate), len(word))
            same_letters = sum(1 for i in range(min_len) if candidate[i] == word[i])
            
            # Calculate longest common subsequence
            m, n = len(candidate), len(word)
            dp = [[0] * (n+1) for _ in range(m+1)]
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if candidate[i-1] == word[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            common_subseq = dp[m][n]
            
            # Calculate matching score with recency factor
            popularity = entry.get("popularity-%", 1.0) / 100
            recency_weight = self._recency_factor(entry["word_index"])
            score = (same_letters * popularity * recency_weight) + (common_subseq * popularity * recency_weight)
            
            # Track best matches
            if score > max_score:
                max_score = score
                best_matches = [(word, same_letters, common_subseq, popularity, score)]
            elif score == max_score:
                best_matches.append((word, same_letters, common_subseq, popularity, score))
                
        return best_matches

    def _recency_factor(self, word_index):
        """Calculate recency factor based on word position in history"""
        try:
            # More recent = higher weight (last 20% get max weight)
            pos = [e["word_index"] for e in self.word_occurrences].index(word_index)
            return max(0.1, 1.0 - (pos / len(self.word_occurrences)))
        except ValueError:
            return 0.1  # Default weight for unseen words

    def enhance_candidate(self, candidate, matches):
        """Enhance generated word using best matches"""
        if not matches:
            return candidate
        
        # Calculate total score for normalization
        total_score = sum(match[4] for match in matches)
        
        # Handle cases where all scores are zero
        if total_score <= 0:
            return candidate
        
        # Calculate probabilities for each match
        match_probs = [match[4]/total_score for match in matches]
        
        # Choose a match based on score probabilities
        chosen_match = random.choices(matches, weights=match_probs, k=1)[0]
        base_word, same_letters, common_subseq, popularity, score = chosen_match
        min_len = min(len(candidate), len(base_word))
        
        # Build enhanced word
        new_word = []
        for i in range(len(candidate)):
            # Calculate position match percentage
            match_percent = same_letters / min_len if min_len > 0 else 0
            
            # Preserve matching letters with probability = match percentage
            if i < min_len and candidate[i] == base_word[i] and random.random() < match_percent:
                new_word.append(candidate[i])
            else:
                # Generate new letter for non-matching positions
                new_word.append(self.generate_letter(i))
        
        return ''.join(new_word)
    
    def generate_sentence(self):
        """Generate a sentence based on language patterns with reverse recursion"""
        # Determine sentence length based on parameters
        length_factor = self.parameters["length"] / 50.0
        if not self.sentence_lengths:
            base_length = random.randint(3, 5)
        else:
            base_length = max(2, min(8, round(sum(self.sentence_lengths) / len(self.sentence_lengths))))
        length = max(2, min(15, int(round(base_length * length_factor))))
        
        # Start with last user word if available
        if self.last_word and random.random() < (0.7 * (self.parameters["imitation"] / 100.0)):
            current_word = self.last_word
            if self.parameters["verbosity"] > 30:
                print(f"Starting with user word: {current_word}")
        else:
            current_word = self.generate_word()
            if self.parameters["verbosity"] > 30:
                print(f"Starting with generated word: {current_word}")
        
        words = [current_word]
        
        for i in range(length-1):
            # Apply imitation parameter
            use_imitation = random.random() < (self.parameters["imitation"] / 100.0)
            
            if use_imitation and current_word in self.word_transitions:
                next_words = list(self.word_transitions[current_word].keys())
                if next_words:  # Check if transitions exist
                    weights = [self.word_transitions[current_word][w] for w in next_words]
                    next_word = random.choices(next_words, weights=weights, k=1)[0]
                    if self.parameters["verbosity"] > 40:
                        print(f"Transition: {current_word} -> {next_word} (Imitation)")
                else:
                    next_word = self.generate_word()  # Fallback to word generation
                    if self.parameters["verbosity"] > 40:
                        print(f"Transition: {current_word} -> {next_word} (No transitions, using Creativity)")
            else:
                # Generate new word
                next_word = self.generate_word()
                if self.parameters["verbosity"] > 40:
                    print(f"Transition: {current_word} -> {next_word} (Creativity)")
            
            words.append(next_word)
            current_word = next_word
        
        return ' '.join(words).capitalize() + ('.' if random.random() > 0.2 else '!')
    
    def _recency_factor_transition(self, src, dest):
        """Calculate recency factor for word transitions"""
        # Check if this transition appears in recent history
        recent_weight = 1.0
        if self.language_history:
            # Look for src followed by dest in recent history
            for i in range(len(self.language_history)-2, max(-1, len(self.language_history)-11), -1):
                if self.language_history[i] == src and self.language_history[i+1] == dest:
                    # More recent = higher weight
                    recency = len(self.language_history) - i
                    recent_weight = max(2.0, min(5.0, recency/2))
                    break
        return self.word_transitions[src][dest] * recent_weight

    def fractal_temperature(self):
        """Calculate creativity level based on language complexity"""
        if len(self.language_history) < 5:
            return 1.0
        
        lengths = [len(word) for word in self.language_history[-50:]]
        if len(lengths) > 2:
            diffs = [abs(lengths[i] - lengths[i-1]) for i in range(1, len(lengths))]
            mean_diff = sum(diffs) / len(diffs)
            if len(diffs) > 1:
                std_dev = math.sqrt(sum((d - mean_diff)**2 for d in diffs) / (len(diffs) - 1))
                fd = mean_diff / std_dev if std_dev > 0 else 1.0
                return min(1.0, max(0.1, fd / 3.0))
        return 0.5
    
    def generate_auto_input(self):
        """Generate automatic input mimicking user behavior"""
        if random.random() < 0.7 and self.last_word:
            # Create phrase continuation
            if (random.random() < 0.6 
                and self.last_word in self.word_transitions 
                and self.word_transitions[self.last_word]):  # Check non-empty transitions
            
                next_words = list(self.word_transitions[self.last_word].keys())
                weights = list(self.word_transitions[self.last_word].values())
            
                if weights:  # Critical safety check
                    next_word = random.choices(next_words, weights=weights, k=1)[0]
                    return f"{self.last_word} {next_word}"
        
            return self.generate_word()
        return self.generate_sentence()
    
    def generate_response(self, user_input):
        """Generate response based on user input and language state"""
        if ' ' in user_input:
            self.update_with_sentence(user_input)
        else:
            self.update_with_word(user_input)
        
        temperature = self.fractal_temperature()
        if random.random() < temperature:
            word_count = min(5, max(1, int(5 * temperature)))
            return ' '.join(self.generate_word() for _ in range(word_count))
        else:
            return self.generate_sentence()
    
    def build_mind_tree(self, max_depth=None):
        """Generate comprehensive mind tree of all word connections"""
        if not self.word_freq:
            return None
        
        # Create index to word mapping
        index_to_word = {}
        for entry in self.word_occurrences:
            index_to_word[entry["word_index"]] = entry["clean_word"]
        
        # If max_depth is not specified, build full infinite tree
        if max_depth is None:
            max_depth = float('inf')
        
        # Find most popular word as root
        popular_word = max(self.word_freq, key=self.word_freq.get)
        
        # Find first occurrence
        first_occurrence = None
        for entry in self.word_occurrences:
            if entry["clean_word"] == popular_word:
                first_occurrence = entry
                break
        if not first_occurrence:
            return None
        
        # Recursive tree builder with infinite depth capability
        def build_branch(word_type, depth, visited=None):
            if visited is None:
                visited = set()
            if depth > max_depth or word_type in visited:
                return None
                
            visited.add(word_type)
            
            # Find word occurrence
            word_entry = None
            for entry in self.word_occurrences:
                if entry["clean_word"] == word_type:
                    word_entry = entry
                    break
            if not word_entry:
                return None
                
            # Get transitions
            transitions = self.word_transitions[word_type]
            if not transitions:
                return [word_entry["word_index"]]
                
            # Build children with all connections (no limit)
            children = []
            for next_word, _ in transitions.items():
                branch = build_branch(next_word, depth+1, visited.copy())
                if branch:
                    children.append(branch)
            
            return {
                "node": word_entry["word_index"],
                "children": children
            }
        
        return build_branch(popular_word, 0)
    
    def print_multifractal_analysis(self, save_to_file=False):
        """Comprehensive multifractal analysis with file saving option"""
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"linguistic_analysis_{timestamp}.txt"
            with open(filename, 'w') as f:
                original_stdout = sys.stdout
                sys.stdout = Tee(original_stdout, f)
                try:
                    self._perform_analysis()
                finally:
                    sys.stdout = original_stdout
            print(f"Saved linguistic analysis to {filename}")
        else:
            self._perform_analysis()
    
    def _perform_analysis(self):
        """Actual analysis implementation - prints ALL data without limits"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MULTIFRACTAL ANALYSIS (FULL DATA)")
        print("="*60)
        
        # 1. Letter Frequency Analysis
        print("\nLETTER FREQUENCIES (Global)")
        print("-"*50)
        for char, count in sorted(self.letter_freq.items(), key=lambda x: x[1], reverse=True):
            print(f"{char}: {count}")
            
        # Position-specific letter frequencies
        print("\nLETTER FREQUENCIES (Position-Specific)")
        print("-"*50)
        for pos in sorted(self.pos_letter_freq.keys()):
            print(f"\nPosition {pos}:")
            for char, count in sorted(self.pos_letter_freq[pos].items(), key=lambda x: x[1], reverse=True):
                print(f"  {char}: {count}")
        
        # 2. Word Frequency Analysis
        print("\nWORD FREQUENCIES (ALL WORDS)")
        print("-"*100)
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        for i, (word, count) in enumerate(sorted_words, 1):
            print(f"{i:4d}. {word}: {count}")
        
        # 3. Phrase Frequency Analysis
        print("\nPHRASE FREQUENCIES (ALL PHRASES)")
        print("-"*100)
        phrase_counts = defaultdict(int)
        for phrase in self.phrase_occurrences:
            phrase_counts[phrase['phrase']] += 1
        sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (phrase, count) in enumerate(sorted_phrases, 1):
            print(f"{i:4d}. {phrase}: {count}")
        
        # 4. Sentence Structure Analysis
        print("\nSENTENCE STRUCTURE ANALYSIS")
        print("-"*100)
        if self.sentence_lengths:
            print(f"Average Sentence Length: {sum(self.sentence_lengths)/len(self.sentence_lengths):.2f}")
        else:
            print("No sentence data available")
        
        print("\nSENTENCE STARTERS (ALL):")
        sorted_starters = sorted(self.sentence_starters.items(), key=lambda x: x[1], reverse=True)
        for i, (word, count) in enumerate(sorted_starters, 1):
            print(f"{i:4d}. {word}: {count}")
            
        print("\nSENTENCE ENDERS (ALL):")
        sorted_enders = sorted(self.sentence_enders.items(), key=lambda x: x[1], reverse=True)
        for i, (word, count) in enumerate(sorted_enders, 1):
            print(f"{i:4d}. {word}: {count}")
        
        # 5. Word Connections
        print("\nWORD CONNECTION NETWORK")
        print("-"*100)
        print(f"Total Words: {len(self.word_occurrences)}")
        print(f"Total Connections: {sum(len(v) for v in self.word_connections.values())}")
        
        # 6. JSON Data Trees
        print("\n" + "="*100)
        print("JSON DATA STRUCTURES (Tree View)")
        print("="*100)
        
        # Words.json tree
        print("\nWORDS.JSON STRUCTURE")
        print("-"*100)
        self.print_json_tree(self.word_occurrences, "word_index", "clean_word", ["connect_indexes", "popularity-%", "phrase_indexes"])
        
        # Phrases.json tree
        print("\nPHRASES.JSON STRUCTURE")
        print("-"*100)
        self.print_json_tree(self.phrase_occurrences, "phrase_index", "phrase", ["word_indices"])
        
        # Mind.json tree
        print("\nMIND.JSON STRUCTURE")
        print("-"*100)
        try:
            if os.path.exists('mind.json'):
                with open('mind.json', 'r') as f:
                    mind_data = json.load(f)
                self.print_json_tree(mind_data, "word_index", "clean_word", ["connect_indexes"])
            else:
                print("Mind data not available yet")
        except Exception as e:
            print(f"Error loading mind.json: {e}")
        
        # Mind tree visualization
        print("\n" + "="*60)
        print("MIND TREE VISUALIZATION")
        print("="*60)
        if self.word_freq:
            tree = self.build_mind_tree(None)  # Full infinite tree
            if tree:
                self.print_mind_tree(tree)
    
    def print_json_tree(self, data, id_key, name_key, value_keys, depth=0, prefix="", is_last=True):
        """Recursively print JSON data in tree structure with infinite depth"""
        connectors = {
            'branch': "├── ",
            'last': "└── ",
            'vertical': "│   ",
            'space': "    "
        }
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                is_last_item = (i == len(data)-1)
                self.print_json_tree(item, id_key, name_key, value_keys, depth+1, 
                                   prefix + (connectors['space'] if is_last else connectors['vertical']), 
                                   is_last_item)
        elif isinstance(data, dict):
            # Print current node
            node_id = data.get(id_key, "?")
            node_name = data.get(name_key, "Unnamed")
            print(f"{prefix}{connectors['last' if is_last else 'branch']}{node_id}: {node_name}")
            
            # Print additional values
            new_prefix = prefix + (connectors['space'] if is_last else connectors['vertical'])
            for key in value_keys:
                if key in data:
                    value = data[key]
                    if isinstance(value, (list, dict)):
                        print(f"{new_prefix}{connectors['branch']}{key}:")
                        self.print_json_tree(value, id_key, name_key, value_keys, depth+1, 
                                           new_prefix + connectors['vertical'], 
                                           False)
                    else:
                        print(f"{new_prefix}{connectors['branch']}{key}: {value}")
        else:
            # Print leaf nodes
            print(f"{prefix}{connectors['last' if is_last else 'branch']}{data}")

    def print_mind_tree(self, node, depth=0, prefix="", is_last=True):
        """Print mind tree with words and connections"""
        connectors = {
            'branch': "├── ",
            'last': "└── ",
            'vertical': "│   ",
            'space': "    "
        }
        
        if node is None:
            return
            
        if isinstance(node, dict):
            node_id = node.get('node', '?')
            children = node.get('children', [])
        elif isinstance(node, list):
            node_id = node[0] if node else '?'
            children = node[1:] if len(node) > 1 else []
        else:
            node_id = str(node)
            children = []
            
        # Get word from index
        word = "Unknown"
        for entry in self.word_occurrences:
            if entry["word_index"] == node_id:
                word = entry["clean_word"]
                break
        
        # Print current node
        if depth == 0:
            print(f"{word}")
        else:
            new_prefix = prefix + (connectors['space'] if is_last else connectors['vertical'])
            print(f"{prefix}{connectors['last' if is_last else 'branch']}{word}")
        
        # Print children
        child_count = len(children)
        for i, child in enumerate(children):
            is_last_child = (i == child_count - 1)
            self.print_mind_tree(child, depth+1, prefix + (connectors['space'] if is_last else connectors['vertical']), is_last_child)

    def process_parameters(self):
        """Apply randomization and dynamic learning to parameters"""
        # Store base values before randomization
        base_params = self.parameters.copy()
        
        # Apply randomization
        randomization = self.parameters["randomization"] / 100.0
        for param in self.parameters:
            if param != "randomization":
                delta = random.uniform(-randomization, randomization) * 100
                self.parameters[param] = max(0, min(100, base_params[param] + delta))
                
        # Calculate success rate
        success_rate = self.calculate_success_rate()
        
        # Verbose output
        if self.parameters["verbosity"] > 50:
            print(f"\nParameter processing:")
            print(f"  Base params: {base_params}")
            print(f"  Current params: {self.parameters}")
            print(f"  Success rate: {success_rate:.2f} (Target: {self.success_threshold})")
                
        # Adjust parameters based on performance
        if success_rate < self.success_threshold:
            # Increase exploration
            self.parameters["randomization"] = min(100, 
                self.parameters["randomization"] * 1.2)
            self.parameters["recursiveness"] = min(100,
                self.parameters["recursiveness"] * 1.3)
            if self.parameters["verbosity"] > 30:
                print("  Low success rate - increasing exploration")
        else:
            # Refine successful parameters
            self.parameters["randomization"] = max(5,
                self.parameters["randomization"] * 0.9)
            if self.parameters["verbosity"] > 30:
                print("  Good success rate - refining parameters")

    def calculate_success_rate(self):
        """Calculate recent feedback success rate"""
        if not self.feedback_log:
            return 0.5
        recent = self.feedback_log[-10:]
        if not recent:
            return 0.5
        return sum(recent) / len(recent)

    def handle_question(self, user_input):
        """Special handling for questions"""
        # Extract question words
        question_words = ["what", "who", "where", "when", "why", "how"]
        words = [w.lower().strip('.,?!') for w in user_input.split()]
        subject = next((w for w in words if w in question_words), "it")
        
        # Increase recursiveness for questions
        self.parameters["recursiveness"] = min(100,
            self.parameters["recursiveness"] * 1.5)
        
        # Increase verbosity for complex questions
        if len(words) > 5:
            self.parameters["verbosity"] = min(100,
                self.parameters["verbosity"] * 1.2)
        
        # Generate response with subject
        if self.parameters["verbosity"] > 20:
            print(f"Detected question about '{subject}'")
        return f"{subject.capitalize()} {self.generate_sentence()}?"
        
       
    def delete_word(self, raw_word):
        """Completely delete a word and all its connections from the linguistic universe"""
        cleaned_word = self.clean_word(raw_word)
        
        if not cleaned_word:
            print(f"Invalid word: '{raw_word}'")
            return
            
        print(f"Deleting word: '{raw_word}' (clean: '{cleaned_word}')")
        
        # Track affected phrases and connections
        affected_phrases = set()
        deleted_word_indices = set()
        
        # 1. Delete from word_occurrences and record word indices
        new_word_occurrences = []
        for entry in self.word_occurrences:
            if entry['clean_word'] == cleaned_word or entry['raw_word'] == raw_word:
                deleted_word_indices.add(entry['word_index'])
                if 'phrase_indexes' in entry:
                    affected_phrases |= set(entry['phrase_indexes'])
                elif 'phrase_index' in entry:
                    affected_phrases.add(entry['phrase_index'])
            else:
                new_word_occurrences.append(entry)
        self.word_occurrences = new_word_occurrences
        
        # 2. Remove from word frequency counts
        if cleaned_word in self.word_freq:
            del self.word_freq[cleaned_word]
        
        # 3. Remove from phrase_occurrences
        new_phrase_occurrences = []
        for phrase in self.phrase_occurrences:
            if phrase['phrase_index'] in affected_phrases:
                # Remove the specific word from the phrase
                new_phrase = ' '.join([w for w in phrase['phrase'].split() 
                                       if w != raw_word and self.clean_word(w) != cleaned_word])
                
                # Update word indices in phrase
                if 'word_indices' in phrase:
                    new_indices = [idx for idx in phrase['word_indices'] 
                                  if idx not in deleted_word_indices]
                else:
                    new_indices = []
                
                # Only keep non-empty phrases
                if new_phrase.strip():
                    phrase['phrase'] = new_phrase
                    phrase['word_indices'] = new_indices
                    new_phrase_occurrences.append(phrase)
            else:
                new_phrase_occurrences.append(phrase)
        self.phrase_occurrences = new_phrase_occurrences
        
        # 4. Remove word connections
        # Remove connections FROM the deleted words
        for idx in deleted_word_indices:
            if idx in self.word_connections:
                del self.word_connections[idx]
                
        # Remove connections TO the deleted words
        for word_idx, connections in self.word_connections.items():
            self.word_connections[word_idx] = connections - deleted_word_indices
        
        # 5. Update language history
        self.language_history = [w for w in self.language_history if w != cleaned_word]
        self.last_word = self.language_history[-1] if self.language_history else None
        
        # 6. Update transitions
        # Remove as key in transitions
        if cleaned_word in self.word_transitions:
            del self.word_transitions[cleaned_word]
            
        # Remove as value in other transitions
        for word, transitions in self.word_transitions.items():
            if cleaned_word in transitions:
                del transitions[cleaned_word]
        
        # 7. Update sentence starters/enders
        if cleaned_word in self.sentence_starters:
            del self.sentence_starters[cleaned_word]
        if cleaned_word in self.sentence_enders:
            del self.sentence_enders[cleaned_word]
            
        # 8. Rebuild frequencies to ensure consistency
        self._rebuild_frequencies()
        
        # 9. Save changes immediately
        self.save_logs()
        print(f"Completed deletion of '{raw_word}'")

class Tee:
    """Duplicate output to both console and file"""
    def __init__(self, *files):
        self.files = files
        
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            
    def flush(self):
        for f in self.files:
            if hasattr(f, 'flush'):
                f.flush()

class LanguageGenerator:
    def __init__(self):
        self.universe = LinguisticUniverse()
        # Cold blue/cyan/light green theme
        self.PROMPT = "\033[92m"  # Green
        self.USER_INPUT = "\033[96m"  # Light Cyan
        self.MACHINE_PROMPT = "\033[95m"  # Purple
        self.MACHINE_RESPONSE = "\033[96m"  # Light Cyan
        self.COMMAND = "\033[93m"  # Yellow
        self.RESET = "\033[0m"
        self.auto_input_counter = 0
        
    def start(self):
        print(f"{self.MACHINE_RESPONSE}M U L T I L I N G U S v. 3.2.2. Linguistic Generator!{self.RESET}")
        print(f"{self.COMMAND}Type 'help' for available commands{self.RESET}")
        
        while True:
            try:
                # Auto-generate input
                if self.auto_input_counter >= random.randint(3, 5):
                    auto_input = self.universe.generate_auto_input()
                    if ' ' in auto_input:
                        self.universe.update_with_sentence(auto_input)
                    else:
                        self.universe.update_with_word(auto_input)
                    self.auto_input_counter = 0
                    print(f"{self.MACHINE_RESPONSE}>> Auto-learned: {auto_input}{self.RESET}")
                
                # Get user input
                user_input = input(f"{self.PROMPT}> {self.USER_INPUT}").strip()
                print(self.RESET, end='')
                
                if user_input.lower() == 'quit':
                    self.universe.save_logs()
                    print(f"{self.MACHINE_RESPONSE}Exiting and saving language data.{self.RESET}")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                elif user_input.lower() == 'print mind':
                    self.handle_print_mind()
                elif user_input.lower() == 'print all':
                    self.universe.print_multifractal_analysis(save_to_file=True)
                elif user_input.lower() == 'params':
                    self.show_parameters()
                elif user_input.startswith('set '):
                    self.handle_set_parameter(user_input)
                elif ' is not a word' in user_input.lower():
                    raw_word = user_input.split(' is not a word', 1)[0].strip()
                    if raw_word:
                        self.universe.delete_word(raw_word)
                        print(f"{self.MACHINE_RESPONSE}Deleted '{raw_word}' from all linguistic data{self.RESET}")
                    else:
                        print(f"{self.MACHINE_RESPONSE}Please specify a word to delete{self.RESET}")
                elif user_input:
                    self.process_input(user_input)
                
            except KeyboardInterrupt:
                print(f"\n{self.MACHINE_RESPONSE}Exiting and saving language data.{self.RESET}")
                self.universe.save_logs()
                break

    def show_help(self):
        """Show available commands with new color scheme"""
        print(f"\n{self.COMMAND}AVAILABLE COMMANDS:{self.RESET}")
        print(f"{self.COMMAND}  help                 {self.RESET}- Show this help message")
        print(f"{self.COMMAND}  quit                 {self.RESET}- Exit and save language data")
        print(f"{self.COMMAND}  print mind           {self.RESET}- Generate word connection tree")
        print(f"{self.COMMAND}  print all            {self.RESET}- Full linguistic analysis")
        print(f"{self.COMMAND}  params               {self.RESET}- Show current parameter settings")
        print(f"{self.COMMAND}  set <param> <value>  {self.RESET}- Adjust parameters")
        print(f"{self.COMMAND}  <word> is not a word {self.RESET}- Delete word from linguistic data")
        print(f"{self.COMMAND}  No, Ei <input>       {self.RESET}- Teach desired output")
    
    def show_parameters(self):
        """Show parameters with percentage bars and numerical values"""
        print(f"\n{self.COMMAND}CURRENT PARAMETERS:{self.RESET}")
        for param, value in self.universe.parameters.items():
            bar_length = 20
            filled = int(bar_length * value / 100)
            bar = '█' * filled + '-' * (bar_length - filled)
            print(f"{self.COMMAND}{param.capitalize().ljust(15)}: {self.RESET}{bar} {value:.1f}%")

    def handle_set_parameter(self, user_input):
        """Handle set parameter command"""
        parts = user_input.split()
        if len(parts) == 3 and parts[1] in self.universe.parameters:
            try:
                value = float(parts[2])
                self.universe.parameters[parts[1]] = max(0, min(100, value))
                print(f"Set {parts[1]} to {value}")
                
                # Special handling for creativity=0
                if parts[1] == "creativity" and value == 0:
                    print(f"{self.COMMAND}Creativity set to 0 - only existing words will be used{self.RESET}")
            except ValueError:
                print("Invalid value. Must be a number between 0-100")
        else:
            print("Usage: set <parameter> <value>")
            print("Available parameters: imitation, creativity, verbosity, randomization, recursiveness, length")
    
    def handle_print_mind(self):
        """Handle print mind command with full word display"""
        try:
            depth_input = input(f"{self.COMMAND}Enter recursion depth or 'all' for complete tree: {self.RESET}").strip().lower()
            
            if depth_input == 'all':
                max_depth = None
                print(f"{self.MACHINE_RESPONSE}Building comprehensive mind tree with all connections...{self.RESET}")
            else:
                depth = int(depth_input)
                max_depth = max(1, depth)
                print(f"{self.MACHINE_RESPONSE}Building mind tree with depth={max_depth}...{self.RESET}")
            
            tree = self.universe.build_mind_tree(max_depth)
            
            if tree:
                print(f"\n{self.MACHINE_RESPONSE}COMPREHENSIVE MIND TREE (All Word Connections):{self.RESET}")
                self.print_tree(tree, node_to_str=lambda idx: self.get_word_by_index(idx))
                
                # Save to JSON
                with open('comprehensive_mind_tree.json', 'w') as f:
                    json.dump(tree, f, indent=2)
                print(f"\n{self.MACHINE_RESPONSE}Saved mind tree to comprehensive_mind_tree.json{self.RESET}")
            else:
                print(f"{self.MACHINE_RESPONSE}Not enough data to generate tree{self.RESET}")
        except ValueError:
            print(f"{self.MACHINE_RESPONSE}Invalid depth input. Use numbers or 'all'{self.RESET}")

    def process_input(self, user_input):
        """Process user input with new color scheme"""
        self.universe.process_parameters()
        
        if user_input.endswith('?'):
            response = self.universe.handle_question(user_input)
        else:
            if ' ' in user_input:
                self.universe.update_with_sentence(user_input)
            else:
                self.universe.update_with_word(user_input)
            
            temperature = self.universe.fractal_temperature()
            if random.random() < temperature:
                word_count = min(5, max(1, int(5 * temperature)))
                response = ' '.join(self.universe.generate_word() for _ in range(word_count))
            else:
                response = self.universe.generate_sentence()
        
        # Display response with new color scheme
        print(f"{self.MACHINE_PROMPT}> {self.MACHINE_RESPONSE}{response}{self.RESET}")
        self.auto_input_counter += 1

    def olddget_popularity_color(self, word):
        """Generate smooth heat map color from blue → cyan → green → yellow → orange → red."""
        if not self.universe.word_freq:
            return self.RESET

        # Get frequencies and calculate normalized popularity
        freqs = list(self.universe.word_freq.values())
        min_freq = min(freqs) if freqs else 0
        max_freq = max(freqs) if freqs else 1
        freq = self.universe.word_freq.get(word, min_freq)

        # Avoid division by zero
        if max_freq == min_freq:
            normalized = 0.5
        else:
            normalized = (freq - min_freq) / (max_freq - min_freq)

        # Smoothly transition colors based on normalized value
        if normalized < 0.2:  # Blue to Cyan
            red = 0
            green = int(255 * (normalized / 0.0))
            blue = 255
        elif normalized < 0.25:  # Cyan to Green
            red = 0
            green = 255
            blue = int(255 * (1 - (normalized - 0.25) / 0.25))
        elif normalized < 0.5:  # Green to Yellow
            red = int(255 * ((normalized - 0.5) / 0.25))
            green = 255
            blue = 0
        elif normalized < 0.75:  # Yellow to Orange
            red = 255
            green = int(255 * (1 - (normalized - 1.0) / 0.25))
            blue = 0
        else:  # Orange to Red
            red = 255
            green = int(128 * (1 - (normalized - 0.8) / 0.2))
            blue = 0

        return f"\033[38;2;{red};{green};{blue}m"


    def o2ldget_popularity_color(self, word):
        """Generate smooth heat map color from blue → cyan → green → yellow → white"""
        if not self.universe.word_freq:
            return self.RESET
    
        # Get frequencies and calculate normalized popularity
        freqs = list(self.universe.word_freq.values())
        min_freq = min(freqs) if freqs else 0
        max_freq = max(freqs) if freqs else 1
        freq = self.universe.word_freq.get(word, min_freq)
    
        # Avoid division by zero
        if max_freq == min_freq:
            normalized = 0.5
        else:
            normalized = (freq - min_freq) / (max_freq - min_freq)
    
        # Calculate color components based on normalized popularity
        if normalized < 0.05:  # Blue to Cyan: (0,0,255) → (0,255,255)
            red = 255
            green = int(255 * (normalized / 0.05))
            blue = 255
        elif normalized < 0.33:  # Cyan to Green: (0,255,255) → (0,255,0)
            red = 139
            green = 255
            blue = int(255 * (1 - ((normalized - 0.33) / 0.33)))
        elif normalized < 0.66:  # Green to Yellow: (0,255,0) → (255,255,0)
            red = int(255 * ((normalized - 0.5) / 0.33))
            green = 255
            blue = 100
        else:  # Yellow to White: (255,255,0) → (255,255,255)
            red = 255
            green = 255
            blue = int(255 * ((normalized - 0.66) / 0.25))
    
        return f"\033[38;2;{red};{green};{blue}m"

    def print_tree(self, node, depth=0, is_last=True, prefix='', node_to_str=None):
        if node_to_str is None:
            node_to_str = str
        if node is None:
            return
        
        # Handle both dict and list structures
        if isinstance(node, dict):
            node_id = node.get('node', '?')
            children = node.get('children', [])
        elif isinstance(node, list):
            node_id = node[0] if node else '?'
            children = node[1:] if len(node) > 1 else []
        else:
    
            node_id = str(node)
            children = []
        
        display_str = node_to_str(node_id)
    
        # Get color based on popularity
        color_code = self.get_popularity_color(display_str)
    
        # Print current node
        connectors = {
   
        'branch': "├── ",
            'last': "└── ",
            'vertical': "│   ",
            'space': "    "
        }
    
        if depth == 0:
            print(f"{color_code}{display_str}{self.RESET}")
        else:
            new_prefix = prefix + (connectors['space'] if is_last else connectors['vertical'])
            print(f"{prefix}{connectors['last' if is_last else 'branch']}{color_code}{display_str}{self.RESET}")
    
        # Print children
        child_count = len(children)
        for i, child in enumerate(children):
            is_last_child = (i == child_count - 1)
            self.print_tree(child, depth+1, is_last_child, 
                           prefix + (connectors['space'] if is_last else connectors['vertical']), 
                           node_to_str)

    def hgget_popularity_color(self, word):
        """Generate smooth heat map color from blue → cyan → green → yellow → white"""
        if not self.universe.word_freq:
            return self.RESET
    
        # Get frequencies and calculate normalized popularity
        freqs = list(self.universe.word_freq.values())
        min_freq = min(freqs) if freqs else 0
        max_freq = max(freqs) if freqs else 1
        freq = self.universe.word_freq.get(word, min_freq)
    
        # Avoid division by zero
        if max_freq == min_freq:
            normalized = 0.5
        else:
            normalized = (freq - min_freq) / (max_freq - min_freq)
    
        # Calculate color components based on normalized popularity
        if normalized < 0.25:  # Blue to Cyan: (0,0,255) → (0,255,255)
            red = 0
            green = int(255 * (normalized / 0.25))
            blue = 255
        elif normalized < 0.5:  # Cyan to Green: (0,255,255) → (0,255,0)
            red = 0
            green = 255
            blue = int(255 * (1 - ((normalized - 0.25) / 0.25)))
        elif normalized < 0.75:  # Green to Yellow: (0,255,0) → (255,255,0)
            red = int(255 * ((normalized - 0.5) / 0.25))
            green = 255
            blue = 0
        else:  # Yellow to White: (255,255,0) → (255,255,255)
            red = 255
            green = 255
            blue = int(255 * ((normalized - 0.75) / 0.25))
    
        return f"\033[38;2;{red};{green};{blue}m"

    def get_popularity_color(self, word):
        """Generate logarithmic heat map color from blue → cyan → green → yellow"""
        if not self.universe.word_freq:
            return self.RESET
    
        # Get frequencies and calculate normalized popularity
        freqs = list(self.universe.word_freq.values())
        min_freq = min(freqs) if freqs else 0
        max_freq = max(freqs) if freqs else 1
        freq = self.universe.word_freq.get(word, min_freq)
    
        # Avoid division by zero
        if max_freq == min_freq:
            normalized = 0.5
        else:
            normalized = (freq - min_freq) / (max_freq - min_freq)
    
        # Apply logarithmic scaling for faster color changes at low popularity
        log_normalized = math.log1p(normalized * 10) / math.log1p(10)
    
        # Calculate color components with logarithmic progression
        if log_normalized < 0.33:  # Blue to Cyan
            # Logarithmic progression: Blue (0,0,255) → Cyan (0,255,255)
            blue = 255
            green = int(255 * log_normalized * 3)
            red = 255
        elif log_normalized < 0.66:  # Cyan to Green
            # Logarithmic progression: Cyan (0,255,255) → Green (0,255,0)
            green = 255
            blue = 255 - int(255 * (log_normalized - 0.33) * 3)
            red = 0
        else:  # Green to Yellow (peak)
            # Logarithmic progression: Green (0,255,0) → Yellow (255,255,0)
            green = 255
            red = int(255 * (log_normalized - 0.66) * 3)
            blue = 0
    
        return f"\033[38;2;{red};{green};{blue}m"


    def print_tree(self, node, depth=0, is_last=True, prefix='', node_to_str=None):
        if node_to_str is None:
            node_to_str = str
        if node is None:
            return
        
        # Handle both dict and list structures
        if isinstance(node, dict):
            node_id = node.get('node', '?')
            children = node.get('children', [])
        elif isinstance(node, list):
            node_id = node[0] if node else '?'
            children = node[1:] if len(node) > 1 else []
        else:
            node_id = str(node)
            children = []
        
        display_str = node_to_str(node_id)
    
        # Get color based on popularity
        color_code = self.get_popularity_color(display_str)
    
        # Print current node
        connectors = {
            'branch': "├── ",
            'last': "└── ",
            'vertical': "│   ",
            'space': "    "
        }
    
        if depth == 0:
            print(f"{color_code}{display_str}{self.RESET}")
        else:
            new_prefix = prefix + (connectors['space'] if is_last else connectors['vertical'])
            print(f"{prefix}{connectors['last' if is_last else 'branch']}{color_code}{display_str}{self.RESET}")
    
        # Print children
        child_count = len(children)
        for i, child in enumerate(children):
            is_last_child = (i == child_count - 1)
            self.print_tree(child, depth+1, is_last_child, 
                           prefix + (connectors['space'] if is_last else connectors['vertical']), 
                           node_to_str)


    def ghhhet_popularity_color(self, word):
        """Generate smooth heat map color from blue → cyan → green → yellow → orange → red."""
        if not self.universe.word_freq:
            return self.RESET

        # Get frequencies and calculate normalized popularity
        freqs = list(self.universe.word_freq.values())
        min_freq = min(freqs) if freqs else 0
        max_freq = max(freqs) if freqs else 1
        freq = self.universe.word_freq.get(word, min_freq)

        # Avoid division by zero
        if max_freq == min_freq:
            normalized = 0.5
        else:
            normalized = (freq - min_freq) / (max_freq - min_freq)

        # Smoothly transition colors based on normalized value
        if normalized < 0.25:  # Blue to Cyan
            red = 0
            green = int(255 * (normalized / 0.25)) 
            blue = 255
        elif normalized < 0.25:  # Cyan to Green
            red = 0
            green = 255
            blue = int(255 * (1 - (normalized - 0.25) / 0.25))
        elif normalized < 0.6:  # Green to Yellow
            red = int(255 * ((normalized - 0.5) / 0.25))
            green = 255
            blue = 0
   #     elif normalized < 0.75:  # Yellow to Orange
   #         red = 0
   #         green = int(255 * (1 - (normalized - 0.75) * 4))
   #         blue = 0
        else:  # Orange to Red
            red = 255
            green =255
            blue = int(255 * ((normalized - 0.75) / 0.25))

        return f"\033[38;2;{red};{green};{blue}m"

    def oldget_popularity_color(self, word):
        """Generate smooth heat map color based on word popularity"""
        if not self.universe.word_freq:
            return self.RESET
        
        # Get frequencies and calculate normalized popularity
        freqs = list(self.universe.word_freq.values())
        min_freq = min(freqs) if freqs else 0
        max_freq = max(freqs) if freqs else 1
        freq = self.universe.word_freq.get(word, min_freq)
        
        # Avoid division by zero
        if max_freq == min_freq:
            normalized = 0.5
        else:
            normalized = (freq - min_freq) / (max_freq - min_freq)
        
        # Create smooth heat map transition
        if normalized < 0.33:  # Blue range
            # Dark blue (0,0,139) to Cyan (0,255,255)
            blue = 139 + int((255-139) * normalized * 3)
            green = int(255 * normalized * 3)
            red = 0
        elif normalized < 0.66:  # Green range
            # Cyan (0,255,255) to Yellow (255,255,0)
            green = 255
            red = int(255 * (normalized - 0.33) * 3)
            blue = 255 - int(255 * (normalized - 0.33) * 3)
        else:  # Yellow to White range
            # Yellow (255,255,0) to White (255,255,255)
            red = 255
            green = 255
            blue = int(255 * (normalized - 0.66) * 3)
        
        return f"\033[38;2;{red};{green};{blue}m"

    def print_tree(self, node, depth=0, is_last=True, prefix='', node_to_str=None):
        if node_to_str is None:
            node_to_str = str
        if node is None:
            return
            
        # Handle both dict and list structures
        if isinstance(node, dict):
            node_id = node.get('node', '?')
            children = node.get('children', [])
        elif isinstance(node, list):
            node_id = node[0] if node else '?'
            children = node[1:] if len(node) > 1 else []
        else:
            node_id = str(node)
            children = []
            
        display_str = node_to_str(node_id)
        
        # Get color based on popularity
        color_code = self.get_popularity_color(display_str)
        
        # Print current node
        connectors = {
            'branch': "├── ",
            'last': "└── ",
            'vertical': "│   ",
            'space': "    "
        }
        
        if depth == 0:
            print(f"{color_code}{display_str}{self.RESET}")
        else:
            new_prefix = prefix + (connectors['space'] if is_last else connectors['vertical'])
            print(f"{prefix}{connectors['last' if is_last else 'branch']}{color_code}{display_str}{self.RESET}")
        
        # Print children
        child_count = len(children)
        for i, child in enumerate(children):
            is_last_child = (i == child_count - 1)
            self.print_tree(child, depth+1, is_last_child, 
                           prefix + (connectors['space'] if is_last else connectors['vertical']), 
                           node_to_str)

    def get_word_by_index(self, idx):
        """Get word by index for tree display"""
        for entry in self.universe.word_occurrences:
            if entry["word_index"] == idx:
                return entry["clean_word"]
        return str(idx)

if __name__ == "__main__":
    generator = LanguageGenerator()
    generator.start()
