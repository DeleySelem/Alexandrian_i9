
import sh663
import ml63 as ml3
import random
import json
import os
import time
import numpy as np
from collections import defaultdict

class CascadeProcessor(sh663.EnhancedLanguageGenerator):
    def __init__(self):
        super().__init__()
        self.hexagram_grid = self.load_hexagram_grid()
        self.word_token_cache = {}

    def load_hexagram_grid(self):
        """Load sentiment mappings from hexagram grid"""
        try:
            if os.path.exists('hexagram.json'):
                with open('hexagram.json', 'r') as f:
                    return json.load(f)
        except:
            return {}
        return {}

    # ADDED MISSING TYPEWRITE METHOD
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

    def get_token_set(self, word):
        """Get token set for a word (cached)"""
        word_lower = word.lower()
        if word_lower in self.word_token_cache:
            return set(self.word_token_cache[word_lower])

        tokens = set()
        n_min = 1
        n_max = min(3, len(word_lower))
        for n in range(n_min, n_max+1):
            for i in range(len(word_lower) - n + 1):
                tokens.add(word_lower[i:i+n])

        self.word_token_cache[word_lower] = list(tokens)
        return tokens

    def token_weight(self, token):
        """Calculate token weight based on length"""
        token_len = len(token)
        if token_len == 1: return 0.1
        elif token_len == 2: return 0.2
        elif token_len == 3: return 0.3
        else: return 0.1

    def get_hexagram_sentiment(self, word):
        """Get hexagram-based sentiment boost (0-1)"""
        return self.hexagram_grid.get(word.lower(), 0.5)

    def apply_correction_pattern(self, user_input, orig, corr):
        """Apply correction pattern with mathematical awareness"""
        input_words = user_input.split()
        orig_words = orig.split()
        corr_words = corr.split()

        # Mathematical pattern detection
        math_ops = {'plus', 'minus', 'times', 'divided'}
        if (len(orig_words) >= 4 and len(corr_words) >= 4 and
            any(op in orig_words for op in math_ops) and
            'is' in corr_words):

            verb_idx = next((i for i, w in enumerate(input_words)
                           if w in {'is', 'are'}), -1)
            op_idx = next((i for i, w in enumerate(input_words)
                         if w in math_ops), -1)

            if verb_idx != -1 and op_idx != -1:
                return f"{input_words[verb_idx-1]} {input_words[op_idx]} " \
                       f"{input_words[op_idx+1]} is {input_words[-1].rstrip('?')}"

        # Standard pattern application
        response_words = []
        min_len = min(len(input_words), len(orig_words))

        for i in range(max(len(input_words), len(corr_words))):
            if i < min_len and input_words[i] == orig_words[i] and i < len(corr_words):
                response_words.append(corr_words[i])
            elif i < len(input_words):
                response_words.append(input_words[i])
            elif i < len(corr_words):
                response_words.append(corr_words[i])

        return " ".join(response_words)

    def cascade_processing(self, user_input):
        """Process input through three weighted cascades"""
        cascades = {
            'exact': {'weight': 0.1, 'candidates': []},
            'pattern': {'weight': 0.2, 'candidates': []},
            'mind_tree': {'weight': 0.7, 'candidates': []}
        }

        # Cascade 1: Exact match in corrections.json
        if user_input in self.corrections:
            for corr, weight in self.corrections[user_input]:
                cascades['exact']['candidates'].append({
                    'response': corr,
                    'weight': weight
                })

        # Cascade 2: Pattern matching
        for orig, corrs in self.corrections.items():
            orig_words = orig.split()
            input_words = user_input.split()

            if len(orig_words) != len(input_words):
                continue

            pattern_strength = sum(1 for i in range(len(orig_words))
                                  if orig_words[i] == input_words[i])

            if pattern_strength >= 2:  # Require at least 2 matching words
                best_corr = max(corrs, key=lambda x: x[1])[0]
                cascades['pattern']['candidates'].append({
                    'response': self.apply_correction_pattern(user_input, orig, best_corr),
                    'weight': pattern_strength / len(orig_words)
                })

        # Cascade 3: Mind tree traversal
        tokens = self.get_token_set(user_input.lower())
        if tokens:
            try:
                # Find subject word with highest token weight
                subject_word = max(tokens, key=lambda t: self.token_weight(t) * len(t))

                if subject_word and subject_word in self.mind_tree:
                    for concept in self.mind_tree[subject_word]:
                        cascades['mind_tree']['candidates'].append({
                            'response': f"{subject_word} {concept}",
                            'weight': self.token_weight(subject_word)
                        })
            except Exception as e:
                if self.universe.parameters.get("verbosity", 0) > 50:
                    print(f"Mind tree error: {e}")

        return cascades

    def weighted_selection(self, cascades):
        """Select response with weighted probabilities"""
        all_candidates = []

        for cascade_name, cascade_data in cascades.items():
            base_weight = cascade_data['weight']

            for candidate in cascade_data['candidates']:
                # Apply Gaussian noise
                noise = max(0.01, random.gauss(1, 0.1))

                # Apply hexagram sentiment boost
                words = candidate['response'].split()
                sentiment = np.mean([self.get_hexagram_sentiment(w)
                                   for w in words]) if words else 0.5

                # Calculate final weight
                final_weight = base_weight * candidate['weight'] * noise * (1 + sentiment)
                all_candidates.append((candidate['response'], final_weight, cascade_name))

        if all_candidates:
            total_weight = sum(w for _, w, _ in all_candidates)
            weights = [w/total_weight for _, w, _ in all_candidates]
            return random.choices(all_candidates, weights=weights, k=1)[0]

        # Fallback to generated sentence
        return (self.universe.generate_sentence(), 1.0, 'fallback')

    def handle_negation(self, user_input):
        """Handle negation commands (e.g., 'no correction_phrase')"""
        words = user_input.split()
        negation_words = ['No', 'no', 'Ei', 'ei', 'Nein', 'nein', 'Non', 'non']

        if words and words[0] in negation_words:
            correction_phrase = ' '.join(words[1:])

            if self.last_original:
                # Update corrections
                new_corrections = []
                total_weight = 0
                updated = False

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

                self.corrections[self.last_original] = [
                    (corr, weight/total_weight)
                    for corr, weight in new_corrections
                ]

                # Build response
                response = "Correction stored."
                if self.universe.parameters["verbosity"] > 0:
                    response += f"\nOriginal: {self.last_original}"
                    response += f"\nCorrection: {correction_phrase}"

                # Update sentiment resources
                self.update_sentiment_resources(self.last_original, False)
                self.update_sentiment_resources(correction_phrase, True)

                # Print response
                self.conversation_log.append(f"System: {response}")
                self.typewrite_text(response)

                # Save and update
                self.save_resources()
                self.universe.update_with_sentence(user_input)
                self.auto_input_counter += 1
                return True

        return False

    def process_input(self, user_input):
        """Enhanced processing with cascade visualization"""
        # Handle negation first
        if self.handle_negation(user_input):
            return

        # Log user input
        self.conversation_log.append(f"User: {user_input}")
        self.universe.update_with_sentence(user_input)
        self.last_original = user_input

        # Process through cascades
        cascades = self.cascade_processing(user_input)
        response, weight, source = self.weighted_selection(cascades)

        # Display cascade details for high verbosity
        verbosity = self.universe.parameters.get("verbosity", 0)
        if verbosity == 100:
            print("\n" + "="*60)
            print("CASCADE PROCESSING REPORT")
            print("="*60)
            print(f"USER INPUT: {user_input}")

            for cascade_name, cascade_data in cascades.items():
                print(f"\nCASCADE: {cascade_name.upper()} (Base Weight: {cascade_data['weight']})")

                if not cascade_data['candidates']:
                    print("  No candidates generated")
                    continue

                for idx, candidate in enumerate(cascade_data['candidates'], 1):
                    print(f"  {idx}. {candidate['response']}")
                    print(f"    Weight: {candidate['weight']:.4f}")

            print("\n" + "="*60)
            print("FINAL SELECTION")
            print("="*60)
            print(f"Response: {response}")
            print(f"Weight: {weight:.4f} | Source: {source}")
            print("="*60 + "\n")

        # Print final response and update state
        self.typewrite_text(response)
        self.conversation_log.append(f"System: {response}")
        self.universe.update_with_sentence(response)
        self.save_resources()

if __name__ == "__main__":
    generator = CascadeProcessor()
    generator.start()