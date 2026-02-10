import multilingus3 as ml3
import qstruct
import random
import difflib
import json
import os                                                         import re
import time                                                       import numpy as np
import matplotlib                                                 matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt                                   from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors                               from scipy.spatial import Delaunay
from datetime import datetime                                     from collections import defaultdict, Counter
import shp295                                                     import shutil
import argparse 
                                                  
class EnhancedLanguageGenerator(ml3.LanguageGenerator):               def __init__(self):                                                   self.initialize_files_from_defaults()                             super().__init__()                                                self.corrections = defaultdict(list)
        self.last_original = None                                         self.positive_tokens = set()                                      self.negative_tokens = set()
        self.conversation_log = []                                        self.token_analysis = defaultdict(lambda: defaultdict(int))                                                                         self.BLUE_BOLD = "\033[1;34m"                                     self.BLUE = "\033[34极"
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
        self.satisfaction_level = 0.5  # 0-1 scale
        self.visualization_count = 0  # Added for visualization
        self.load_resources()
        # Initialize shp295 components
        self.shp_nlp_processor = shp295.NeuroLinguisticProcessor()
        self.shp_neuromorph = shp295.NeuroMorphicProcessor()      
        self.shp_hex_grid = shp295.HexagramGrid()
        self.hexagram_predictions = {}  # Store hexagram predictions
        self.current_hexagram = None    # Track current hexagram
        self.shp_resources = [
            "words.json",
            "phrases.json",
            "mind.json",
            "parameters.json",
            "minus.json",
            "plus.json",
            "./def/words.json",
            "./def/phrases.json",
            "./def/mind.json",
            "./def/minus.json",
            "./def/plus.json"
        ]
        # Track recent words and pairs to avoid repetition
        self.recent_words = []
        self.recent_pairs = set()
        # Character similarity mapping for Gaussian noise
        self.similar_chars = {
            'a': 'aeiouàáâãäåā',
            'b': 'bp',
            'c': 'ckqs',
            'd': 'dt',
            'e': 'aeiouèéêëē',
            'f': 'fv',
            'g': 'gj',
            'h': 'h',
            'i': 'aeiouìíîïī',
            'j': 'gj',
            'k': 'ckq',
            'l': 'lr',
            'm': 'n',  # Only allow 'n' as similar to 'm'
            'n': 'm',  # Only allow 'm' as similar to 'n'
            'o': 'aeiouòóôõöøō',
            'p': 'b',
            'q': 'ck',
            'r': 'l',
            's': 'cz',
            't': 'd',
            'u': 'aeiouùúûüū',
            'v': 'f',
            'w': 'v',
            'x': 'z',
            'y': 'ij',
            'z': 's',
            'A': 'AEIOUÀÁÂÃÄÅĀ',
            'B': 'P',
            'C': 'KQS',
            'D': 'T',
            'E': 'AEIOUÈÉÊËĒ',
            'F': 'V',
            'G': 'J',
            'H': 'H',
            'I': 'AEIOUÌÍÎÏĪ',
            'J': 'G',
            'K': 'CQ',
            'L': 'R',
            'M': 'N',  # Only allow 'N' as similar to 'M'
            'N': 'M',  # Only allow 'M' as similar to 'N'
            'O': 'AEIOUÒÓÔÕÖØŌ',
            'P': 'B',
            'Q': 'CK',
            'R': 'L',
            'S': 'CZ',
            'T': 'D',
            'U': 'AEIOUÙÚÛÜŪ',
            'V': 'F',
            'W': 'V',
            'X': 'Z',
            'Y': 'IJ',
            'Z': 'S'
        }
        # Load mind tree
        self.mind_tree = self.load_mind_tree()
        # Dynamic parameters
        self.recursion_depth = 3  # Default recursion depth
        self.length_scaling = 0.5  # Default length scaling

    def generate_visualization(self):
        """Generate and save linguistic visualization without popping up window"""
        try:
            self.visualization_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mind_visualization_{timestamp}_{self.visualization_count}.png"

            visualizer = self.LinguisticVisualizer(
                words_file='words.json',
                phrases_file='phrases.json',
                mind_file='mind.json'
            )
            visualizer.prepare_data()
            visualizer.visualize(filename)
            return f"Visualization saved as {filename}"
        except Exception as e:
            return f"Error generating visualization: {str(e)}"

    class LinguisticVisualizer:
        """Integrated visualization class with randomization based on word relations"""
        def __init__(self, words_file='words.json', phrases_file='phrases.json', mind_file='mind.json'):
            self.words = self.load_json(words_file)
            self.phrases = self.load_json(phrases_file)
            self.mind = self.load_json(mind_file)
            self.node_positions = {}
            self.edge_connections = []
            self.phrase_networks = []

        def load_json(self, filename):
            try:
                with open(filename, 'r') as f:
                    return json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                return []

        def prepare_data(self):
            """Prepare data with randomization based on word relations"""
            self.prepare_nodes()
            self.prepare_edges()
            self.prepare_phrase_networks()

        def prepare_nodes(self):
            """Position nodes with randomization based on relations"""
            # Calculate connection strengths
            connection_strengths = defaultdict(int)
            for node in self.mind:
                source = node['word_index']
                for target in node.get('connect_indexes', []):
                    key = tuple(sorted([source, target]))
                    connection_strengths[key] += 1

            # Create positions with relation-based randomization
            num_nodes = len(self.words)
            base_angles = np.linspace(0, 4 * np.pi, num_nodes)
            base_radii = np.linspace(0.3, 1.5, num_nodes)
            z_values = np.linspace(-1, 1, num_nodes)

            # Position adjustment based on connections
            position_adjustments = {}
            for i, word in enumerate(self.words):
                idx = word['word_index']
                position_adjustments[idx] = {
                    'angle': 0,
                    'radius': 0,
                    'z': 0,
                    'connections': 0
                }

            # Apply relation-based randomization
            for (a, b), strength in connection_strengths.items():
                if a in position_adjustments and b in position_adjustments:
                    # Create attraction between related nodes
                    adjustment = strength * 0.1
                    position_adjustments[a]['angle'] -= adjustment
                    position_adjustments[b]['angle'] += adjustment
                    position_adjustments[a]['radius'] += adjustment * 0.3
                    position_adjustments[b]['radius'] += adjustment * 0.3
                    position_adjustments[a]['connections'] += 1
                    position_adjustments[b]['connections'] += 1

            # Create final positions
            for i, word in enumerate(self.words):
                idx = word['word_index']
                popularity = word.get('popularity-%', 1)
                normalized_popularity = max(0.1, min(popularity / 100, 1.0))
                connections = position_adjustments[idx]['connections']

                # Apply base position + relation adjustments + randomization
                angle = base_angles[i] + position_adjustments[idx]['angle'] + np.random.uniform(-0.5, 0.5)
                radius = base_radii[i] + position_adjustments[idx]['radius'] + np.random.uniform(-0.2, 0.2)
                z = z_values[i] + position_adjustments[idx]['z'] + np.random.uniform(-0.3, 0.3)

                x = radius * np.cos(angle)
                y = radius * np.sin(angle)

                self.node_positions[idx] = {
                    'pos': (x, y, z),
                    'size': normalized_popularity * 200,
                    'popularity': normalized_popularity,
                    'word': word['clean_word'],
                    'connections': connections
                }

        def prepare_edges(self):
            """Prepare edges with relation-based curvature"""
            connection_counts = defaultdict(int)
            for node in self.mind:
                source = node['word_index']
                for target in node.get('connect_indexes', []):
                    key = tuple(sorted([source, target]))
                    connection_counts[key] += 1

            max_count = max(connection_counts.values(), default=1)

            for (source, target), count in connection_counts.items():
                if source in self.node_positions and target in self.node_positions:
                    source_pos = self.node_positions[source]['pos']
                    target_pos = self.node_positions[target]['pos']

                    # Create midpoint with curvature based on connection strength
                    strength = count / max_count
                    curve_factor = 0.3 + 0.4 * strength
                    midpoint = (
                        (source_pos[0] + target_pos[0]) / 2 + np.random.uniform(-0.2, 0.2) * curve_factor,
                        (source_pos[1] + target_pos[1]) / 2 + np.random.uniform(-0.2, 0.2) * curve_factor,
                        (source_pos[2] + target_pos[2]) / 2 + np.random.uniform(-0.1, 0.1) * curve_factor
                    )

                    self.edge_connections.append({
                        'source': source,
                        'target': target,
                        'source_pos': source_pos,
                        'target_pos': target_pos,
                        'midpoint': midpoint,
                        'strength': strength
                    })

        def prepare_phrase_networks(self):
            """Prepare phrase networks with randomization"""
            for phrase in self.phrases:
                word_indices = phrase.get('word_indices', [])
                positions = [self.node_positions[idx]['pos'] for idx in word_indices if idx in self.node_positions]

                if len(positions) > 2:
                    points = np.array(positions)

                    # Add randomization to hull points
                    randomized_points = points + np.random.uniform(-0.1, 0.1, points.shape)

                    try:
                        hull = Delaunay(randomized_points)
                        self.phrase_networks.append({
                            'points': randomized_points,
                            'hull': hull,
                            'color': np.random.rand(3,),
                            'alpha': 0.1 + np.random.random() * 0.1
                        })
                    except:
                        continue

        def create_bezier_curve(self, source, target, midpoint, num_points=10):
            """Create Bézier curve with randomized control"""
            curve = []
            for t in np.linspace(0, 1, num_points):
                # Add randomization to curve path
                rand_factor = 0.05 * np.random.random()
                x = (1-t)**2*source[0] + 2*(1-t)*t*midpoint[0] + t**2*target[0] + rand_factor
                y = (1-t)**2*source[1] + 2*(1-t)*t*midpoint[1] + t**2*target[1] + rand_factor
                z = (1-t)**2*source[2] + 2*(1-t)*t*midpoint[2] + t**2*target[2] + rand_factor
                curve.append([x, y, z])
            return np.array(curve)

        def visualize(self, filename):
            """Save visualization to file without displaying"""
            fig = plt.figure(figsize=(16, 12), facecolor='black')
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('black')

            # Configure dark theme
            fig.patch.set_facecolor('black')
            ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('dimgray')
            ax.yaxis.pane.set_edgecolor('dimgray')
            ax.zaxis.pane.set_edgecolor('dimgray')
            ax.grid(False)
            ax.set_xlabel('X', color='lightgray')
            ax.set_ylabel('Y', color='lightgray')
            ax.set_zlabel('Z', color='lightgray')
            ax.tick_params(colors='lightgray')

            # Draw elements
            self.draw_phrase_networks(ax)
            self.draw_edges(ax)
            self.draw_nodes(ax)

            plt.title("Linguistic Network Visualization", color='white', fontsize=16)
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig)

        def draw_phrase_networks(self, ax):
            """Draw phrase networks"""
            for phrase in self.phrase_networks:
                points = phrase['points']
                try:
                    ax.plot_trisurf(
                        points[:, 0], points[:, 1], points[:, 2],
                        triangles=phrase['hull'].simplices,
                        color=phrase['color'],
                        alpha=phrase['alpha'],
                        edgecolor=(*phrase['color'], 0.3),
                        linewidth=0.7
                    )
                except:
                    continue

        def draw_edges(self, ax):
            """Draw edges as Bézier curves"""
            edge_colors = plt.cm.plasma(np.linspace(0, 1, len(self.edge_connections)))

            for i, edge in enumerate(self.edge_connections):
                curve = self.create_bezier_curve(
                    edge['source_pos'],
                    edge['target_pos'],
                    edge['midpoint']
                )

                # Draw curve
                ax.plot(
                    curve[:, 0], curve[:, 1], curve[:, 2],
                    color=edge_colors[i],
                    linewidth=1.0 + 3.0 * edge['strength'],
                    alpha=0.7
                )

                # Draw arrow
                direction = curve[-1] - curve[-2]
                direction /= np.linalg.norm(direction)
                ax.quiver(
                    *curve[-1], *direction,
                    color=edge_colors[i],
                    length=0.15,
                    arrow_length_ratio=0.3
                )

        def draw_nodes(self, ax):
            """Draw nodes with size based on popularity"""
            # Create colormap for connection count
            connections = [node['connections'] for node in self.node_positions.values()]
            max_conn = max(connections) if connections else 1
            norm = mcolors.Normalize(vmin=0, vmax=max_conn)
            cmap = plt.cm.viridis

            for node in self.node_positions.values():
                x, y, z = node['pos']
                color = cmap(norm(node['connections']))

                # Draw node
                ax.scatter(
                    [x], [y], [z],
                    s=node['size'],
                    c=[color],
                    edgecolors='white',
                    alpha=0.85
                )

                # Add label
                ax.text(
                    x, y, z + 0.07,
                    node['word'],
                    color='lightgray',
                    fontsize=8,
                    ha='center',
                    alpha=0.7
                )

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
            cbar.set_label('Connection Strength', color='white')
            cbar.ax.yaxis.set_tick_params(color='lightgray')
            cbar.outline.set_edgecolor('lightgray')
            plt.setp(cbar.ax.axes.get_yticklabels(), color='lightgray')

    def call_qstruct(self, user_input):
        """Handle question detection and response using universal patterns"""
        verbosity = self.universe.parameters["verbosity"]

        # Initialize detector on first use
        if not hasattr(self, 'qstruct_detector'):
            self.qstruct_detector = qstruct.UniversalQuestionDetector(verbosity=verbosity)
            if verbosity > 10:
                print(f"{self.BLUE_BOLD}Initialized universal question detector{self.RESET}")

        # Update verbosity level
        self.qstruct_detector.verbosity = verbosity

        # Check if it's a question using universal patterns
        if user_input.endswith('?') or self.qstruct_detector.is_question(user_input):
            # Update the detector with this question
            self.qstruct_detector.update_patterns(user_input, is_question=True)

            # Generate response
            responses = self.qstruct_detector.get_responses(user_input)
            response = random.choice(responses)

            # Apply formatting and return
            return self.avoid_repetition(response)
        else:
            # Update with statement patterns
            self.qstruct_detector.update_patterns(user_input, is_question=False)
        return None

    def generate_token_changes(self, word):
        """Generate token change predictions for a word"""
        changes = []
        for i in range(6):  # 6 lines per hexagram
            # Determine change probability based on token position
            change_prob = 0.3
            if i in [1, 3, 5]:  # Odd positions more stable
                change_prob = 0.2
            changes.append('○' if random.random() < change_prob else '-')
        return ''.join(changes)

    def update_hexagram_predictions(self):
        """Update hexagram predictions based on token frequencies"""
        # Get most significant tokens from analysis
        top_tokens = []
        for level in ['letters', 'pairs', 'triplets']:
            if self.token_analysis[level]:
                # Collect up to 64 tokens from each level
                num_to_collect = min(64, len(self.token_analysis[level]))
                top_tokens.extend(self.token_analysis[level].most_common(num_to_collect))

        # Sort by frequency and select top 64
        top_tokens.sort(key=lambda x: x[1], reverse=True)
        top_tokens = top_tokens[:64]

        # Assign tokens to hexagrams
        self.hexagram_predictions = {}
        for i, (symbol, name) in enumerate(shp295.HEXAGRAMS[:len(top_tokens)]):
            token, freq = top_tokens[i]
            sentiment = self.get_word_sentiment(token)

            # Determine color based on sentiment
            if sentiment > 0.3:
                color = "green"
            elif sentiment < -0.3:
                color = "red"
            else:
                color = "yellow"

            # Generate change predictions
            changes = self.generate_token_changes(token)

            self.hexagram_predictions[symbol] = {
                'token': token,
                'color': color,
                'changes': changes,
                'actual_changes': None
            }

    def display_hexagram_grid(self):
        """Display the latest hexagram grid with predictions"""
        if not self.hexagram_predictions:
            print(f"{self.BLUE_BOLD}Subconscious Processes: No hexagram predictions available{self.RESET}")
            return

        print(f"\n{self.BLUE_BOLD}Hexagram Mind Matrix (Predictive Tokens):{self.RESET}")
        symbols = list(self.hexagram_predictions.keys())

        for i in range(0, 64, 8):
            row_symbols = symbols[i:i+8]
            row_line = []
            for symbol in row_symbols:
                data = self.hexagram_predictions[symbol]
                color_code = self.GREEN if data['color'] == "green" else self.RED if data['color'] == "red" else self.YELLOW
                row_line.append(f"{color_code}{symbol} {data['changes']}{self.RESET}")
            print("  ".join(row_line))

    def evaluate_prediction_accuracy(self, user_input):
        """Evaluate hexagram prediction accuracy against user input"""
        if not self.current_hexagram or not self.hexagram_predictions:
            return

        hex_data = self.hexagram_predictions[self.current_hexagram]
        predicted_changes = hex_data['changes']
        token = hex_data['token']

        # Skip evaluation if token not in input
        if token not in user_input:
            return

        # Compare predicted vs actual token usage
        token_count = user_input.count(token)
        expected_changes = sum(1 for c in predicted_changes if c == '○')

        # Update color based on accuracy
        if abs(token_count - expected_changes) <= 1:
            # Successful prediction - upgrade color
            if hex_data['color'] == "yellow":
                hex_data['color'] = "green"
            elif hex_data['color'] == "red":
                hex_data['color'] = "yellow"
        else:
            # Failed prediction - downgrade color
            if hex_data['color'] == "green":
                hex_data['color'] = "yellow"
            elif hex_data['color'] == "yellow":
                hex_data['color'] = "red"

    def load_mind_tree(self):
        """Load mind tree from JSON file"""
        if os.path.exists('mind.json'):
            try:
                with open('mind.json', 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    else:
                        print(f"{self.MACHINE_RESPONSE}Warning: mind.json has invalid format, expected dict{self.RESET}")
                        return {}
            except Exception as e:
                print(f"{self.MACHINE_RESPONSE}Error loading mind.json: {e}{self.RESET}")
                return {}
        return {}

    def save_mind_tree(self):
        """Save mind tree to JSON file"""
        with open('mind.json', 'w') as f:
            json.dump(self.mind_tree, f, indent=2)

    def update_mind_tree(self, concept, related_concept):
        """Update mind tree with new connection"""
        # Ensure mind_tree is a dictionary
        if not isinstance(self.mind_tree, dict):
            self.mind_tree = {}

        if concept not in self.mind_tree:
            self.mind_tree[concept] = []
        if related_concept not in self.mind_tree[concept]:
            self.mind_tree[concept].append(related_concept)
        self.save_mind_tree()

    def traverse_mind_tree(self, concept, depth=1):
        """Recursively traverse mind tree to find related concepts"""
        if depth > self.recursion_depth or concept not in self.mind_tree:
            return []

        results = []
        for related in self.mind_tree[concept]:
            results.append(related)
            # Recursive traversal
            results.extend(self.traverse_mind_tree(related, depth+1))

        return results

    def find_related_concepts(self, user_input):
        """Find related concepts in mind tree based on user input"""
        words = user_input.lower().split()
        related_concepts = set()

        for word in words:
            # Direct matches
            if word in self.mind_tree:
                related_concepts.update(self.mind_tree[word])
                # Recursive traversal based on recursion_depth parameter
                related_concepts.update(self.traverse_mind_tree(word))

            # Substring matches
            for concept in self.mind_tree.keys():
                if concept in word or word in concept:
                    related_concepts.update(self.mind_tree[concept])

        return list(related_concepts)

    def adjust_parameters_based_on_satisfaction(self):
        """Auto-adjust parameters based on satisfaction level"""
        # Only adjust non-user parameters
        pass

    def load_resources(self):
        """Load all resources from files"""
        try:
            # Load corrections
            if os.path.exists('corrections.json'):
                with open('corrections.json', 'r') as f:
                    data = json.load(f)
                    for orig, corrs in data.items():
                        # Deduplicate corrections and combine weights
                        combined_corrs = defaultdict(int)
                        for corr, weight in corrs:
                            combined_corrs[corr] += weight
                        self.corrections[orig] = [(c, w) for c, w in combined_corrs.items()]

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

            # Load dynamic parameters if they exist
            if os.path.exists('dynamic_parameters.json'):
                with open('dynamic_parameters.json', 'r') as f:
                    params = json.load(f)
                    self.recursion_depth = params.get('recursion_depth', 3)
                    self.length_scaling = params.get('length_scaling', 0.5)

        except Exception as e:
            print(f"{self.MACHINE_RESPONSE}Error loading resources: {e}{self.RESET}")

    def save_resources(self):
        """Save all resources to files"""
        try:
            # Save corrections (deduplicated during save)
            with open('corrections.json', 'w') as f:
                # Deduplicate before saving
                dedup_corrections = {}
                for orig, corrs in self.corrections.items():
                    combined = defaultdict(int)
                    for corr, weight in corrs:
                        combined[corr] += weight
                    dedup_corrections[orig] = [(c, w) for c, w in combined.items()]
                json.dump(dedup_corrections, f, indent=2)

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

            # Save dynamic parameters
            with open('dynamic_parameters.json', 'w') as f:
                json.dump({
                    'recursion_depth': self.recursion_depth,
                    'length_scaling': self.length_scaling
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

    def print_with_sentiment(self, text):
        """Print text with sentiment coloring"""
        if not text:
            return

        # Ensure proper sentence formatting
        text = text.strip()
        if not text[0].isupper():
            text = text[0].upper() + text[1:]
        if not any(text.endswith(p) for p in ['.', '!', '?']):
            text += '.'

        words = text.split()

        # Print with sentiment coloring
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

            # Print word
            end_char = ' ' if i < len(words)-1 else '\n'
            print(f"{color_code}{word}{self.RESET}", end=end_char, flush=True)
        print()

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

    def avoid_repetition(self, response):
        """Avoid repeating words or word pairs from recent responses"""
        words = response.split()
        new_words = []
        word_pairs = set()

        for i, word in enumerate(words):
            # Skip if word was recently used
            if word in self.recent_words:
                continue

            # Skip if word pair was recently used
            if i > 0:
                pair = (words[i-1], word)
                if pair in self.recent_pairs:
                    continue
                word_pairs.add(pair)

            new_words.append(word)

        # Update recent words and pairs
        self.recent_words = words[:10]  # Keep last 10 words
        self.recent_pairs = word_pairs

        return ' '.join(new_words) if new_words else response

    def combined_sim(self, s1, s2):
        """Calculate combined similarity score between two strings"""
        char_sim = self.char_sequence_similarity(s1, s2)
        pair_sim = self.char_pair_similarity(s1, s2)
        triplet_sim = self.char_triplet_similarity(s1, s2)
        word_sim = self.word_sequence_similarity(s1, s2)
        phrase_sim = self.phrase_similarity(s1, s2)
        combined_sim = (
            0.15 * char_sim +
            0.20 * pair_sim +
            0.25 * triplet_sim +
            0.20 * word_sim +
            0.20 * phrase_sim
        )
        return combined_sim

    def multifractal_similarity(self, correction):
        """Calculate multifractal similarity score for correction"""
        if not self.token_analysis or not correction:
            return 0.0

        # Get frequency distributions
        letter_freq = self.token_analysis['letters']
        pair_freq = self.token_analysis['pairs']
        triplet_freq = self.token_analysis['triplets']

        # Calculate total counts with Laplace smoothing
        total_letters = sum(letter_freq.values()) + len(letter_freq)
        total_pairs = sum(pair_freq.values()) + len(pair_freq)
        total_triplets = sum(triplet_freq.values()) + len(triplet_freq)

        # Calculate probabilities
        score = 1.0
        for i in range(len(correction)):
            # Letter probability
            char = correction[i].lower()
            char_count = letter_freq.get(char, 0) + 1
            char_prob = char_count / total_letters

            # Pair probability
            if i > 0:
                pair = correction[i-1:i+1].lower()
                pair_count = pair_freq.get(pair, 0) + 1
                pair_prob = pair_count / total_pairs
            else:
                pair_prob = 1.0

            # Triplet probability
            if i > 1:
                triplet = correction[i-2:i+1].lower()
                triplet_count = triplet_freq.get(triplet, 0) + 1
                triplet_prob = triplet_count / total_triplets
            else:
                triplet_prob = 1.0

            # Combine probabilities (geometric mean)
            score *= (char_prob * 0.4 + pair_prob * 0.3 + triplet_prob * 0.3)

        return score ** (1/len(correction)) if correction else 0.0

    def add_gaussian_noise(self, input_str, noise_factor=0.1):
        """Add Gaussian noise to input string without adding m's to word boundaries"""
        noisy_inputs = []
        words = input_str.split()

        for _ in range(37):
            noisy_sentence = []
            for word in words:
                # Skip noise for short words
                if len(word) <= 2:
                    noisy_sentence.append(word)
                    continue

                # Add noise to each character with probability = noise_factor
                if random.random() < noise_factor:
                    noisy_chars = []
                    for j, char in enumerate(word):
                        # Preserve first and last characters
                        if j == 0 or j == len(word)-1:
                            noisy_chars.append(char)
                        else:
                            similar = self.similar_chars.get(char, char)
                            # Remove 'm' from similar characters for middle positions
                            similar = similar.replace('m', '').replace('M', '')
                            if similar:
                                noisy_chars.append(random.choice(similar))
                            else:
                                noisy_chars.append(char)
                    noisy_word = ''.join(noisy_chars)
                    noisy_sentence.append(noisy_word)
                else:
                    noisy_sentence.append(word)
            noisy_inputs.append(' '.join(noisy_sentence))

        return noisy_inputs

    def initialize_files_from_defaults(self):
        """Copy default files if target files are missing or empty"""
        default_files = {
            'words.json': './def/words.json',
            'phrases.json': './def/phrases.json',
            'corrections.json': './def/corrections.json',
            'mind.json': './def/mind.json',
            'dynamic_parameters.json': './def/dynamic_parameters.json'
        }

        for target, source in default_files.items():
            # Check if target doesn't exist or is empty
            if not os.path.exists(target) or os.path.getsize(target) == 0:
                if os.path.exists(source):
                    shutil.copyfile(source, target)
                    print(f"Copied default {source} to {target}")
                else:
                    print(f"Warning: Default file {source} not found")

    def process_input(self, user_input):
        # Handle visualization command FIRST
        if user_input.strip().lower() == "visualize mind":
            response = self.generate_visualization()
            self.conversation_log.append(f"System: {response}")
            self.print_with_sentiment(response)
            return

        # Handle parameter updates
        if user_input.startswith("set recursiveness="):
            try:
                new_value = float(user_input.split("=")[1])
                if 1 <= new_value <= 10:
                    self.recursion_depth = int(new_value)
                    response = f"Recursion depth set to {self.recursion_depth}"
                    self.conversation_log.append(f"System: {response}")
                    self.print_with_sentiment(response)
                    self.save_resources()
                    return
            except:
                pass

        if user_input.startswith("set length_scaling="):
            try:
                new_value = float(user_input.split("=")[1])
                if 0.1 <= new_value <= 2.0:
                    self.length_scaling = new_value
                    response = f"Length scaling set to {self.length_scaling:.2f}"
                    self.conversation_log.append(f"System: {response}")
                    self.print_with_sentiment(response)
                    self.save_resources()
                    return
            except:
                pass

        # Evaluate previous prediction
        self.evaluate_prediction_accuracy(user_input)

        # Generate new predictions
        self.update_hexagram_predictions()

        # Select random hexagram for this response
        if self.hexagram_predictions:
            self.current_hexagram = random.choice(list(self.hexagram_predictions.keys()))
            hex_data = self.hexagram_predictions[self.current_hexagram]
            if self.universe.parameters["verbosity"] > 10:
                print(f"{self.BLUE_BOLD}Subconscious Processes: Active hexagram {self.current_hexagram} - {hex_data['token']} ({hex_data['color']}){self.RESET}")

        # Display single hexagram grid
        self.display_hexagram_grid()

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

        # Clean input for shp295 - remove punctuation and convert to lowercase
        shp_input = re.sub(r'[?!.,]', '', user_input).lower()

        # Get shp295 response with cleaned input
        shp_response = shp295.generate_response(
            shp_input,
            self.shp_nlp_processor,
            self.shp_neuromorph,
            self.shp_hex_grid,
            self.shp_resources
        )

        # Remove "Response:" prefix if present
        if shp_response.startswith("Response:"):
            shp_response = shp_response.replace("Response:", "").strip()

        # Handle "correct" command
        if user_input.startswith('correct "') and '" to "' in user_input:
            parts = user_input.split('"')
            if len(parts) >= 5:
                old_word = parts[1]
                new_word = parts[3]
                self.correct_word_in_files(old_word, new_word)
                response = f'Corrected "{old_word}" to "{new_word}" in all resources'
                self.conversation_log.append(f"System极 {response}")
                self.print_with_sentiment(response)
                return

        # Process combined input through ML57
        self.universe.process_parameters()
        self.conversation_log.append(f"User: {user_input}")
        if shp_response:
            self.conversation_log.append(f"Online: {shp_response}")

        words = user_input.split()
        is_negation = False
        negation_words = ['No', 'no', 'Ei', 'ei', 'Nein', 'nein', 'Non', 'non']

        # Check for negation commands
        if words and words[0] in negation_words:
            is_negation = True
            correction_phrase = ' '.join(words[1:])

            if self.last_original:
                # Check if this exact correction already exists
                existing_index = -1
                existing_weight = 0
                for i, (corr, weight) in enumerate(self.corrections[self.last_original]):
                    if corr == correction_phrase:
                        existing_index = i
                        existing_weight = weight
                        break

                # If the correction is the same as the input, double the weight
                if self.last_original == correction_phrase:
                    new_weight = existing_weight * 2 if existing_weight > 0 else 2
                else:
                    new_weight = existing_weight + 3  # Stronger weight for corrections

                if existing_index >= 0:
                    # Update existing correction
                    self.corrections[self.last_original][existing_index] = (correction_phrase, new_weight)
                else:
                    # Add new correction
                    self.corrections[self.last_original].append((correction_phrase, new_weight))

                # Build response
                response = "Correction stored."
                if new_weight > 3:  # Show weight for significant corrections
                    response += f" (Weight: {new_weight})"

                # Add correction details
                if self.universe.parameters["verbosity"] > 0:
                    response += f"\nOriginal: {self.last_original}"
                    response += f"\nCorrection: {correction_phrase}"

                # Update sentiment resources
                self.update_sentiment_resources(self.last_original, is_positive=False)
                self.update_sentiment_resources(correction_phrase, is_positive=True)

                # Update satisfaction level (negative feedback)
                self.satisfaction_level = max(0.1, self.satisfaction_level - 0.1)

                # Print response
                self.conversation_log.append(f"System: {response}")
                self.print_with_sentiment(response)

                # Save and update
                self.save_resources()
                self.universe.update_with_sentence(user_input)
                self.auto_input_counter += 1
                return

        # Update language model
        if ' ' in user_input:
            self.universe.update_with_sentence(user_input)

            # Extract concepts and update mind tree
            words = user_input.split()
            if len(words) > 1:
                for i in range(len(words) - 1):
                    self.update_mind_tree(words[i], words[i+1])
        else:
            self.universe.update_with_word(user_input)

        # Store original phrase for potential correction
        if not is_negation:
            self.last_original = user_input
            # Positive feedback for no correction needed
            self.satisfaction_level = min(0.9, self.satisfaction_level + 0.05)

        # Auto-adjust parameters based on satisfaction
        self.adjust_parameters_based_on_satisfaction()

        # Generate responses with Gaussian noise
        noisy_inputs = self.add_gaussian_noise(user_input)
        responses = []

        # Find related concepts from mind tree
        related_concepts = self.find_related_concepts(user_input)
        if related_concepts and self.universe.parameters["verbosity"] > 10:
            print(f"{self.BLUE_BOLD}Subconscious Processes: Related concepts: {', '.join(related_concepts)}{self.RESET}")

        # Determine response length based on parameters
        base_length = len(user_input.split())
        target_length = max(3, int(base_length * self.length_scaling))

        for noisy_input in noisy_inputs:
            try:
                # Augment input with related concepts and shp response
                augmented_input = f"{noisy_input} {shp_response} {' '.join(related_concepts)}"
                response = self.universe.generate_sentence(augmented_input)

                # Adjust response length based on recursiveness
                if len(response.split()) < target_length:
                    # Generate additional related content
                    additional = self.universe.generate_sentence(' '.join(related_concepts))
                    response = f"{response} {additional}"

                responses.append(response)
            except:
                responses.append(self.universe.generate_sentence())

        weights = [1.0] * len(responses)
        # Apply corrections with enhanced weighting and strict threshold
        matching_corrections = []
        best_correction = None
        best_weight = 0
        if not is_negation and self.corrections:
            # Create a merged corrections dictionary to avoid duplicates
            merged_corrections = defaultdict(lambda: defaultdict(int))
            for orig, corrs in self.corrections.items():
                for corr, weight in corrs:
                    merged_corrections[orig][corr] += weight

            for orig, corr_dict in merged_corrections.items():
                # Calculate combined similarity with strict threshold
                combined_sim_score = self.combined_sim(user_input, orig)

                # Only consider corrections with 66.66% or higher similarity
                if combined_sim_score >= 0.6666:
                    for corr, stored_weight in corr_dict.items():
                        # Enhanced weighting: higher weights have more influence
                        candidate_weight = combined_sim_score * stored_weight * 2000
                        matching_corrections.append((corr, candidate_weight))

                        # Track the best correction overall
                        if candidate_weight > best_weight:
                            best_weight = candidate_weight
                            best_correction = corr

                        # Verbose output
                        if self.universe.parameters["verbosity"] > 20 and self.universe.parameters["verbosity"] != 99:
                            details = (
                                f"Match: '{orig}' → '{corr}' "
                                f"(char:{self.char_sequence_similarity(user_input, orig):.2f} pair:{self.char_pair_similarity(user_input, orig):.2f} "
                                f"triplet:{self.char_triplet_similarity(user_input, orig):.2f} word:{self.word_sequence_similarity(user_input, orig):.2f} "
                                f"phrase:{self.phrase_similarity(user_input, orig):.2f} combined:{combined_sim_score:.2f} "
                                f"weight:{candidate_weight:.1f})"
                            )
                            print(f"{self.BLUE_BOLD}Subconscious Processes: {self.BLUE}{details}{self.RESET}")

        # If we have a best correction, use it
        if best_correction:
            # Handle tie-breaking for same weight corrections
            top_corrections = [corr for corr, weight in matching_corrections if weight == best_weight]
            if len(top_corrections) > 1:
                # Use multifractal analysis to break ties
                best_multifractal = -1
                for corr in top_corrections:
                    mf_score = self.multifractal_similarity(corr)
                    if mf_score > best_multifractal:
                        best_multifractal = mf_score
                        best_correction = corr
                if self.universe.parameters["verbosity"] > 20:
                    print(f"{self.BLUE_BOLD}Subconscious Processes: Tie broken with multifractal score: {best_multifractal:.4f}{self.RESET}")

            # Add best correction to responses with high weight
            responses.append(best_correction)
            weights.append(best_weight * 100)  # Ensure strong weighting
        else:
            # Add all matching corrections normally
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
        # Avoid repetition
        formatted_response = self.avoid_repetition(formatted_response)

        # Handle verbosity level 99
        if self.universe.parameters["verbosity"] == 99:
            self.print_heatmap(user_input, responses)

        # Handle other verbosity levels
        elif self.universe.parameters["verbosity"] > 30:
            if best_correction and selected_response == best_correction:
                # Find the original for the best correction
                source_orig = None
                for orig, corrs in self.corrections.items():
                    for corr, _ in corrs:
                        if corr == best_correction:
                            source_orig = orig
                            break
                    if source_orig:
                        break
                if source_orig:
                    verbose = f"Selected best correction: '{source_orig}' → '{best_correction}' (Weight: {best_weight:.1f})"
            elif selected_index >= len(responses) - len(matching_corrections):
                orig = next(orig for orig, corrs in self.corrections.items()
                           if any(corr == selected_response for corr, _ in corrs))
                verbose = f"Selected correction: '{orig}' → '{selected_response}'"
            else:
                verbose = f"Selected generated response: '{selected_response}'"

            print(f"{self.BLUE_BOLD}Subconscious Processes: {self.BLUE}{verbose}{self.RESET}")

            # Handle questions with universal pattern detection
            qstruct_response = self.call_qstruct(user_input)
            if qstruct_response is not None:
                # Apply final formatting
                formatted_response = self.detect_and_reformat_question(qstruct_response)

                # Print and log the response
                self.conversation_log.append(f"System: {formatted_response}")
                self.print_with_sentiment(formatted_response)
                self.auto_input_counter += 1
                return

        # Print final response with sentiment coloring
        self.conversation_log.append(f"System: {formatted_response}")
        self.print_with_sentiment(formatted_response)
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

    # New method for loop conversation mode
    def loop_conversation(self, seed_input="Hello", cycles=100):
        """Run an infinite conversation loop between two instances"""
        print(f"{self.MACHINE_RESPONSE}Starting loop conversation mode{self.RESET}")
        print(f"{self.MACHINE_RESPONSE}Seed input: {seed_input}{self.RESET}")

        current_input = seed_input
        cycle_count = 0

        while cycle_count < cycles:
            cycle_count += 1
            print(f"\n{self.BLUE_BOLD}=== CYCLE {cycle_count} ==={self.RESET}")

            # Process input and get response
            print(f"{self.BLUE}Input: {current_input}{self.RESET}")
            response = self.process_input_internal(current_input)

            # Clean response by removing "Response:" prefix
            clean_response = re.sub(r'^Response:\s*', '', response).strip()

            # Print the cleaned response
            print(f"{self.MACHINE_RESPONSE}Response: {clean_response}{self.RESET}")

            # Set the next input to the cleaned response
            current_input = clean_response
            time.sleep(1)  # Pause between cycles

        print(f"{self.MACHINE_RESPONSE}Completed {cycles} conversation cycles{self.RESET}")

    # Helper method for loop conversation
    def process_input_internal(self, user_input):
        """Process input and return response text without printing"""
        # Create a string buffer to capture output
        from io import StringIO
        import sys

        # Redirect stdout
        original_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Process the input using existing method
            self.process_input(user_input)

            # Get captured output
            captured_output = sys.stdout.getvalue()

            # Extract the machine response line
            response_line = None
            for line in captured_output.splitlines():
                if line.startswith(self.MACHINE_PROMPT + "> "):
                    response_line = line
                    break

            return response_line.replace(self.MACHINE_PROMPT + "> ", "").strip() if response_line else ""
        finally:
            # Restore stdout
            sys.stdout = original_stdout

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Enhanced Language Generator')
    parser.add_argument('-l', '--loop', action='store_true',
                        help='Enable loop conversation mode')
    parser.add_argument('-s', '--seed', type=str, default="Hello",
                        help='Seed input for loop conversation')
    parser.add_argument('-c', '--cycles', type=int, default=100,
                        help='Number of conversation cycles')
    args = parser.parse_args()

    generator = EnhancedLanguageGenerator()

    if args.loop:
        generator.loop_conversation(args.seed, args.cycles)
    else:
        generator.start()