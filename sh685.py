import multilingus3 as ml3
import qstruct
import random
import difflib
import json
import os
import re
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.spatial import Delaunay
from datetime import datetime
from collections import defaultdict, Counter
import shp295
import shutil
import argparse
import sys
import math
import requests

# NEW: Import boolean processor
from boolean_processor import TruthTableEvaluator, MultiLanguageWikiFetcher

class EnhancedLanguageGenerator(ml3.LanguageGenerator):
    def __init__(self):
        self.initialize_files_from_defaults()
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
        self.CYAN = "\033[36m"
        self.RESET = "\033[0m"
        self.MACHINE_PROMPT = "\033[1;31m"
        self.MACHINE_RESPONSE = "\033[1;32m"
        self.question_struct = {
            'words': set(),
            'phrases': set(),
            'patterns': set()
        }
        self.satisfaction_level = 0.5
        self.visualization_count = 0
        self.mind_tree = self.load_mind_tree()
        self.recent_words = []
        self.recent_pairs = set()
        
        # NEW: Wiki fetcher and truth evaluator
        self.wiki_fetcher = MultiLanguageWikiFetcher()
        self.truth_evaluator = TruthTableEvaluator()
        
        self.recursion_depth = 3
        self.length_scaling = 0.5
        
        self.shp_nlp_processor = shp295.NeuroLinguisticProcessor()
        self.shp_neuromorph = shp295.NeuroMorphicProcessor()
        self.shp_hex_grid = shp295.HexagramGrid()
        self.hexagram_predictions = {}
        self.current_hexagram = None
        self.shp_resources = [
            "words.json", "phrases.json", "mind.json", "parameters.json",
            "minus.json", "plus.json", "./def/words.json", "./def/phrases.json",
            "./def/mind.json", "./def/minus.json", "./def/plus.json"
        ]
        self.similar_chars = {
            'a': 'aeiouàáâãäåā', 'b': 'bp', 'c': 'ckqs', 'd': 'dt',
            'e': 'aeiouèéêëē', 'f': 'fv', 'g': 'gj', 'h': 'h',
            'i': 'aeiouìíîïī', 'j': 'gj', 'k': 'ckq', 'l': 'lr',
            'm': 'n', 'n': 'm', 'o': 'aeiouòóôõöøō', 'p': 'b',
            'q': 'ck', 'r': 'l', 's': 'cz', 't': 'd',
            'u': 'aeiouùúûüū', 'v': 'f', 'w': 'v', 'x': 'z',
            'y': 'ij', 'z': 's',
            'A': 'AEIOUÀÁÂÃÄÅĀ', 'B': 'P', 'C': 'KQS', 'D': 'T',
            'E': 'AEIOUÈÉÊËĒ', 'F': 'V', 'G': 'J', 'H': 'H',
            'I': 'AEIOUÌÍÎÏĪ', 'J': 'G', 'K': 'CQ', 'L': 'R',
            'M': 'N', 'N': 'M', 'O': 'AEIOUÒÓÔÕÖØŌ', 'P': 'B',
            'Q': 'CK', 'R': 'L', 'S': 'CZ', 'T': 'D',
            'U': 'AEIOUÙÚÛÜŪ', 'V': 'F', 'W': 'V', 'X': 'Z',
            'Y': 'IJ', 'Z': 'S'
        }
        
        self.load_resources()
        self._trim_conversation_log()
    
    def _trim_conversation_log(self, max_lines=500):
        if os.path.exists('conversation.log'):
            try:
                with open('conversation.log', 'r') as f:
                    lines = f.readlines()
                if len(lines) > max_lines:
                    with open('conversation.log', 'w') as f:
                        f.writelines(lines[-max_lines:])
            except:
                pass

    # ========== NEW: Definition fetching (skip commands) ==========
    def fetch_and_learn_definitions(self, user_input, verbose=True):
        cmd = user_input.lower().strip()
        # Commands that should skip definition fetching
        skip_commands = ['help', 'commands', '?', 'quit', 'visualize mind', 'print mind', 'print mind full', 'print all', 'params']
        if cmd in skip_commands:
            return
        if cmd.startswith('set ') or cmd.startswith('correct "') or cmd.startswith('no,') or cmd.startswith('no '):
            return
        words = list(set(re.findall(r'\b\w{3,}\b', user_input.lower())))[:10]
        if not words:
            return
        if verbose:
            print(f"{self.BLUE}Fetching definitions for: {', '.join(words)}{self.RESET}")
        defs = self.wiki_fetcher.fetch_multiple_definitions(words)
        for w, dlist in defs.items():
            for d in dlist[:2]:
                if d:
                    self.universe.update_with_sentence(f"{w}: {d}")
            if verbose:
                for i, d in enumerate(dlist[:2]):
                    print(f"{self.BLUE}  {w} [{i+1}]: {d}{self.RESET}")
        self.save_resources()

    # ========== NEW: Truth evaluation with iterative improvement ==========
    def evaluate_response_truth(self, response, iterative=True):
        sentences = re.split(r'(?<=[.!?])\s+', response)
        true_sentences = []
        truth_outputs = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 5:
                continue
            has_logic = any(op in sent.lower() for group in TruthTableEvaluator.OPERATORS.values() for op in group)
            if has_logic:
                if iterative:
                    table, true_rows = self.truth_evaluator.iterative_improve(sent, max_iterations=3)
                else:
                    table, true_rows = self.truth_evaluator.process_sentence(sent)
                if table:
                    truth_outputs.append(f"\n[{sent[:50]}...]\n{table}")
                    if true_rows:
                        true_sentences.append(sent)
            else:
                true_sentences.append(sent)
        if truth_outputs:
            print(f"\n{self.YELLOW}=== TRUTH TABLE ANALYSIS ==={self.RESET}")
            for out in truth_outputs:
                print(out)
        if len(true_sentences) > 1:
            print(f"\n{self.CYAN}=== TRUE SENTENCE SIMILARITIES ==={self.RESET}")
            for i in range(len(true_sentences)):
                for j in range(i+1, len(true_sentences)):
                    sim = difflib.SequenceMatcher(None, true_sentences[i], true_sentences[j]).ratio()
                    print(f"  S{i+1} ↔ S{j+1}: {sim*100:.1f}%")
        return true_sentences

    def select_best_response(self, candidate_responses, true_sentences, original_words):
        best_resp = None
        best_score = -1
        for resp in candidate_responses:
            total_sim = 0
            for ref in true_sentences:
                sim = difflib.SequenceMatcher(None, resp, ref).ratio()
                total_sim += sim
            orig_sim = sum(1 for w in original_words if w in resp.lower()) / max(1, len(original_words))
            avg_sim = (total_sim / max(1, len(true_sentences))) * 0.7 + orig_sim * 0.3
            if avg_sim > best_score:
                best_score = avg_sim
                best_resp = resp
        if best_resp:
            print(f"{self.CYAN}Selected response with {best_score*100:.1f}% combined similarity.{self.RESET}")
        return best_resp if best_resp else (candidate_responses[0] if candidate_responses else "")

    # ========== NEW: Recursive truth evaluation (for recursiveness >=2) ==========
    def recursive_truth_evaluation(self, user_input, initial_response, depth):
        if depth < 2 or self.recursion_depth < 2:
            return initial_response
        
        print(f"\n{self.BLUE_BOLD}=== RECURSIVE TRUTH EVALUATION (Pass 2) ==={self.RESET}")
        second_input = initial_response
        original_words = set(re.findall(r'\b\w+\b', user_input.lower()))
        
        # Fetch additional definitions based on the response
        self.fetch_and_learn_definitions(second_input, verbose=True)
        
        # Generate new candidate responses using the same pipeline but without recursion
        shp_input = re.sub(r'[?!.,]', '', second_input).lower()
        shp_resp = shp295.generate_response(shp_input, self.shp_nlp_processor,
                                            self.shp_neuromorph, self.shp_hex_grid,
                                            self.shp_resources)
        if shp_resp.startswith("Response:"):
            shp_resp = shp_resp.replace("Response:", "").strip()
        
        related = self.find_related_concepts(second_input)
        noisy_inputs = self.add_gaussian_noise(second_input)
        new_responses = []
        for ni in noisy_inputs[:5]:
            try:
                aug = f"{ni} {shp_resp} {' '.join(related)}"
                self.universe.update_with_sentence(aug)
                r = self.universe.generate_sentence()
                new_responses.append(r)
            except:
                new_responses.append(self.universe.generate_sentence())
        
        # Evaluate truth of all candidate responses (including original)
        all_candidates = [initial_response] + new_responses
        all_true_sentences = []
        for cand in all_candidates:
            true_sents = self.evaluate_response_truth(cand, iterative=True)
            all_true_sentences.extend(true_sents)
        
        unique_true = list(dict.fromkeys(all_true_sentences))
        best_response = self.select_best_response(all_candidates, unique_true, original_words)
        
        return best_response

    # ========== ORIGINAL METHODS (from sh671) ==========
    def generate_visualization(self):
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
            except:
                return []
        def prepare_data(self):
            self.prepare_nodes()
            self.prepare_edges()
            self.prepare_phrase_networks()
        def prepare_nodes(self):
            connection_strengths = defaultdict(int)
            for node in self.mind:
                source = node['word_index']
                for target in node.get('connect_indexes', []):
                    key = tuple(sorted([source, target]))
                    connection_strengths[key] += 1
            num_nodes = len(self.words)
            base_angles = np.linspace(0, 4 * np.pi, num_nodes)
            base_radii = np.linspace(0.3, 1.5, num_nodes)
            z_values = np.linspace(-1, 1, num_nodes)
            position_adjustments = {}
            for i, word in enumerate(self.words):
                idx = word['word_index']
                position_adjustments[idx] = {'angle':0,'radius':0,'z':0,'connections':0}
            for (a,b), strength in connection_strengths.items():
                if a in position_adjustments and b in position_adjustments:
                    adj = strength * 0.1
                    position_adjustments[a]['angle'] -= adj
                    position_adjustments[b]['angle'] += adj
                    position_adjustments[a]['radius'] += adj * 0.3
                    position_adjustments[b]['radius'] += adj * 0.3
                    position_adjustments[a]['connections'] += 1
                    position_adjustments[b]['connections'] += 1
            for i, word in enumerate(self.words):
                idx = word['word_index']
                pop = word.get('popularity-%',1)
                norm_pop = max(0.1, min(pop/100,1.0))
                conn = position_adjustments[idx]['connections']
                angle = base_angles[i] + position_adjustments[idx]['angle'] + np.random.uniform(-0.5,0.5)
                radius = base_radii[i] + position_adjustments[idx]['radius'] + np.random.uniform(-0.2,0.2)
                z = z_values[i] + position_adjustments[idx]['z'] + np.random.uniform(-0.3,0.3)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                self.node_positions[idx] = {
                    'pos': (x,y,z),
                    'size': norm_pop * 200,
                    'popularity': norm_pop,
                    'word': word['clean_word'],
                    'connections': conn
                }
        def prepare_edges(self):
            connection_counts = defaultdict(int)
            for node in self.mind:
                src = node['word_index']
                for tgt in node.get('connect_indexes', []):
                    key = tuple(sorted([src,tgt]))
                    connection_counts[key] += 1
            maxc = max(connection_counts.values(), default=1)
            for (src,tgt), cnt in connection_counts.items():
                if src in self.node_positions and tgt in self.node_positions:
                    spos = self.node_positions[src]['pos']
                    tpos = self.node_positions[tgt]['pos']
                    strength = cnt / maxc
                    curve = 0.3 + 0.4 * strength
                    mid = (
                        (spos[0]+tpos[0])/2 + np.random.uniform(-0.2,0.2)*curve,
                        (spos[1]+tpos[1])/2 + np.random.uniform(-0.2,0.2)*curve,
                        (spos[2]+tpos[2])/2 + np.random.uniform(-0.1,0.1)*curve
                    )
                    self.edge_connections.append({
                        'source':src, 'target':tgt,
                        'source_pos':spos, 'target_pos':tpos,
                        'midpoint':mid, 'strength':strength
                    })
        def prepare_phrase_networks(self):
            for phrase in self.phrases:
                indices = phrase.get('word_indices', [])
                pos = [self.node_positions[idx]['pos'] for idx in indices if idx in self.node_positions]
                if len(pos) > 2:
                    pts = np.array(pos) + np.random.uniform(-0.1,0.1, (len(pos),3))
                    try:
                        hull = Delaunay(pts)
                        self.phrase_networks.append({
                            'points':pts, 'hull':hull,
                            'color': np.random.rand(3),
                            'alpha': 0.1 + np.random.random()*0.1
                        })
                    except:
                        continue
        def create_bezier_curve(self, src, tgt, mid, num=10):
            curve = []
            for t in np.linspace(0,1,num):
                rand = 0.05 * np.random.random()
                x = (1-t)**2*src[0] + 2*(1-t)*t*mid[0] + t**2*tgt[0] + rand
                y = (1-t)**2*src[1] + 2*(1-t)*t*mid[1] + t**2*tgt[1] + rand
                z = (1-t)**2*src[2] + 2*(1-t)*t*mid[2] + t**2*tgt[2] + rand
                curve.append([x,y,z])
            return np.array(curve)
        def visualize(self, filename):
            fig = plt.figure(figsize=(16,12), facecolor='black')
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('black')
            fig.patch.set_facecolor('black')
            for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
                pane.fill = False
                pane.set_edgecolor('dimgray')
            ax.grid(False)
            ax.set_xlabel('X', color='lightgray')
            ax.set_ylabel('Y', color='lightgray')
            ax.set_zlabel('Z', color='lightgray')
            ax.tick_params(colors='lightgray')
            self.draw_phrase_networks(ax)
            self.draw_edges(ax)
            self.draw_nodes(ax)
            plt.title("Linguistic Network Visualization", color='white', fontsize=16)
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig)
        def draw_phrase_networks(self, ax):
            for phr in self.phrase_networks:
                try:
                    ax.plot_trisurf(phr['points'][:,0], phr['points'][:,1], phr['points'][:,2],
                                    triangles=phr['hull'].simplices, color=phr['color'],
                                    alpha=phr['alpha'], edgecolor=(*phr['color'],0.3), linewidth=0.7)
                except:
                    continue
        def draw_edges(self, ax):
            colors = plt.cm.plasma(np.linspace(0,1,len(self.edge_connections)))
            for i, e in enumerate(self.edge_connections):
                curve = self.create_bezier_curve(e['source_pos'], e['target_pos'], e['midpoint'])
                ax.plot(curve[:,0], curve[:,1], curve[:,2], color=colors[i],
                        linewidth=1.0+3.0*e['strength'], alpha=0.7)
                direc = curve[-1] - curve[-2]
                direc /= np.linalg.norm(direc)
                ax.quiver(*curve[-1], *direc, color=colors[i], length=0.15, arrow_length_ratio=0.3)
        def draw_nodes(self, ax):
            conns = [n['connections'] for n in self.node_positions.values()]
            maxc = max(conns) if conns else 1
            norm = mcolors.Normalize(vmin=0, vmax=maxc)
            cmap = plt.cm.viridis
            for node in self.node_positions.values():
                x,y,z = node['pos']
                col = cmap(norm(node['connections']))
                ax.scatter([x],[y],[z], s=node['size'], c=[col], edgecolors='white', alpha=0.85)
                ax.text(x,y,z+0.07, node['word'], color='lightgray', fontsize=8, ha='center', alpha=0.7)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
            cbar.set_label('Connection Strength', color='white')
            cbar.ax.yaxis.set_tick_params(color='lightgray')
            cbar.outline.set_edgecolor('lightgray')
            plt.setp(cbar.ax.axes.get_yticklabels(), color='lightgray')

    def call_qstruct(self, user_input):
        verbosity = self.universe.parameters["verbosity"]
        if not hasattr(self, 'qstruct_detector'):
            self.qstruct_detector = qstruct.UniversalQuestionDetector(verbosity=verbosity)
            if verbosity > 10:
                print(f"{self.BLUE_BOLD}Initialized universal question detector{self.RESET}")
        self.qstruct_detector.verbosity = verbosity
        if user_input.endswith('?') or self.qstruct_detector.is_question(user_input):
            self.qstruct_detector.update_patterns(user_input, is_question=True)
            responses = self.qstruct_detector.get_responses(user_input)
            return self.avoid_repetition(random.choice(responses))
        else:
            self.qstruct_detector.update_patterns(user_input, is_question=False)
            return None

    def generate_token_changes(self, word):
        return ''.join('○' if random.random()< (0.2 if i in (1,3,5) else 0.3) else '-' for i in range(6))

    def update_hexagram_predictions(self):
        top = []
        for lvl in ['letters','pairs','triplets']:
            if self.token_analysis.get(lvl):
                top.extend(self.token_analysis[lvl].most_common(min(64, len(self.token_analysis[lvl]))))
        top.sort(key=lambda x: -x[1])
        top = top[:64]
        self.hexagram_predictions = {}
        for i, (sym, _) in enumerate(shp295.HEXAGRAMS[:len(top)]):
            tok, _ = top[i]
            sent = self.get_word_sentiment(tok)
            col = 'green' if sent>0.3 else ('red' if sent<-0.3 else 'yellow')
            self.hexagram_predictions[sym] = {'token': tok, 'color': col, 'changes': self.generate_token_changes(tok)}

    def display_hexagram_grid(self):
        if not self.hexagram_predictions:
            print(f"{self.BLUE_BOLD}Subconscious Processes: No hexagram predictions available{self.RESET}")
            return
        print(f"\n{self.BLUE_BOLD}Hexagram Mind Matrix (Predictive Tokens):{self.RESET}")
        syms = list(self.hexagram_predictions.keys())
        for i in range(0, 64, 8):
            row = []
            for s in syms[i:i+8]:
                d = self.hexagram_predictions[s]
                col = self.GREEN if d['color']=='green' else (self.RED if d['color']=='red' else self.YELLOW)
                row.append(f"{col}{s} {d['changes']}{self.RESET}")
            print("  ".join(row))

    def evaluate_prediction_accuracy(self, user_input):
        if not self.current_hexagram or not self.hexagram_predictions:
            return
        d = self.hexagram_predictions.get(self.current_hexagram)
        if not d:
            return
        token = d['token']
        if token not in user_input:
            return
        cnt = user_input.count(token)
        exp = sum(1 for c in d['changes'] if c=='○')
        if abs(cnt-exp)<=1:
            if d['color']=='yellow':
                d['color']='green'
            elif d['color']=='red':
                d['color']='yellow'
        else:
            if d['color']=='green':
                d['color']='yellow'
            elif d['color']=='yellow':
                d['color']='red'

    def load_mind_tree(self):
        if os.path.exists('mind.json'):
            try:
                with open('mind.json','r') as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except:
                return {}
        return {}

    def save_mind_tree(self):
        try:
            with open('mind.json','w') as f:
                json.dump(self.mind_tree, f, indent=2)
        except:
            pass

    def update_mind_tree(self, concept, related):
        if not isinstance(self.mind_tree, dict):
            self.mind_tree = {}
        if concept not in self.mind_tree:
            self.mind_tree[concept] = []
        if related not in self.mind_tree[concept]:
            self.mind_tree[concept].append(related)
        self.save_mind_tree()

    def traverse_mind_tree(self, concept, depth=1):
        if depth > self.recursion_depth or concept not in self.mind_tree:
            return []
        results = []
        for r in self.mind_tree[concept]:
            results.append(r)
            results.extend(self.traverse_mind_tree(r, depth+1))
        return results

    def find_related_concepts(self, user_input):
        words = user_input.lower().split()
        related = set()
        for w in words:
            if w in self.mind_tree:
                related.update(self.mind_tree[w])
                related.update(self.traverse_mind_tree(w))
            for c in self.mind_tree:
                if c in w or w in c:
                    related.update(self.mind_tree[c])
        return list(related)

    def adjust_parameters_based_on_satisfaction(self):
        pass

    def load_resources(self):
        try:
            if os.path.exists('corrections.json'):
                with open('corrections.json','r') as f:
                    data = json.load(f)
                    for orig, clist in data.items():
                        comb = defaultdict(int)
                        for c,w in clist:
                            comb[c] += w
                        self.corrections[orig] = [(c,w) for c,w in comb.items()]
            if os.path.exists('plus.json'):
                with open('plus.json','r') as f:
                    d = json.load(f)
                    if isinstance(d, list):
                        self.positive_tokens = set(d)
            if os.path.exists('minus.json'):
                with open('minus.json','r') as f:
                    d = json.load(f)
                    if isinstance(d, list):
                        self.negative_tokens = set(d)
            if os.path.exists('conversation.log'):
                with open('conversation.log','r') as f:
                    self.conversation_log = [l.strip() for l in f.readlines()]
                    self.analyze_conversation()
            if os.path.exists('qstructdetect.json'):
                with open('qstructdetect.json','r') as f:
                    data = json.load(f)
                    self.question_struct = {
                        'words': set(data.get('words',[])),
                        'phrases': set(data.get('phrases',[])),
                        'patterns': set(data.get('patterns',[]))
                    }
            if os.path.exists('dynamic_parameters.json'):
                with open('dynamic_parameters.json','r') as f:
                    p = json.load(f)
                    self.recursion_depth = p.get('recursion_depth',3)
                    self.length_scaling = p.get('length_scaling',0.5)
        except Exception as e:
            pass

    def save_resources(self):
        try:
            with open('corrections.json','w') as f:
                dedup = {}
                for orig, clist in self.corrections.items():
                    comb = defaultdict(int)
                    for c,w in clist:
                        comb[c] += w
                    dedup[orig] = [(c,w) for c,w in comb.items()]
                json.dump(dedup, f, indent=2)
            with open('plus.json','w') as f:
                json.dump(list(self.positive_tokens), f, indent=2)
            with open('minus.json','w') as f:
                json.dump(list(self.negative_tokens), f, indent=2)
            with open('conversation.log','a') as f:
                for entry in self.conversation_log[-100:]:
                    f.write(entry+'\n')
            with open('qstructdetect.json','w') as f:
                json.dump({
                    'words': list(self.question_struct['words']),
                    'phrases': list(self.question_struct['phrases']),
                    'patterns': list(self.question_struct['patterns'])
                }, f, indent=2)
            with open('dynamic_parameters.json','w') as f:
                json.dump({
                    'recursion_depth': self.recursion_depth,
                    'length_scaling': self.length_scaling
                }, f, indent=2)
        except Exception as e:
            pass

    def analyze_conversation(self):
        if not self.conversation_log:
            return
        text = " ".join(self.conversation_log).lower()
        levels = {'letters': r'[a-z]', 'pairs': r'[a-z]{2}', 'triplets': r'[a-z]{3}', 'words': r'\b\w+\b'}
        for lvl, pat in levels.items():
            self.token_analysis[lvl] = Counter(re.findall(pat, text))

    def get_word_sentiment(self, word):
        wl = word.lower()
        if wl in self.positive_tokens:
            return 1.0
        if wl in self.negative_tokens:
            return -1.0
        pos, neg, tot = 0,0,0
        for n in range(1,len(wl)+1):
            for i in range(len(wl)-n+1):
                tok = wl[i:i+n]
                tot += 1
                if tok in self.positive_tokens:
                    pos += 1
                if tok in self.negative_tokens:
                    neg += 1
        return (pos-neg)/tot if tot else 0.0

    def print_with_sentiment(self, text):
        if not text:
            return
        text = text.strip()
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        if not any(text.endswith(p) for p in '.!?'):
            text += '.'
        words = text.split()
        print(f"{self.MACHINE_PROMPT}> ", end='', flush=True)
        for i, w in enumerate(words):
            sent = self.get_word_sentiment(w)
            if sent > 0.3:
                col = self.GREEN
            elif sent < -0.3:
                col = self.RED
            else:
                col = self.YELLOW
            end = ' ' if i < len(words)-1 else '\n'
            print(f"{col}{w}{self.RESET}", end=end, flush=True)
        print()

    def update_sentiment_resources(self, text, is_positive):
        words = re.findall(r'\b\w+\b', text.lower())
        for w in words:
            for n in range(1, len(w)+1):
                for i in range(len(w)-n+1):
                    tok = w[i:i+n]
                    (self.positive_tokens if is_positive else self.negative_tokens).add(tok)
        self.save_resources()

    def print_heatmap(self, user_input, responses):
        all_words = user_input.split()
        for r in responses:
            all_words.extend(r.split())
        wf = Counter(w.lower() for w in all_words)
        if not wf:
            return
        maxf = max(wf.values())
        print(f"{self.BLUE_BOLD}Subconscious Processes: {self.RESET}", end='')
        for w, f in wf.most_common():
            norm = f / maxf
            if norm < 0.33:
                b = 128 + int(127 * norm * 3)
                col = f"\033[38;2;0;0;{b}m"
            elif norm < 0.66:
                i = int(128 * (norm-0.33)*3)
                col = f"\033[38;2;{i};{i};255m"
            else:
                i = 128 + int(127 * (norm-0.66)*3)
                col = f"\033[38;2;{i};{i};255m"
            print(f"{col}{w}{self.RESET} ", end='')
        print()

    def update_question_structure(self, input_str):
        clean = re.sub(r'[^\w\s]', '', input_str.lower())
        words = clean.split()
        self.question_struct['words'].update(words)
        for n in range(2,6):
            for i in range(len(words)-n+1):
                self.question_struct['phrases'].add(' '.join(words[i:i+n]))
        self.question_struct['patterns'].add(clean)
        self.save_resources()

    def detect_and_reformat_question(self, response):
        rl = response.lower()
        for pat in sorted(self.question_struct['patterns'], key=len, reverse=True):
            if pat and pat in rl:
                idx = rl.find(pat)
                if idx >= 0:
                    before = response[:idx].strip()
                    qp = response[idx:idx+len(pat)].strip()
                    return f"{before}. {qp}?" if before else f"{qp}?"
        for phrase in sorted(self.question_struct['phrases'], key=len, reverse=True):
            if phrase and phrase in rl:
                idx = rl.find(phrase)
                if idx >= 0:
                    before = response[:idx].strip()
                    qp = response[idx:idx+len(phrase)].strip()
                    return f"{before}. {qp}?" if before else f"{qp}?"
        return response

    def correct_word_in_files(self, old, new):
        for fname in [f for f in os.listdir('.') if f.endswith('.json')]:
            try:
                with open(fname,'r') as f:
                    data = json.load(f)
                def repl(obj):
                    if isinstance(obj, dict):
                        return {repl(k): repl(v) for k,v in obj.items()}
                    elif isinstance(obj, list):
                        return [repl(i) for i in obj]
                    elif isinstance(obj, str):
                        return obj.replace(old, new)
                    return obj
                updated = repl(data)
                with open(fname,'w') as f:
                    json.dump(updated, f, indent=2)
            except:
                pass

    def avoid_repetition(self, response):
        words = response.split()
        new = []
        pairs = set()
        for i, w in enumerate(words):
            if w in self.recent_words:
                continue
            if i>0:
                p = (words[i-1], w)
                if p in self.recent_pairs:
                    continue
                pairs.add(p)
            new.append(w)
        self.recent_words = words[:10]
        self.recent_pairs = pairs
        return ' '.join(new) if new else response

    def combined_sim(self, s1, s2):
        return (0.15*self.char_sequence_similarity(s1,s2) +
                0.20*self.char_pair_similarity(s1,s2) +
                0.25*self.char_triplet_similarity(s1,s2) +
                0.20*self.word_sequence_similarity(s1,s2) +
                0.20*self.phrase_similarity(s1,s2))

    def multifractal_similarity(self, correction):
        if not self.token_analysis or not correction:
            return 0.0
        lf = self.token_analysis.get('letters', Counter())
        pf = self.token_analysis.get('pairs', Counter())
        tf = self.token_analysis.get('triplets', Counter())
        total_let = sum(lf.values())+len(lf)
        total_pair = sum(pf.values())+len(pf)
        total_trip = sum(tf.values())+len(tf)
        score = 1.0
        for i,ch in enumerate(correction.lower()):
            if not ch.isalpha():
                continue
            cp = (lf.get(ch,0)+1)/total_let if total_let else 0
            pp = 1.0
            if i>0 and i<len(correction):
                p = correction[i-1:i+1].lower()
                if all(c.isalpha() for c in p):
                    pp = (pf.get(p,0)+1)/total_pair if total_pair else 0
            tp = 1.0
            if i>1 and i<len(correction)-1:
                t = correction[i-2:i+1].lower()
                if all(c.isalpha() for c in t):
                    tp = (tf.get(t,0)+1)/total_trip if total_trip else 0
            score *= (cp*0.4 + pp*0.3 + tp*0.3)
        return score**(1/len(correction)) if correction else 0.0

    def add_gaussian_noise(self, s, noise=0.1):
        noisy = []
        words = s.split()
        for _ in range(37):
            ns = []
            for w in words:
                if len(w)<=2 or random.random()>=noise:
                    ns.append(w)
                else:
                    nc = []
                    for j,c in enumerate(w):
                        if j==0 or j==len(w)-1:
                            nc.append(c)
                        else:
                            sim = self.similar_chars.get(c, c)
                            if isinstance(sim, str):
                                sim = sim.replace('m','').replace('M','')
                                nc.append(random.choice(sim) if sim else c)
                            else:
                                nc.append(c)
                    ns.append(''.join(nc))
            noisy.append(' '.join(ns))
        return noisy

    def char_sequence_similarity(self, a,b):
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    def char_pair_similarity(self, a,b):
        p1 = set(a[i:i+2] for i in range(len(a)-1))
        p2 = set(b[i:i+2] for i in range(len(b)-1))
        u = len(p1|p2)
        return len(p1&p2)/u if u else 0.0
    def char_triplet_similarity(self, a,b):
        t1 = set(a[i:i+3] for i in range(len(a)-2))
        t2 = set(b[i:i+3] for i in range(len(b)-2))
        u = len(t1|t2)
        return len(t1&t2)/u if u else 0.0
    def word_sequence_similarity(self, a,b):
        return difflib.SequenceMatcher(None, a.lower().split(), b.lower().split()).ratio()
    def phrase_similarity(self, a,b):
        seq = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
        w1 = set(a.lower().split())
        w2 = set(b.lower().split())
        jac = len(w1&w2)/len(w1|w2) if (w1|w2) else 0.0
        return seq*0.7 + jac*0.3

    def initialize_files_from_defaults(self):
        defaults = {
            'words.json': './def/words.json',
            'phrases.json': './def/phrases.json',
            'corrections.json': './def/corrections.json',
            'mind.json': './def/mind.json',
            'dynamic_parameters.json': './def/dynamic_parameters.json'
        }
        for tgt, src in defaults.items():
            if (not os.path.exists(tgt) or os.path.getsize(tgt)==0) and os.path.exists(src):
                shutil.copyfile(src, tgt)

    def loop_conversation(self, seed="Hello", cycles=100):
        print(f"{self.MACHINE_RESPONSE}Starting loop conversation mode{self.RESET}")
        print(f"{self.MACHINE_RESPONSE}Seed input: {seed}{self.RESET}")
        cur = seed
        for c in range(cycles):
            print(f"\n{self.BLUE_BOLD}=== CYCLE {c+1} ==={self.RESET}")
            resp = self.process_input_internal(cur)
            clean = re.sub(r'^Response:\s*', '', resp).strip()
            print(f"{self.MACHINE_RESPONSE}Response: {clean}{self.RESET}")
            cur = clean
            time.sleep(1)

    def process_input_internal(self, inp):
        from io import StringIO
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            self.process_input(inp)
            out = sys.stdout.getvalue()
            for line in out.splitlines():
                if line.startswith(self.MACHINE_PROMPT + "> "):
                    return line.replace(self.MACHINE_PROMPT + "> ", "").strip()
            return ""
        finally:
            sys.stdout = old

    # ========== MODIFIED process_input (with full command handling) ==========
    def process_input(self, user_input):
        # ------------------------------------------------------------------
        # ORIGINAL COMMAND HANDLING (must come first, before any fetching)
        # ------------------------------------------------------------------
        # Handle visualization command
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

        # Handle help command
        if user_input.lower() in ['help', 'commands', '?']:
            self.show_help()
            return

        # Handle print mind
        if user_input.lower() in ['print mind', 'print mind full']:
            self.print_mind_tree_comprehensive()
            return

        # Handle print all
        if user_input.lower() == 'print all':
            self.print_all_analysis()
            return

        # Handle params
        if user_input.lower() == 'params':
            self.show_parameters()
            return

        # Handle correct command
        if user_input.startswith('correct "') and '" to "' in user_input:
            parts = user_input.split('"')
            if len(parts) >= 5:
                old_word = parts[1]
                new_word = parts[3]
                self.correct_word_in_files(old_word, new_word)
                response = f'Corrected "{old_word}" to "{new_word}" in all resources'
                self.conversation_log.append(f"System: {response}")
                self.print_with_sentiment(response)
                return

        # ------------------------------------------------------------------
        # If we reach here, it's a regular conversation input
        # ------------------------------------------------------------------
        # Fetch definitions (skip for commands already handled)
        self.fetch_and_learn_definitions(user_input, verbose=True)

        # Original hexagram and question processing
        self.evaluate_prediction_accuracy(user_input)
        self.update_hexagram_predictions()
        if self.hexagram_predictions:
            self.current_hexagram = random.choice(list(self.hexagram_predictions.keys()))
            hex_data = self.hexagram_predictions[self.current_hexagram]
            if self.universe.parameters["verbosity"] > 10:
                print(f"{self.BLUE_BOLD}Subconscious Processes: Active hexagram {self.current_hexagram} - {hex_data['token']} ({hex_data['color']}){self.RESET}")
        self.display_hexagram_grid()

        # Question handling
        if user_input.endswith('?') or any(word in self.question_struct['words'] for word in user_input.lower().split()):
            if user_input.endswith('?'):
                clean_input = user_input.rstrip('?').strip()
                self.update_question_structure(clean_input)
            words = re.findall(r'\b\w+\b', user_input.lower())
            for word in words:
                self.token_analysis['words'][word] += 1
                self.token_analysis['question_words'][word] += 1
            question_words = []
            if self.token_analysis['question_words']:
                total_questions = sum(self.token_analysis['question_words'].values())
                for word, count in self.token_analysis['question_words'].items():
                    general_freq = self.token_analysis['words'].get(word, 1)
                    distinctiveness = count / (general_freq + 1)
                    if distinctiveness > 0.7 and count > total_questions * 0.1:
                        question_words.append(word)
            content_words = [w for w in words if len(w) > 2 and w not in question_words]

        # shp295 response
        shp_input = re.sub(r'[?!.,]', '', user_input).lower()
        shp_response = shp295.generate_response(
            shp_input,
            self.shp_nlp_processor,
            self.shp_neuromorph,
            self.shp_hex_grid,
            self.shp_resources
        )
        if shp_response.startswith("Response:"):
            shp_response = shp_response.replace("Response:", "").strip()

        self.universe.process_parameters()
        self.conversation_log.append(f"User: {user_input}")
        if shp_response:
            self.conversation_log.append(f"Online: {shp_response}")

        words = user_input.split()
        is_negation = False
        negation_words = ['No', 'no', 'Ei', 'ei', 'Nein', 'nein', 'Non', 'non']

        # Negation handling
        if words and words[0] in negation_words:
            is_negation = True
            correction_phrase = ' '.join(words[1:])
            if self.last_original:
                existing_index = -1
                existing_weight = 0
                for i, (corr, weight) in enumerate(self.corrections[self.last_original]):
                    if corr == correction_phrase:
                        existing_index = i
                        existing_weight = weight
                        break
                if self.last_original == correction_phrase:
                    new_weight = existing_weight * 2 if existing_weight > 0 else 2
                else:
                    new_weight = existing_weight + 3
                if existing_index >= 0:
                    self.corrections[self.last_original][existing_index] = (correction_phrase, new_weight)
                else:
                    self.corrections[self.last_original].append((correction_phrase, new_weight))
                response = "Correction stored."
                if new_weight > 3:
                    response += f" (Weight: {new_weight})"
                if self.universe.parameters["verbosity"] > 0:
                    response += f"\nOriginal: {self.last_original}\nCorrection: {correction_phrase}"
                self.update_sentiment_resources(self.last_original, is_positive=False)
                self.update_sentiment_resources(correction_phrase, is_positive=True)
                self.satisfaction_level = max(0.1, self.satisfaction_level - 0.1)
                self.conversation_log.append(f"System: {response}")
                self.print_with_sentiment(response)
                self.save_resources()
                self.universe.update_with_sentence(user_input)
                self.auto_input_counter += 1
                return

        # Update language model
        if ' ' in user_input:
            self.universe.update_with_sentence(user_input)
            words = user_input.split()
            if len(words) > 1:
                for i in range(len(words) - 1):
                    self.update_mind_tree(words[i], words[i+1])
        else:
            self.universe.update_with_word(user_input)

        if not is_negation:
            self.last_original = user_input
            self.satisfaction_level = min(0.9, self.satisfaction_level + 0.05)

        # Generate initial candidate responses
        noisy_inputs = self.add_gaussian_noise(user_input)
        responses = []
        related = self.find_related_concepts(user_input)
        base_length = len(user_input.split())
        target_length = max(3, int(base_length * self.length_scaling))
        for ni in noisy_inputs:
            try:
                aug = f"{ni} {shp_response} {' '.join(related)}"
                response = self.universe.generate_sentence(aug)
                if len(response.split()) < target_length:
                    additional = self.universe.generate_sentence(' '.join(related))
                    response = f"{response} {additional}"
                responses.append(response)
            except:
                responses.append(self.universe.generate_sentence())

        # Apply corrections
        weights = [1.0] * len(responses)
        matching_corrections = []
        best_correction = None
        best_weight = 0
        if not is_negation and self.corrections:
            merged_corrections = defaultdict(lambda: defaultdict(int))
            for orig, corrs in self.corrections.items():
                for corr, weight in corrs:
                    merged_corrections[orig][corr] += weight
            for orig, corr_dict in merged_corrections.items():
                combined_sim_score = self.combined_sim(user_input, orig)
                if combined_sim_score >= 0.6666:
                    for corr, stored_weight in corr_dict.items():
                        candidate_weight = combined_sim_score * stored_weight * 2000
                        matching_corrections.append((corr, candidate_weight))
                        if candidate_weight > best_weight:
                            best_weight = candidate_weight
                            best_correction = corr
        if best_correction:
            top_corrections = [c for c, w in matching_corrections if w == best_weight]
            if len(top_corrections) > 1:
                best_multifractal = -1
                for corr in top_corrections:
                    mf_score = self.multifractal_similarity(corr)
                    if mf_score > best_multifractal:
                        best_multifractal = mf_score
                        best_correction = corr
            responses.append(best_correction)
            weights.append(best_weight * 100)
        else:
            for corr, weight in matching_corrections:
                responses.append(corr)
                weights.append(weight)

        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w/total_weight for w in weights]
            selected_index = random.choices(range(len(responses)), weights=normalized_weights, k=1)[0]
        else:
            selected_index = random.randint(0, len(responses)-1)
        selected_response = responses[selected_index]

        # Apply question reformatting and repetition avoidance
        formatted_response = self.detect_and_reformat_question(selected_response)
        formatted_response = self.avoid_repetition(formatted_response)

        # Handle verbosity
        if self.universe.parameters.get("verbosity", 0) == 99:
            self.print_heatmap(user_input, responses)
        elif self.universe.parameters.get("verbosity", 0) > 30:
            if best_correction and selected_response == best_correction:
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
                orig = next((orig for orig, corrs in self.corrections.items()
                           if any(corr == selected_response for corr, _ in corrs)), None)
                if orig:
                    verbose = f"Selected correction: '{orig}' → '{selected_response}'"
                else:
                    verbose = f"Selected generated response: '{selected_response}'"
            else:
                verbose = f"Selected generated response: '{selected_response}'"
            print(f"{self.BLUE_BOLD}Subconscious Processes: {self.BLUE}{verbose}{self.RESET}")

            # Question handling via qstruct
            qstruct_response = self.call_qstruct(user_input)
            if qstruct_response:
                formatted_response = self.detect_and_reformat_question(qstruct_response)
                self.conversation_log.append(f"System: {formatted_response}")
                self.print_with_sentiment(formatted_response)
                self.auto_input_counter += 1
                return

        # NEW: Recursive truth evaluation if recursion_depth >= 2
        if self.recursion_depth >= 2:
            final_response = self.recursive_truth_evaluation(user_input, formatted_response, self.recursion_depth)
        else:
            final_response = formatted_response

        self.conversation_log.append(f"System: {final_response}")
        self.print_with_sentiment(final_response)
        self.auto_input_counter += 1
        self.analyze_conversation()
        self.save_resources()

    # ========== Additional methods needed for original commands ==========
    def print_mind_tree_comprehensive(self):
        if not self.mind_tree:
            print(f"{self.YELLOW}Mind tree is empty{self.RESET}")
            return
        print(f"\n{self.BLUE_BOLD}╔══════════════════════════════════════════════════════════╗{self.RESET}")
        print(f"{self.BLUE_BOLD}║               COMPREHENSIVE MIND TREE                     ║{self.RESET}")
        print(f"{self.BLUE_BOLD}╚══════════════════════════════════════════════════════════╝{self.RESET}")
        word_conn = {w: len(conn) for w, conn in self.mind_tree.items()}
        for w, cnt in sorted(word_conn.items(), key=lambda x: -x[1])[:30]:
            col = self.GREEN if cnt > 10 else (self.YELLOW if cnt > 5 else self.RESET)
            print(f"\n{col}● {w} ({cnt} connections){self.RESET}")
            conns = sorted(set(self.mind_tree[w]))
            for i, c in enumerate(conns[:8]):
                pref = "    ├── " if i < min(7, len(conns)-1) else "    └── "
                print(f"{pref}{c}")
            if len(conns) > 8:
                print(f"    └── ... and {len(conns)-8} more")

    def print_all_analysis(self):
        print(f"\n{self.BLUE_BOLD}LINGUISTIC ANALYSIS:{self.RESET}")
        print(f"  Words in vocabulary: {len(self.universe.word_freq)}")
        print(f"  Phrases learned: {len(self.universe.phrase_occurrences)}")
        print(f"  Total connections: {sum(len(v) for v in self.universe.word_connections.values())}")
        print(f"  Conversation entries: {len(self.conversation_log)}")
        mem = self.memory_system.get_memory_stats()
        print(f"\n{self.CYAN}MEMORY STATISTICS:{self.RESET}")
        print(f"  Working: {mem['working_memory_size']}/7")
        print(f"  Short-term: {mem['short_term_size']}/50")
        print(f"  Long-term: {mem['long_term_size']} words")
        if self.universe.word_freq:
            print(f"\n{self.YELLOW}TOP WORDS:{self.RESET}")
            for w, c in sorted(self.universe.word_freq.items(), key=lambda x: -x[1])[:20]:
                print(f"  {w}: {c}")

    def show_parameters(self):
        print(f"\n{self.COMMAND}CURRENT PARAMETERS:{self.RESET}")
        for param, value in self.universe.parameters.items():
            bar_length = 20
            filled = int(bar_length * value / 100)
            bar = '█' * filled + '-' * (bar_length - filled)
            print(f"{self.COMMAND}{param.capitalize().ljust(15)}: {self.RESET}{bar} {value:.1f}%")

    def show_help(self):
        help_txt = f"""
{self.BLUE_BOLD}╔══════════════════════════════════════════════════════════╗
║         ENHANCED LANGUAGE GENERATOR v11.0                 ║
║      with Truth Tables & Multi-Language Wiki               ║
╚══════════════════════════════════════════════════════════╝{self.RESET}

{self.GREEN}CORE COMMANDS:{self.RESET}
  help                 - Show this help
  quit                 - Exit and save data
  params               - Show parameter settings

{self.GREEN}VISUALIZATION:{self.RESET}
  visualize mind       - Generate 3D mind visualization

{self.GREEN}ORIGINAL COMMANDS:{self.RESET}
  print mind           - Print mind tree
  print all            - Full analysis
  No, <correction>     - Teach desired output
  correct "old" "new"  - Replace word in all files
  set recursiveness=X  - Set recursion depth (1-10)
  set length_scaling=X - Set response length scaling (0.1-2.0)

{self.GREEN}EXAMPLES:{self.RESET}
  > Hello
  > If it is raining then the ground is wet.
  > What is the capital of France?
        """
        print(help_txt)

    def start(self):
        print(f"{self.MACHINE_RESPONSE}M U L T I L I N G U S v. 3.2.2 – Enhanced Memory & Truth Tables{self.RESET}")
        print(f"{self.COMMAND}Type 'help' for commands{self.RESET}")
        while True:
            try:
                if self.auto_input_counter >= random.randint(3,5):
                    auto = self.universe.generate_auto_input()
                    if ' ' in auto:
                        self.universe.update_with_sentence(auto)
                    else:
                        self.universe.update_with_word(auto)
                    self.auto_input_counter = 0
                    print(f"{self.MACHINE_RESPONSE}>> Auto-learned: {auto}{self.RESET}")
                ui = input(f"{self.PROMPT}> {self.USER_INPUT}").strip()
                print(self.RESET, end='')
                if ui.lower() == 'quit':
                    self.universe.save_logs()
                    print(f"{self.MACHINE_RESPONSE}Exiting and saving data.{self.RESET}")
                    break
                elif ui:
                    self.process_input(ui)
            except KeyboardInterrupt:
                print(f"\n{self.MACHINE_RESPONSE}Exiting and saving data.{self.RESET}")
                self.universe.save_logs()
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--loop', action='store_true')
    parser.add_argument('-s','--seed', default='Hello')
    parser.add_argument('-c','--cycles', type=int, default=100)
    parser.add_argument('-v','--visualize', action='store_true')
    args = parser.parse_args()
    gen = EnhancedLanguageGenerator()
    print(f"\n{gen.BLUE_BOLD}╔══════════════════════════════════════════════════════════╗{gen.RESET}")
    print(f"{gen.BLUE_BOLD}║         ENHANCED LANGUAGE GENERATOR v11.0                 ║{gen.RESET}")
    print(f"{gen.BLUE_BOLD}║      with Truth Tables & Multi-Language Wiki              ║{gen.RESET}")
    print(f"{gen.BLUE_BOLD}╚══════════════════════════════════════════════════════════╝{gen.RESET}")
    print(f"\n{gen.GREEN}Type '{gen.YELLOW}help{gen.GREEN}' for commands | Truth tables auto-evaluated{gen.RESET}")
    if args.visualize:
        print(gen.generate_advanced_visualization())
    if args.loop:
        gen.loop_conversation(args.seed, args.cycles)
    else:
        gen.start()
