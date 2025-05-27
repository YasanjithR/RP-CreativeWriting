import itertools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import networkx as nx
from transformers import pipeline
import heapq
from functools import lru_cache
import math
import time

class StoryPermutationValidator:
    def __init__(self):
        # Download necessary NLTK resources only if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        # Load spaCy language model - disable unnecessary components for speed
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
        
        # Lazy loading of transformer model
        self._coherence_checker = None
        
        # Stopwords for filtering
        self.stop_words = set(stopwords.words('english'))
        
        # Cache for expensive operations
        self._coherence_scores = {}

    @property
    def coherence_checker(self):
        """Lazy load the transformer model only when needed"""
        if self._coherence_checker is None:
            self._coherence_checker = pipeline('text-classification', 
                                            model='facebook/bart-large-mnli')
        return self._coherence_checker

    def tokenize_story(self, story):
        """Tokenize the story into sentences"""
        return sent_tokenize(story)

    def generate_batch_permutations(self, sentences, batch_size=1000):
        """
        Generate permutations in batches to avoid memory issues
        
        Args:
            sentences (list): List of story sentences
            batch_size (int): Number of permutations to generate at once
            
        Yields:
            list: Batches of permutations
        """
        perm_iter = itertools.permutations(sentences)
        while True:
            batch = list(itertools.islice(perm_iter, batch_size))
            if not batch:
                break
            yield batch

    @lru_cache(maxsize=512)
    def calculate_coherence_score(self, permutation):
        """
        Calculate coherence score for a permutation (combined grammatical and semantic)
        
        Args:
            permutation (tuple): Sequence of sentences
            
        Returns:
            float: Overall coherence score (0-1)
        """
        # Check cache
        if permutation in self._coherence_scores:
            return self._coherence_scores[permutation]
            
        # Calculate grammatical coherence
        gram_score = self.check_grammatical_coherence(permutation)
        
        # Calculate semantic coherence
        sem_score = self.semantic_coherence_check(permutation)
        
        # Weighted combination - semantic is more important for sensibility
        final_score = (gram_score * 0.3) + (sem_score * 0.7)
        
        # Cache the result
        self._coherence_scores[permutation] = final_score
        
        return final_score

    @lru_cache(maxsize=512)
    def check_grammatical_coherence(self, permutation):
        """Check grammatical coherence of a sentence permutation"""
        # Join sentences
        story_text = ' '.join(permutation)
        
        # Process with spaCy
        doc = self.nlp(story_text)
        
        # Analyze structure
        return self._analyze_dependency_structure(doc)

    def _analyze_dependency_structure(self, doc):
        """Analyze dependency structure with optimized graph metrics"""
        # Create dependency graph - use indices for nodes (faster than text)
        G = nx.DiGraph()
        
        # Add important nodes and edges - skip stopwords and punctuation
        for token in doc:
            if not token.is_stop and not token.is_punct and token.text.strip():
                G.add_node(token.i, pos=token.pos_, dep=token.dep_)
                if token.head.i != token.i:
                    G.add_edge(token.head.i, token.i)
        
        # Calculate simplified metrics for speed
        try:
            if len(G) < 3:
                return 0.5
                
            # Calculate node connectivity - faster than clustering coefficient
            connectivity = sum(len(list(G.neighbors(n))) for n in G.nodes()) / max(1, len(G))
            
            # Check for connected components - fragmented graphs suggest poor coherence
            components = nx.number_connected_components(G.to_undirected())
            component_penalty = 1.0 / (1.0 + components)
            
            # Normalized score
            coherence_score = min(connectivity / 4, 1.0) * component_penalty
            return coherence_score
        except:
            return 0.5

    @lru_cache(maxsize=1024)
    def semantic_coherence_check(self, permutation):
        """Check semantic coherence of sentences using NLI model"""
        coherence_scores = []
        
        # For each adjacent pair of sentences
        for i in range(len(permutation) - 1):
            pair_key = (permutation[i], permutation[i+1])
            
            # Get from cache if possible
            if pair_key in self._coherence_scores:
                coherence_scores.append(self._coherence_scores[pair_key])
                continue
                
            try:
                # Use transformer model to check logical connection
                result = self.coherence_checker(
                    permutation[i] + ' ' + permutation[i+1], 
                    hypothesis='These sentences are logically connected'
                )[0]
                
                # Score interpretation
                if result['label'] == 'ENTAILMENT':
                    score = result['score']
                elif result['label'] == 'CONTRADICTION':
                    score = 1 - result['score']
                else:  # NEUTRAL
                    score = 0.5
                
                # Store in cache
                self._coherence_scores[pair_key] = score
                coherence_scores.append(score)
            except:
                coherence_scores.append(0.5)
        
        # Add penalty for low scores - this emphasizes overall consistency
        if coherence_scores:
            min_score = min(coherence_scores)
            avg_score = sum(coherence_scores) / len(coherence_scores)
            # Weight average more, but penalize very low minimum scores
            return (avg_score * 0.7) + (min_score * 0.3)
        else:
            return 0.5

    def generate_most_sensical_permutations(self, story, max_permutations=10, time_limit=30):
        """
        Generate the most sensical permutations within time limit
        
        Args:
            story (str): Original story text
            max_permutations (int): Maximum number of permutations to return
            time_limit (int): Maximum seconds to spend processing
            
        Returns:
            list: List of the most sensical story permutations
        """
        # Tokenize the story
        sentences = self.tokenize_story(story)
        n_sentences = len(sentences)
        
        # For extremely short stories, we can try all permutations
        if n_sentences <= 3:
            all_perms = list(itertools.permutations(sentences))
            scored_perms = [(self.calculate_coherence_score(perm), perm) for perm in all_perms]
            top_perms = heapq.nlargest(max_permutations, scored_perms)
            return [' '.join(perm) for score, perm in top_perms]
        
        # For longer stories, use progressive searching with a time limit
        start_time = time.time()
        
        # Priority queue to keep track of the best permutations
        best_perms = []
        
        # Always include the original permutation
        original_perm = tuple(sentences)
        original_score = self.calculate_coherence_score(original_perm)
        heapq.heappush(best_perms, (original_score, original_perm))
        
        # Track evaluated permutations to avoid duplicates
        evaluated = {original_perm}
        
        # Set batch size based on story length
        batch_size = max(10, min(1000, 10000 // (math.factorial(min(n_sentences, 5)))))
        
        # Generate and evaluate permutations in batches
        perm_generator = self.generate_batch_permutations(sentences, batch_size)
        
        batches_processed = 0
        total_evaluated = 1  # We already evaluated the original
        
        try:
            for batch in perm_generator:
                batches_processed += 1
                
                # Score all permutations in this batch
                for perm in batch:
                    # Skip if already evaluated
                    if perm in evaluated:
                        continue
                        
                    # Calculate coherence score
                    score = self.calculate_coherence_score(perm)
                    total_evaluated += 1
                    
                    # Add to priority queue if it's one of the best
                    if len(best_perms) < max_permutations:
                        heapq.heappush(best_perms, (score, perm))
                    elif score > best_perms[0][0]:  # Better than the worst in our queue
                        heapq.heapreplace(best_perms, (score, perm))
                    
                    # Mark as evaluated
                    evaluated.add(perm)
                
                # Check time limit after each batch
                elapsed = time.time() - start_time
                if elapsed > time_limit:
                    print(f"Time limit reached after evaluating {total_evaluated} permutations.")
                    break
                    
                # If we have a very large number of sentences, stop after a reasonable number of batches
                if n_sentences > 7 and batches_processed >= 100:
                    print(f"Stopping after {batches_processed} batches for a story with {n_sentences} sentences.")
                    break
        except KeyboardInterrupt:
            print("Process interrupted by user.")
        
        # Sort by score (highest first) and convert to text
        result = [' '.join(perm) for score, perm in sorted(best_perms, reverse=True)]
        
        print(f"Evaluated {total_evaluated} permutations in {time.time() - start_time:.2f} seconds.")
        return result

def main():
    # Example usage
    story = """
    Once upon a time, there was a brave knight. 
    He lived in a tall castle overlooking a vast kingdom. 
    One day, he decided to go on a quest to save a captured princess. 
    After many challenges, he finally rescued the princess from the dragon's lair. 
    The kingdom celebrated his heroic deed with a grand feast.
    """
    
    # Initialize the validator
    validator = StoryPermutationValidator()
    
    # Generate the most sensical permutations
    sensical_permutations = validator.generate_most_sensical_permutations(story)
    
    # Print results
    print("Most Sensical Story Permutations:")
    for i, perm in enumerate(sensical_permutations, 1):
        print(f"\nPermutation {i}:")
        print(perm)

if __name__ == "__main__":
    main()