import itertools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import networkx as nx
from transformers import pipeline

class StoryPermutationValidator:
    def __init__(self):
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        
        # Load spaCy language model for linguistic analysis
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize coherence checker
        self.coherence_checker = pipeline('text-classification', 
                                          model='facebook/bart-large-mnli')
        
        # Stopwords for filtering
        self.stop_words = set(stopwords.words('english'))

    def tokenize_story(self, story):
        """
        Tokenize the story into sentences
        
        Args:
            story (str): Full story text
        
        Returns:
            list: List of sentences
        """
        return sent_tokenize(story)

    def generate_permutations(self, sentences):
        """
        Generate all possible permutations of sentences
        
        Args:
            sentences (list): List of story sentences
        
        Returns:
            list: List of all possible sentence arrangements
        """
        return list(itertools.permutations(sentences))

    def check_grammatical_coherence(self, permutation):
        """
        Check basic grammatical coherence of a sentence permutation
        
        Args:
            permutation (tuple): Sequence of sentences
        
        Returns:
            float: Coherence score
        """
        # Join sentences to create a potential story
        story_text = ' '.join(permutation)
        
        # Use spaCy for linguistic analysis
        doc = self.nlp(story_text)
        
        # Check dependency parsing and sentence structure
        dependency_score = self._analyze_dependency_structure(doc)
        
        return dependency_score

    def _analyze_dependency_structure(self, doc):
        """
        Analyze the dependency parsing of the document
        
        Args:
            doc (spacy.tokens.Doc): Parsed document
        
        Returns:
            float: Dependency coherence score
        """
        # Create a dependency graph
        G = nx.DiGraph()
        
        # Add nodes and edges based on dependency parsing
        for token in doc:
            G.add_node(token.text, pos=token.pos_, dep=token.dep_)
            if token.head.text != token.text:
                G.add_edge(token.head.text, token.text)
        
        # Calculate graph metrics
        try:
            clustering = nx.average_clustering(G)
            connectivity = nx.average_degree_connectivity(G)
            avg_connectivity = sum(connectivity.values()) / len(connectivity)
            
            # Combine metrics
            coherence_score = (clustering + avg_connectivity) / 2
            return min(max(coherence_score, 0), 1)
        except:
            return 0.5

    def semantic_coherence_check(self, permutation):
        """
        Check semantic coherence using transformer-based NLI
        
        Args:
            permutation (tuple): Sequence of sentences
        
        Returns:
            float: Semantic coherence score
        """
        # Compare adjacent sentences for logical progression
        coherence_scores = []
        for i in range(len(permutation) - 1):
            try:
                result = self.coherence_checker(
                    permutation[i] + ' ' + permutation[i+1], 
                    hypothesis='These sentences are logically connected'
                )[0]
                
                # Convert label to numerical score
                if result['label'] == 'ENTAILMENT':
                    score = result['score']
                elif result['label'] == 'CONTRADICTION':
                    score = 1 - result['score']
                else:
                    score = 0.5
                
                coherence_scores.append(score)
            except:
                coherence_scores.append(0.5)
        
        # Return average coherence
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5

    def validate_permutation(self, permutation, threshold=0.6):
        """
        Validate if a permutation makes sense
        
        Args:
            permutation (tuple): Sequence of sentences
            threshold (float): Minimum coherence threshold
        
        Returns:
            bool: Whether the permutation is sensical
        """
        # Check different coherence aspects
        grammatical_score = self.check_grammatical_coherence(permutation)
        semantic_score = self.semantic_coherence_check(permutation)
        
        # Combine scores
        total_score = (grammatical_score + semantic_score) / 2
        
        return total_score >= threshold

    def generate_sensical_permutations(self, story, max_permutations=10):
        """
        Generate sensical permutations of a story
        
        Args:
            story (str): Original story text
            max_permutations (int): Maximum number of sensical permutations to return
        
        Returns:
            list: List of sensical story permutations
        """
        # Tokenize the story
        sentences = self.tokenize_story(story)
        
        # Generate all permutations
        all_permutations = self.generate_permutations(sentences)
        
        # Filter sensical permutations
        sensical_permutations = []
        for perm in all_permutations:
            if self.validate_permutation(perm):
                sensical_permutations.append(' '.join(perm))
                
                # Stop if we've found enough permutations
                if len(sensical_permutations) >= max_permutations:
                    break
        
        return sensical_permutations

def main():
    # Example usage
    story = """
    Once upon a time, there was a brave knight. 
    He lived in a tall castle overlooking a vast kingdom. 
    One day, he decided to go on a quest to save a captured princess. 
    After many challenges, he finally rescued the princess from the dragon's lair. 
    The kingdom celebrated his heroic deed with a grand feast.
    """
    
    # Initialize the permutation validator
    validator = StoryPermutationValidator()
    
    # Generate sensical permutations
    sensical_permutations = validator.generate_sensical_permutations(story)
    
    # Print the permutations
    print("Sensical Story Permutations:")
    for i, perm in enumerate(sensical_permutations, 1):
        print(f"\nPermutation {i}:")
        print(perm)

if __name__ == "__main__":
    main()

