from Levenshtein import distance

class TextPermutationEvaluator:
    def __init__(self, baseline_permutation, valid_permutations):
        """
        Initialize the evaluator with baseline and valid permutations
        
        :param baseline_permutation: Original order of sentences
        :param valid_permutations: List of acceptable alternative orders
        """
        self.baseline_permutation = baseline_permutation
        self.valid_permutations = valid_permutations

    def calculate_validity(self, child_perm):
        """
        Calculate the validity of the child's permutation
        
        :param child_perm: Permutation to evaluate
        :return: Minimum Levenshtein distance to valid permutations
        """
        distances = [distance(tuple(child_perm), tuple(valid_perm)) 
                     for valid_perm in self.valid_permutations]
        return min(distances)

    def calculate_creativity(self, child_perm):
        """
        Calculate creativity by measuring distance from baseline
        
        :param child_perm: Permutation to evaluate
        :return: Levenshtein distance from baseline permutation
        """
        return distance(tuple(child_perm), tuple(self.baseline_permutation))

    def evaluate_permutation(self, child_perm, validity_weight=0.7, creativity_weight=0.3):
        """
        Evaluate the child's permutation with weighted scoring
        
        :param child_perm: Permutation to evaluate
        :param validity_weight: Weight for validity score
        :param creativity_weight: Weight for creativity score
        :return: Dictionary of scores
        """
        # Calculate validity
        validity_distance = self.calculate_validity(child_perm)
        
        # Validity score calculation
        if validity_distance > 0:
            max_valid_distance = max(len(child_perm), len(self.valid_permutations[0]))
            validity_score = 1 - (validity_distance / max_valid_distance)
        else:
            validity_score = 1
        
        # Creativity score calculation
        creativity_distance = self.calculate_creativity(child_perm)
        max_creativity_distance = len(self.baseline_permutation)
        creativity_score = creativity_distance / max_creativity_distance if validity_distance == 0 else 0
        
        # Final score
        final_score = (validity_weight * validity_score) + (creativity_weight * creativity_score)
        
        return {
            "validity_score": validity_score,
            "creativity_score": creativity_score,
            "final_score": final_score
        }

# Main script to demonstrate usage
def main():
    # Example baseline permutation (simplest linear order)
    baseline_permutation = [
        "The sun rose over the mountains.",
        "The village began to wake up.",
        "Birds chirped in the distance."
    ]

    # Example valid permutations
    valid_permutations = [
        [
            "The sun rose over the mountains.",
            "The village began to wake up.",
            "Birds chirped in the distance."
        ],
        [
            "Birds chirped in the distance.",
            "The sun rose over the mountains.",
            "The village began to wake up."
        ]
    ]

    # Create an instance of the evaluator
    evaluator = TextPermutationEvaluator(
        baseline_permutation, 
        valid_permutations
    )

    # Example child's rearrangement to evaluate
    child_permutation1 = [
        "The sun rose over the mountains.",
        "The village began to wake up.",
        "Birds chirped in the distance."
        
    ]

    # Another example permutation
    child_permutation2 = [
        "The sun rose over the mountains.",
        "Birds chirped in the distance.",
        "The village began to wake up."
    ]

    # Evaluate different permutations
    print("Evaluation of Child Permutation 1:")
    scores1 = evaluator.evaluate_permutation(child_permutation1)
    print(scores1)

    print("\nEvaluation of Child Permutation 2:")
    scores2 = evaluator.evaluate_permutation(child_permutation2)
    print(scores2)

# Ensure the script runs only when directly executed
if __name__ == "__main__":
    main()



# from Levenshtein import distance
# import numpy as np

# class AdvancedTextPermutationEvaluator:
#     def __init__(self, baseline_permutation, valid_permutations, semantic_weights=None):
#         """
#         Enhanced initialization with optional semantic weighting
        
#         :param baseline_permutation: Original order of sentences
#         :param valid_permutations: List of acceptable alternative orders
#         :param semantic_weights: Optional weights for semantic importance of sentences
#         """
#         self.baseline_permutation = baseline_permutation
#         self.valid_permutations = valid_permutations
        
#         # Default semantic weights if not provided
#         self.semantic_weights = semantic_weights or [1.0] * len(baseline_permutation)

#     def calculate_semantic_distance(self, perm1, perm2):
#         """
#         Calculate semantic-weighted distance between permutations
        
#         :param perm1: First permutation
#         :param perm2: Second permutation
#         :return: Weighted Levenshtein distance
#         """
#         # Calculate position differences with semantic weighting
#         weighted_distance = 0
#         for i, (sent1, sent2) in enumerate(zip(perm1, perm2)):
#             if sent1 != sent2:
#                 # Multiply Levenshtein distance by semantic weight
#                 weighted_distance += distance(sent1, sent2) * self.semantic_weights[i]
        
#         return weighted_distance

#     def calculate_validity(self, child_perm):
#         """
#         Enhanced validity calculation with semantic consideration
        
#         :param child_perm: Permutation to evaluate
#         :return: Minimum semantic-weighted distance to valid permutations
#         """
#         distances = [self.calculate_semantic_distance(child_perm, valid_perm) 
#                      for valid_perm in self.valid_permutations]
#         return min(distances)

#     def calculate_creativity(self, child_perm):
#         """
#         Enhanced creativity calculation with more flexible approach
        
#         :param child_perm: Permutation to evaluate
#         :return: Creativity score that doesn't penalize correct arrangements
#         """
#         # Calculate the number of unique rearrangements
#         unique_arrangements = len(set(tuple(child_perm)))
#         total_arrangements = len(child_perm)
        
#         # Check if the permutation is different from baseline
#         is_different_from_baseline = tuple(child_perm) != tuple(self.baseline_permutation)
        
#         # Calculate a creativity score that rewards:
#         # 1. Different arrangements from baseline
#         # 2. Unique positioning of sentences
#         # 3. Meaningful rearrangements
#         creativity_score = (
#             is_different_from_baseline * 
#             (unique_arrangements / total_arrangements)
#         )
        
#         return creativity_score

#     def evaluate_permutation(self, child_perm, 
#                               validity_weight=0.7, 
#                               creativity_weight=0.3, 
#                               max_deviation_threshold=0.5):
#         """
#         Enhanced evaluation with more nuanced scoring
        
#         :param child_perm: Permutation to evaluate
#         :param validity_weight: Weight for validity score
#         :param creativity_weight: Weight for creativity score
#         :param max_deviation_threshold: Maximum allowed deviation
#         :return: Dictionary of detailed scores
#         """
#         # Calculate distances
#         validity_distance = self.calculate_validity(child_perm)
#         creativity_distance = self.calculate_creativity(child_perm)
        
#         # Normalize distances
#         max_valid_distance = max(len(child_perm), len(self.valid_permutations[0]))
#         max_creativity_distance = len(self.baseline_permutation)
        
#         # Validity score with more granular calculation
#         if validity_distance > 0:
#             # Exponential decay for validity to be more forgiving of small deviations
#             validity_score = np.exp(-validity_distance / max_valid_distance)
#         else:
#             validity_score = 1.0
        
#         # Creativity score with threshold
#         if validity_distance <= max_valid_distance * max_deviation_threshold:
#             creativity_score = min(1, creativity_distance / max_creativity_distance)
#         else:
#             creativity_score = 0
        
#         # Final score calculation with more robust weighting
#         final_score = (validity_weight * validity_score) + (creativity_weight * creativity_score)
        
#         return {
#             "raw_validity_distance": validity_distance,
#             "raw_creativity_distance": creativity_distance,
#             "validity_score": validity_score,
#             "creativity_score": creativity_score,
#             "final_score": final_score,
#             "is_different_from_baseline": creativity_score > 0
#         }

# def main():
#     # Example baseline permutation (simplest linear order)
#     baseline_permutation = [
#         "The sun rose over the mountains.",
#         "The village began to wake up.",
#         "Birds chirped in the distance."
#     ]

#     # Example valid permutations
#     valid_permutations = [
#         [
#             "The sun rose over the mountains.",
#             "The village began to wake up.",
#             "Birds chirped in the distance."
#         ],
#         [
#             "Birds chirped in the distance.",
#             "The sun rose over the mountains.",
#             "The village began to wake up."
#         ]
#     ]

#     # Optional: Add semantic weights to prioritize certain sentences
#     semantic_weights = [1.2, 1.0, 0.8]  # More importance to first and less to last sentence

#     # Create an instance of the advanced evaluator
#     evaluator = AdvancedTextPermutationEvaluator(
#         baseline_permutation, 
#         valid_permutations,
#         semantic_weights
#     )

#     # Example child's rearrangements to evaluate
#     child_permutations = [
#         [
#             "The sun rose over the mountains.",
#             "The village began to wake up.",
#             "Birds chirped in the distance."
#         ],
#         [
#             "The sun rose over the mountains.",
#             "Birds chirped in the distance.",
#             "The village began to wake up."
#         ]
#     ]

#     # Evaluate different permutations
#     for i, child_perm in enumerate(child_permutations, 1):
#         print(f"\nEvaluation of Child Permutation {i}:")
#         scores = evaluator.evaluate_permutation(child_perm)
#         for key, value in scores.items():
#             print(f"{key}: {value}")

# if __name__ == "__main__":
#     main()