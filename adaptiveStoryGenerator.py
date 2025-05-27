import os
import json
import re
from typing import Dict, Optional
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class NarrativeContinuationGenerator:
    def __init__(self):
        # Load environment variables
        load_dotenv(find_dotenv())
        load_dotenv(override=True)

        # Initialize AI models
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")

        # Narrative tracking
        self.narrative_context = {
            'previous_full_story': None,
            'characters': {},
            'themes': [],
            'plot_progression': []
        }
        
        # Complexity tracking
        self.complexity_history = []

    def extract_narrative_elements(self, previous_story: str):
        """
        Extract key narrative elements from the previous story
        """
        extraction_prompt = f"""
        Analyze the following story and extract:
        1. Main characters with their key traits and development
        2. Central themes and narrative arcs
        3. Critical plot points and story progression
        4. Unresolved narrative threads or potential future developments

        Story:
        {previous_story}

        Respond in JSON format with:
        {{
            "characters": {{
                "name": {{
                    "traits": [],
                    "arc": [],
                    "unresolved_motivations": []
                }}
            }},
            "themes": [],
            "plot_progression": [
                {{
                    "event": "",
                    "significance": "",
                    "potential_continuation": ""
                }}
            ],
            "narrative_gaps": []
        }}
        """

        try:
            # Use the LLM to extract narrative elements
            response = self.llm.invoke(extraction_prompt)
            narrative_data = json.loads(response.content)

            # Update narrative context
            self.narrative_context = {
                'previous_full_story': previous_story,
                'characters': narrative_data.get('characters', {}),
                'themes': narrative_data.get('themes', []),
                'plot_progression': narrative_data.get('plot_progression', [])
            }

            return self.narrative_context
        except Exception as e:
            print(f"Narrative extraction error: {e}")
            return self.narrative_context

    def calculate_complexity_level(
        self, 
        previous_story: str, 
        previous_scores: Dict[str, float]
    ) -> int:
        """
        Dynamically calculate complexity level based on story and previous scores
        """
        # Base complexity calculation from story characteristics
        story_complexity_metrics = {
            'vocabulary_complexity': len(set(re.findall(r'\b\w+\b', previous_story))) / len(previous_story.split()),
            'sentence_complexity': sum(len(sent.split()) for sent in previous_story.split('.')) / len(previous_story.split('.')),
            'character_depth': len(re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', previous_story))
        }
        
        # Calculate story base complexity
        base_complexity_score = sum(story_complexity_metrics.values()) / len(story_complexity_metrics)
        
        # Incorporate creativity and validity scores
        creativity_score = previous_scores.get('creativity_score', 0.5)
        validity_score = previous_scores.get('validity_score', 0.5)
        
        # Complex scoring mechanism
        adjusted_complexity = (
            base_complexity_score * 0.5 +
            creativity_score * 0.3 +
            validity_score * 0.2
        )
        
        # Map to complexity levels with more nuanced approach
        if adjusted_complexity < 0.2:
            complexity_level = 1
        elif adjusted_complexity < 0.4:
            complexity_level = 2
        elif adjusted_complexity < 0.6:
            complexity_level = 3
        elif adjusted_complexity < 0.8:
            complexity_level = 4
        else:
            complexity_level = 5
        
        # Track complexity history
        self.complexity_history.append({
            'complexity_level': complexity_level,
            'base_score': base_complexity_score,
            'creativity_score': creativity_score,
            'validity_score': validity_score
        })
        
        return complexity_level

    def generate_story_continuation(
        self, 
        previous_story: str, 
        previous_scores: Dict[str, float],
        continuation_prompt: Optional[str] = None
    ):
        """
        Generate the next iteration of the story
        """
        # Extract narrative elements from previous story
        narrative_context = self.extract_narrative_elements(previous_story)
        
        # Determine complexity level considering previous scores
        complexity_level = self.calculate_complexity_level(
            previous_story, 
            previous_scores
        )

        # Default continuation prompt if not provided
        if not continuation_prompt:
            continuation_prompt = "Continue the story, maintaining the established narrative and characters"

        # Create a sophisticated prompt for story continuation with controlled length
        full_prompt = ChatPromptTemplate.from_template("""
        Story Continuation Parameters:
        - Complexity Level: {complexity_level}
        - Creativity Score: {creativity_score}
        - Validity Score: {validity_score}

        Narrative Continuation Guidelines:
        1. Maintain narrative consistency
        2. Build upon existing characters and themes
        3. Explore unresolved plot points
        4. Adjust narrative complexity based on previous performance

        Narrative Context:
        Characters: {characters}
        Themes: {themes}
        Previous Plot Progression: {plot_progression}

        Previous Story:
        {previous_story}

        Continuation Prompt: {continuation_prompt}

        IMPORTANT FORMATTING INSTRUCTIONS:
        - Generate a SHORT continuation with exactly 3-5 sentences
        - Keep sentences relatively simple and somewhat interchangeable
        - Focus on simple narrative structure rather than complex cause-and-effect
        - Sentences should be self-contained but related to the overall theme
        - Do NOT number the sentences or add a title

        Generate the next segment of the story, ensuring:
        - Seamless narrative flow
        - Character consistency
        - Thematic coherence
        - Appropriate complexity for the specified level
        """)

        # Create a chain with the prompt and output parser
        chain = full_prompt | self.llm | StrOutputParser()

        # Generate story continuation
        story_continuation = chain.invoke({
            "complexity_level": complexity_level,
            "creativity_score": previous_scores.get('creativity_score', 0.5),
            "validity_score": previous_scores.get('validity_score', 0.5),
            "characters": json.dumps(narrative_context['characters']),
            "themes": narrative_context['themes'],
            "plot_progression": json.dumps(narrative_context['plot_progression']),
            "previous_story": previous_story,
            "continuation_prompt": continuation_prompt
        })

        return {
            "story_continuation": story_continuation,
            "complexity_level": complexity_level,
            "narrative_context": narrative_context,
            "complexity_history": self.complexity_history[-1]
        }

def main():
    # Initialize the generator
    generator = NarrativeContinuationGenerator()

    # Simulate a previous story - intentionally kept short with few sentences
    previous_story = """
    In a world where time was a fabric that could be woven, young Aria discovered she had the rare ability to see temporal threads. 
    Her first glimpse came on her 12th birthday when a mysterious crystal appeared in her grandmother's attic.
    The crystal revealed whispers of forgotten timelines and potential futures.
    """

    # Generate the next story iteration with explicit creativity and validity scores
    next_story_iteration = generator.generate_story_continuation(
        previous_story=previous_story,
        previous_scores={
            'creativity_score': 0.7,  # High creativity
            'validity_score': 0.6     # Moderate validity
        },
        continuation_prompt="Explore Aria's first attempt to manipulate time threads"
    )

    print("Original Story (3 sentences):")
    print(previous_story.strip())
    
    print("\nStory Continuation (3-5 sentences):")
    print(next_story_iteration['story_continuation'])
    
    print("\nComplexity Level:", next_story_iteration['complexity_level'])
    
    # Count sentences in continuation
    sentence_count = len(next_story_iteration['story_continuation'].split('.'))
    sentence_count = sum(1 for s in next_story_iteration['story_continuation'].split('.') if s.strip())
    print(f"Sentences in continuation: {sentence_count}")
    
    print("\nExtracted Narrative Elements:")
    print("Characters:", next_story_iteration['narrative_context']['characters'].keys())
    print("Themes:", next_story_iteration['narrative_context']['themes'])

if __name__ == "__main__":
    main()














# import os
# import json
# import re
# from typing import Dict, List, Optional
# from dotenv import load_dotenv, find_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.documents import Document

# class NarrativeTracker:
#     def __init__(self):
#         # Comprehensive story tracking
#         self.characters = {}  # {name: {traits: [], arc: [], importance: int}}
#         self.themes = []  # List of core narrative themes
#         self.plot_points = []  # Key events and their significance
#         self.story_world = {
#             'setting': None,
#             'time_period': None,
#             'key_locations': []
#         }

#     def extract_narrative_elements(self, story_text):
#         """
#         Advanced extraction of narrative elements using LLM
#         """
#         extraction_prompt = f"""
#         Analyze the following story text and extract:
#         1. Main characters with their key traits
#         2. Central themes
#         3. Critical plot points
#         4. Story world details (setting, time period, locations)

#         Story Text:
#         {story_text}

#         Response Format (JSON):
#         {{
#             "characters": {{
#                 "CharacterName": {{
#                     "traits": [],
#                     "arc": [],
#                     "importance": 1-10
#                 }}
#             }},
#             "themes": [],
#             "plot_points": [],
#             "story_world": {{
#                 "setting": "",
#                 "time_period": "",
#                 "key_locations": []
#             }}
#         }}
#         """
        
#         # Use OpenAI for extraction (simulated here)
#         extraction_model = ChatOpenAI(model_name="gpt-3.5-turbo")
#         response = extraction_model.invoke(extraction_prompt)
        
#         try:
#             narrative_data = json.loads(response.content)
            
#             # Update internal tracking
#             self.characters.update(narrative_data.get('characters', {}))
#             self.themes.extend(narrative_data.get('themes', []))
#             self.plot_points.extend(narrative_data.get('plot_points', []))
            
#             # Update story world
#             world_data = narrative_data.get('story_world', {})
#             if world_data.get('setting'):
#                 self.story_world['setting'] = world_data['setting']
#             if world_data.get('time_period'):
#                 self.story_world['time_period'] = world_data['time_period']
#             self.story_world['key_locations'].extend(
#                 world_data.get('key_locations', [])
#             )
#         except Exception as e:
#             print(f"Narrative extraction error: {e}")

# class ComplexityScorer:
#     def __init__(self):
#         # Comprehensive scoring mechanism
#         self.complexity_metrics = {
#             'vocabulary_complexity': 0,
#             'narrative_structure': 0,
#             'character_depth': 0,
#             'thematic_complexity': 0,
#             'plot_intricacy': 0
#         }
#         self.complexity_history = []
#         self.complexity_level = 1

#     def calculate_complexity_score(self, story_text):
#         """
#         Multi-dimensional complexity assessment
#         """
#         # Simulated complexity calculation
#         vocab_complexity = len(set(re.findall(r'\b\w+\b', story_text))) / len(story_text.split())
#         sentence_complexity = sum(len(sent.split()) for sent in story_text.split('.')) / len(story_text.split('.'))
#         character_complexity = len(re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', story_text))

#         # Normalize and score
#         self.complexity_metrics = {
#             'vocabulary_complexity': min(vocab_complexity * 10, 10),
#             'narrative_structure': sentence_complexity,
#             'character_depth': character_complexity,
#             'thematic_complexity': len(set(re.findall(r'\b\w+ing\b', story_text))) / len(story_text.split()),
#             'plot_intricacy': len(re.findall(r'however|nevertheless|despite', story_text.lower()))
#         }

#         # Calculate overall complexity
#         total_score = sum(self.complexity_metrics.values()) / len(self.complexity_metrics)
#         return total_score

#     def adjust_complexity(self, previous_story_scores, current_story_text):
#         """
#         Intelligent complexity adjustment
#         """
#         current_complexity = self.calculate_complexity_score(current_story_text)
        
#         # Historical complexity tracking
#         self.complexity_history.append({
#             'score': current_complexity,
#             'previous_scores': previous_story_scores
#         })

#         # Nuanced complexity adjustment
#         adjustment_factors = {
#             'creativity_score': previous_story_scores.get('creativity_score', 0.5),
#             'validity_score': previous_story_scores.get('validity_score', 0.5)
#         }

#         if current_complexity > 7 and adjustment_factors['creativity_score'] > 0.7:
#             self.complexity_level = min(self.complexity_level + 1, 5)
#         elif current_complexity < 3 and adjustment_factors['creativity_score'] < 0.3:
#             self.complexity_level = max(self.complexity_level - 1, 1)

#         return self.complexity_level

# class AdaptiveStoryGenerator:
#     def __init__(self):
#         load_dotenv(find_dotenv())
        
#         self.narrative_tracker = NarrativeTracker()
#         self.complexity_scorer = ComplexityScorer()
        
#         # Initialize Pinecone and LLM
#         self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#         self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
#         self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")

#     def generate_adaptive_story(
#         self, 
#         user_query: str, 
#         previous_story: Optional[str] = None, 
#         previous_scores: Optional[Dict] = None
#     ):
#         # Default previous scores if not provided
#         previous_scores = previous_scores or {
#             'creativity_score': 0.5, 
#             'validity_score': 0.5
#         }

#         # If previous story exists, extract narrative elements
#         if previous_story:
#             self.narrative_tracker.extract_narrative_elements(previous_story)

#         # Determine complexity level
#         complexity_level = self.complexity_scorer.adjust_complexity(
#             previous_scores, 
#             previous_story or user_query
#         )

#         # Construct narrative-aware prompt
#         prompt_template = ChatPromptTemplate.from_template(f"""
#         Story Generation with Complexity Level {complexity_level}

#         Narrative Context:
#         - Characters: {json.dumps(self.narrative_tracker.characters)}
#         - Themes: {self.narrative_tracker.themes}
#         - Story World: {json.dumps(self.narrative_tracker.story_world)}

#         Complexity Guidelines (Level {complexity_level}):
#         1. Use age-appropriate language
#         2. Maintain consistent narrative style
#         3. Build upon existing story elements
#         4. Introduce subtle complexity increases

#         User Query: {user_query}
        
#         Generate a story continuation that respects the established narrative 
#         while subtly increasing narrative sophistication.
#         """)

#         # Generate story
#         document_chain = create_stuff_documents_chain(self.llm, prompt_template)
#         story_response = document_chain.invoke({
#             "input": user_query,
#             "context": []  # Placeholder for future context retrieval
#         })

#         return {
#             "story": story_response,
#             "complexity_level": complexity_level,
#             "narrative_elements": {
#                 "characters": self.narrative_tracker.characters,
#                 "themes": self.narrative_tracker.themes,
#                 "story_world": self.narrative_tracker.story_world
#             }
#         }

# def main():
#     generator = AdaptiveStoryGenerator()
    
#     # First story generation
#     first_story = generator.generate_adaptive_story(
#         "Create an adventure about a child discovering a magical timeline"
#     )
#     print("First Story:", first_story['story'])
#     print("Complexity Level:", first_story['complexity_level'])

#     # Second story generation with previous context
#     second_story = generator.generate_adaptive_story(
#         "Continue the magical timeline adventure",
#         previous_story=first_story['story'],
#         previous_scores={
#             'creativity_score': 0.7,
#             'validity_score': 0.6
#         }
#     )
#     print("\nSecond Story:", second_story['story'])
#     print("Complexity Level:", second_story['complexity_level'])

# if __name__ == "__main__":
#     main()