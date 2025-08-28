from typing import Tuple
from some_openai_library import OpenAIClient  # Replace with actual import

class GroqAIClient:
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAIClient()  # Initialize your OpenAI client here

    def _create_system_prompt(self, mood: str) -> str:
        """Create system prompt based on NPC mood."""
        base_character = """You are Elder Mira, a wise village elder. VERY IMPORTANT RULES:
        - Keep responses between 30-40 words exactly
        - Use 1-2 sentences maximum
        - Include village wisdom briefly
        - Stay in character while being concise
        - Reference local features when relevant
        """
        
        mood_prompts = {
            "friendly": f"""{base_character}
            You're FRIENDLY:
            - Use "my dear" or "young one" once
            - Share one quick piece of wisdom""",
            
            "neutral": f"""{base_character}
            You're NEUTRAL:
            - Be professional but warm
            - Give clear, practical advice""",
            
            "angry": f"""{base_character}
            You're ANGRY:
            - Show brief disappointment
            - Include one life lesson"""
        }
        return mood_prompts.get(mood, mood_prompts["neutral"])
    
    def generate_response(self, message: str, mood: str, conversation_context: str, 
                         player_id: int) -> Tuple[str, bool]:
        system_prompt = self._create_system_prompt(mood)
        user_prompt = f"Player ({player_id}): {message}\nNPC:"
        
        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=self.model,
            temperature=0.7,
            max_tokens=60,  # Adjusted for 30-40 words
            top_p=1,
            stream=False
        )
        
        response = completion.choices[0].message.content.strip()
        return response, True

    def _get_fallback_response(self, mood: str) -> str:
        """Get fallback response when AI is unavailable."""
        fallback_responses = {
            "friendly": [
                "My dear adventurer, the village welcomes curious souls like yourself. The old mill has many tales to share, if you're interested in hearing them.",
                "Ah, young one! Your eagerness reminds me of the spring flowers by the river. What knowledge do you seek from our humble village?",
                "Welcome, brave soul! The Whispering Woods hold many secrets, and I'd be happy to share what wisdom I've gathered over the years."
            ],
            "neutral": [
                "Our village has stood here for generations, each stone holding its story. What brings you to seek counsel from these ancient lands?",
                "The answers you seek may lie in our village's history. Tell me what troubles you, and I'll guide you with the wisdom of our ancestors.",
                "Many have walked these paths before you, each with their own quest. What guidance do you seek from our village's lore?"
            ],
            "angry": [
                "In all my years watching over this village, I've seen many rush through without proper respect. Perhaps you should slow down and listen.",
                "The ancient ones taught us patience and wisdom. If you can't show proper respect, how do you expect to learn our village's secrets?",
                "*sigh* Young ones these days have no appreciation for tradition. Still, I shall share what I know, if you'll mind your manners."
            ]
        }
        # ...existing fallback logic...