import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Deque
import re

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Run: pip install python-dotenv")

# Third-party imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("Warning: Groq library not installed. Run: pip install groq")
    GROQ_AVAILABLE = False


class NPCMoodSystem:
    """Handles NPC mood analysis and transitions."""
    
    FRIENDLY_KEYWORDS = {
        "help", "please", "thank", "thanks", "sorry", "great", 
        "awesome", "excellent", "wonderful", "amazing", "love", 
        "appreciate", "grateful", "kind", "nice", "good"
    }
    
    ANGRY_KEYWORDS = {
        "useless", "stupid", "waste", "annoying", "move", "slow", 
        "hate", "terrible", "awful", "worst", "dumb", "idiot", 
        "broken", "suck", "garbage", "trash", "crap"
    }
    
    def __init__(self):
        self.mood_states = ["angry", "neutral", "friendly"]
    
    def analyze_sentiment(self, message: str) -> Tuple[str, float]:
        """
        Analyze message sentiment and return mood and confidence score.
        
        Args:
            message: Player message to analyze
            
        Returns:
            Tuple of (mood, confidence_score)
        """
        message_lower = message.lower()
        words = re.findall(r'\b\w+\b', message_lower)
        
        friendly_count = sum(1 for word in words if word in self.FRIENDLY_KEYWORDS)
        angry_count = sum(1 for word in words if word in self.ANGRY_KEYWORDS)
        
        total_sentiment_words = friendly_count + angry_count
        if total_sentiment_words == 0:
            return "neutral", 0.5
        
        friendly_ratio = friendly_count / total_sentiment_words
        angry_ratio = angry_count / total_sentiment_words
        
        if friendly_ratio > angry_ratio:
            confidence = min(0.9, 0.5 + friendly_ratio)
            return "friendly", confidence
        elif angry_ratio > friendly_ratio:
            confidence = min(0.9, 0.5 + angry_ratio)
            return "angry", confidence
        else:
            return "neutral", 0.5
    
    def transition_mood(self, current_mood: str, new_sentiment: str, confidence: float) -> str:
        """
        Determine mood transition based on current mood and new sentiment.
        
        Args:
            current_mood: Current NPC mood
            new_sentiment: Detected sentiment from message
            confidence: Confidence score of sentiment analysis
            
        Returns:
            New mood state
        """
        # Require higher confidence for mood changes
        threshold = 0.6
        
        if confidence < threshold:
            # Gradual recovery towards neutral for low confidence
            if current_mood == "angry":
                return "neutral" if confidence > 0.3 else "angry"
            return current_mood
        
        # Direct transition for high confidence
        if new_sentiment in self.mood_states:
            return new_sentiment
        
        return current_mood


class PlayerState:
    """Represents the state of a single player."""
    
    def __init__(self, player_id: int, max_history: int = 3):
        self.player_id = player_id
        self.conversation_history: Deque[str] = deque(maxlen=max_history)
        self.npc_mood = "neutral"
        self.last_interaction: Optional[datetime] = None
        self.message_count = 0
        self.mood_history: List[Tuple[datetime, str]] = []
    
    def add_message(self, message: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(message)
        self.message_count += 1
        self.last_interaction = datetime.now(timezone.utc)
    
    def update_mood(self, new_mood: str) -> None:
        """Update NPC mood and record the change."""
        if new_mood != self.npc_mood:
            self.mood_history.append((datetime.now(timezone.utc), new_mood))
            self.npc_mood = new_mood
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for AI prompt."""
        if not self.conversation_history:
            return "No previous conversation history."
        
        history_text = "Recent conversation:\n"
        for i, msg in enumerate(self.conversation_history, 1):
            history_text += f"{i}. Player: {msg}\n"
        return history_text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert player state to dictionary for logging."""
        return {
            "player_id": self.player_id,
            "conversation_history": list(self.conversation_history),
            "npc_mood": self.npc_mood,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "message_count": self.message_count,
            "mood_changes": len(self.mood_history)
        }


class GroqAIClient:
    """Handles communication with Groq AI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3-70b-8192"):
        # Mask API key in logs
        if api_key and len(api_key) > 8:
            masked_key = f"{api_key[:4]}...{api_key[-4:]}"
            logging.info(f"Initializing Groq client with API key: {masked_key}")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.client = None
        
        if not GROQ_AVAILABLE:
            logging.warning("Groq library not installed. Install with: pip install groq")
            return
            
        if not self.api_key:
            logging.warning("No Groq API key provided. Set GROQ_API_KEY environment variable")
            return
            
        try:
            self.client = Groq(api_key=self.api_key)
            logging.info("Successfully initialized Groq client")
        except Exception as e:
            logging.error(f"Failed to initialize Groq client: {str(e)}", extra={
                "component": "groq_client",
                "action": "initialize",
                "error": str(e)
            })
    
    def generate_response(self, message: str, mood: str, conversation_context: str, 
                         player_id: int) -> Tuple[str, bool]:
        """
        Generate NPC response using Groq AI.
        
        Args:
            message: Player message
            mood: Current NPC mood
            conversation_context: Previous conversation history
            player_id: Player identifier
            
        Returns:
            Tuple of (response, success_flag)
        """
        if not self.client:
            return self._get_fallback_response(mood), False
        
        try:
            system_prompt = self._create_system_prompt(mood)
            user_prompt = self._create_user_prompt(message, conversation_context, player_id)
            
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=100,  # Increased to allow 50-60 words
                top_p=1,
                stream=False
            )
            
            response = completion.choices[0].message.content.strip()
            return response, True
            
        except Exception as e:
            logging.error(f"Groq API error for player {player_id}: {e}")
            return self._get_fallback_response(mood), False
    
    def _create_system_prompt(self, mood: str) -> str:
        """Create system prompt based on NPC mood."""
        base_character = """You are Elder Mira, a wise village elder. VERY IMPORTANT RULES:
        - Keep responses between 50-60 words
        - Use 2-3 short sentences maximum
        - Include some local lore or village wisdom
        - Stay in character while being informative
        - Reference landmarks or village life when relevant
        """
        
        mood_prompts = {
            "friendly": f"""{base_character}
            You're FRIENDLY:
            - Use warm phrases like "my dear" or "young one"
            - Share wisdom with gentle enthusiasm
            - Include small personal anecdotes""",
            
            "neutral": f"""{base_character}
            You're NEUTRAL:
            - Be professional but approachable
            - Give factual information with context
            - Share practical village knowledge""",
            
            "angry": f"""{base_character}
            You're ANGRY:
            - Show stern disappointment
            - Reference your years of experience
            - Include life lessons in your criticism"""
        }
        
        return mood_prompts.get(mood, mood_prompts["neutral"])
    
    def _create_user_prompt(self, message: str, context: str, player_id: int) -> str:
        """Create user prompt with context."""
        return f"""Player {player_id} says: "{message}"

{context}

Please respond as an NPC would, staying in character according to your current mood."""
    
    def _get_fallback_response(self, mood: str) -> str:
        """Get fallback response when AI is unavailable."""
        fallback_responses = {
            "friendly": [
                "Ah, my dear adventurer! The village always welcomes those with curious hearts. I remember when I was your age, exploring these very paths. What would you like to know about our humble home?",
                "Welcome, young one! Your eagerness reminds me of the spring blossoms by the old mill. Let me share some of our village's wisdom with you.",
                "Such a pleasure to meet another brave soul! The Whispering Woods have many secrets, and I'd be delighted to share what I know."
            ],
            "neutral": [
                "The village has stood here for generations, each stone holding its own story. What knowledge do you seek from our ancient lands?",
                "Many travelers come through these parts, each with their own quest. Tell me what brings you to our village, and I'll guide you as best I can.",
                "The answers you seek may lie in our village's long history. Speak your mind, and I'll share what wisdom I have gathered over the years."
            ],
            "angry": [
                "*sigh* In all my years watching over this village, I've seen countless young ones rush through without proper respect. Perhaps you'd learn more if you slowed down to listen.",
                "The ancient ones taught us patience and wisdom, virtues that seem lost on today's youth. Still, I shall share what I know, if you're willing to listen properly.",
                "Your haste does you no credit, young one. The village's secrets reveal themselves only to those who show proper respect for our ways."
            ]
        }
        
        import random
        responses = fallback_responses.get(mood, fallback_responses["neutral"])
        return random.choice(responses)


class NPCChatSystem:
    """Main NPC Chat System orchestrator."""
    
    def __init__(self, api_key: Optional[str] = None, log_file: str = "npc_chat_log.json"):
        self.player_states: Dict[int, PlayerState] = {}
        self.mood_system = NPCMoodSystem()
        self.ai_client = GroqAIClient(api_key)
        self.log_file = log_file
        self.responses_file = "npc_responses.json"
        self.interactions = []
        self.processed_messages = 0
        self.api_successes = 0
        self.api_failures = 0
        
        # Setup logging
        self._setup_logging()
        
        logging.info("NPC Chat System initialized", extra={
            "component": "system",
            "action": "initialize",
            "groq_available": GROQ_AVAILABLE,
            "ai_client_ready": self.ai_client.client is not None
        })
    
    def _setup_logging(self) -> None:
        """Configure structured logging."""
        
        class StructuredFormatter(logging.Formatter):
            """Custom formatter for structured JSON logging."""
            
            def format(self, record):
                # Only log errors and player interactions
                if record.levelno >= logging.ERROR or record.name == "root":
                    log_entry = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "level": record.levelname,
                        "message": record.getMessage()
                    }
                    return json.dumps(log_entry, default=str)
                return ""  # Skip other log messages
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Silence httpx and groq client logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("groq").setLevel(logging.WARNING)
        
        # Console handler - only show player interactions
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(StructuredFormatter())
        console_handler.addFilter(lambda record: record.name == "root")
        logger.addHandler(console_handler)
        
        # File handler - keep detailed logs in file
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    
    def load_messages(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load player messages from JSON file.
        
        Args:
            file_path: Path to JSON file containing messages
            
        Returns:
            List of message dictionaries
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Support both {"messages": [...]} and [...] formats
            if isinstance(data, dict) and 'messages' in data:
                messages = data['messages']
            elif isinstance(data, list):
                messages = data
            else:
                messages = []
            
            logging.info(f"Loaded {len(messages)} messages from {file_path}", extra={
                "component": "loader",
                "action": "load_messages",
                "file_path": file_path,
                "message_count": len(messages)
            })
            
            return messages
            
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}", extra={
                "component": "loader",
                "action": "load_messages",
                "error": "file_not_found",
                "file_path": file_path
            })
            return []
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {file_path}: {e}", extra={
                "component": "loader",
                "action": "load_messages",
                "error": "json_decode_error",
                "file_path": file_path,
                "json_error": str(e)
            })
            return []
        except Exception as e:
            logging.error(f"Error loading messages: {e}", extra={
                "component": "loader",
                "action": "load_messages",
                "error": "unexpected_error",
                "exception": str(e)
            })
            return []
    
    def sort_messages_chronologically(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort messages by timestamp to handle out-of-order delivery.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Sorted list of messages
        """
        try:
            sorted_messages = sorted(
                messages, 
                key=lambda x: datetime.fromisoformat(x['timestamp'].replace('Z', '+00:00'))
            )
            
            logging.info(f"Sorted {len(messages)} messages chronologically", extra={
                "component": "processor",
                "action": "sort_messages",
                "message_count": len(messages)
            })
            
            return sorted_messages
            
        except Exception as e:
            logging.error(f"Error sorting messages: {e}", extra={
                "component": "processor",
                "action": "sort_messages",
                "error": "sort_failed",
                "exception": str(e)
            })
            return messages
    def get_or_create_player_state(self, player_id: int) -> PlayerState:
        """Get existing player state or create new one."""
        if player_id not in self.player_states:
            self.player_states[player_id] = PlayerState(player_id)
            logging.info(f"Created new player state", extra={
                "component": "state_manager",
                "action": "create_player_state",
                "player_id": player_id
            })
        
        return self.player_states[player_id]
    
    def process_message(self, message_data: Dict[str, Any]) -> None:
        try:
            player_id = int(message_data['player_id'])
            message_text = message_data['text']
            timestamp = message_data['timestamp']
            
            # Get player state
            player_state = self.get_or_create_player_state(player_id)
            
            # Store current mood for comparison
            previous_mood = player_state.npc_mood
            
            # Analyze sentiment and update mood
            sentiment, confidence = self.mood_system.analyze_sentiment(message_text)
            new_mood = self.mood_system.transition_mood(previous_mood, sentiment, confidence)
            player_state.update_mood(new_mood)
            
            # Add message to conversation history
            player_state.add_message(message_text)
            
            # Generate NPC response
            conversation_context = player_state.get_conversation_context()
            npc_response, ai_success = self.ai_client.generate_response(
                message_text, new_mood, conversation_context, player_id
            )
            
            # Update statistics
            self.processed_messages += 1
            if ai_success:
                self.api_successes += 1
            else:
                self.api_failures += 1
            
            # Log the interaction
            self._log_interaction(
                player_id=player_id,
                message_text=message_text,
                npc_response=npc_response,
                previous_mood=previous_mood,
                new_mood=new_mood,
                sentiment=sentiment,
                confidence=confidence,
                ai_success=ai_success,
                timestamp=timestamp
            )
            
            # Console output
            self._display_interaction(
                player_id, message_text, npc_response, previous_mood, 
                new_mood, ai_success
            )
            
        except Exception as e:
            logging.error(f"Error processing message: {e}", extra={
                "component": "processor",
                "action": "process_message",
                "error": "processing_failed",
                "message_data": message_data,
                "exception": str(e)
            })
    
    def process_all_messages(self, file_path: str) -> None:
        """Process all messages from the JSON file."""
        start_time = time.time()
        
        # Load messages
        messages = self.load_messages(file_path)
        if not messages:
            print("No messages to process. Exiting.")
            return
        
        # Sort chronologically to handle out-of-order messages
        sorted_messages = self.sort_messages_chronologically(messages)
        
        print(f"\n🚀 Starting to process {len(sorted_messages)} messages...")
        print("=" * 80)
        
        # Process each message
        for i, message in enumerate(sorted_messages, 1):
            self.process_message(message)
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        # Calculate and display processing time
        end_time = time.time()
        self._display_summary(end_time - start_time)

    def _log_interaction(self, **kwargs) -> None:
        """Log interaction with structured data."""
        logging.info("Player interaction processed", extra={
            "component": "interaction",
            "action": "message_processed",
            **kwargs
        })
    
    def _display_interaction(self, player_id: int, message: str, response: str,
                           old_mood: str, new_mood: str, ai_success: bool) -> None:
        """Display interaction in console and save to file."""
        # Create interaction record
        interaction = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "player_id": player_id,
            "message": message,
            "response": response,
            "mood": {
                "previous": old_mood,
                "current": new_mood
            },
            "ai_success": ai_success
        }
        self.interactions.append(interaction)
        
        # Save to file after each interaction
        with open(self.responses_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "total_interactions": len(self.interactions),
                    "unique_players": len(self.player_states),
                    "api_success_rate": (self.api_successes / max(self.processed_messages, 1)) * 100
                },
                "interactions": self.interactions
            }, f, indent=2, ensure_ascii=False)
        
        # Console display
        mood_emoji = {"friendly": "😊", "neutral": "😐", "angry": "😠"}
        print(f"\n{'='*60}")
        print(f"Player {player_id}: {message}")
        print(f"Mood: {old_mood} {mood_emoji.get(old_mood, '')} → {new_mood} {mood_emoji.get(new_mood, '')}")
        print(f"NPC Response: {response}")
        print(f"AI Success: {'✅' if ai_success else '❌ (Fallback used)'}")
        print(f"{'='*60}")
    
    def _display_summary(self, processing_time: float) -> None:
        """Display summary statistics and save final state."""
        print("\n" + "="*80)
        print("🎯 PROCESSING COMPLETE - SUMMARY STATISTICS")
        print("="*80)
        
        print(f"📊 Messages Processed: {self.processed_messages}")
        print(f"👥 Unique Players: {len(self.player_states)}")
        print(f"✅ AI API Successes: {self.api_successes}")
        print(f"❌ API Failures (Fallbacks): {self.api_failures}")
        print(f"📈 Success Rate: {(self.api_successes / max(self.processed_messages, 1)) * 100:.1f}%")
        print(f"⏱️  Total Processing Time: {processing_time:.2f} seconds")
        print(f"⚡ Average Time per Message: {(processing_time / max(self.processed_messages, 1)):.3f} seconds")
        
        # Mood distribution
        print("\n🎭 MOOD DISTRIBUTION:")
        mood_counts = {"friendly": 0, "neutral": 0, "angry": 0}
        for player_state in self.player_states.values():
            mood_counts[player_state.npc_mood] += 1
        
        for mood, count in mood_counts.items():
            percentage = (count / len(self.player_states)) * 100 if self.player_states else 0
            emoji = {"friendly": "😊", "neutral": "😐", "angry": "😠"}[mood]
            print(f"  {emoji} {mood.capitalize()}: {count} players ({percentage:.1f}%)")
        
        # Top active players
        print("\n🏆 TOP 5 MOST ACTIVE PLAYERS:")
        sorted_players = sorted(
            self.player_states.values(), 
            key=lambda x: x.message_count, 
            reverse=True
        )[:5]
        
        for i, player in enumerate(sorted_players, 1):
            emoji = {"friendly": "😊", "neutral": "😐", "angry": "😠"}[player.npc_mood]
            print(f"  {i}. Player {player.player_id}: {player.message_count} messages {emoji}")
        
        print(f"\n📝 Detailed logs saved to: {self.log_file}")
        print(f"📝 Responses saved to: {self.responses_file}")
        print("="*80)
        
        # Log summary
        logging.info("Processing completed", extra={
            "component": "system",
            "action": "process_complete",
            "total_messages": self.processed_messages,
            "unique_players": len(self.player_states),
            "api_successes": self.api_successes,
            "api_failures": self.api_failures,
            "processing_time": processing_time,
            "success_rate": (self.api_successes / max(self.processed_messages, 1)) * 100
        })


def create_sample_data(file_path: str = "players.json") -> None:
    """Create sample players.json file with 100 messages."""
    import random
    from datetime import timedelta
    
    # Sample player messages with various sentiments
    sample_messages = [
        # Friendly messages
        "Hi! Can you please help me with my quest?",
        "Thank you so much for your assistance!",
        "You're awesome, I really appreciate this!",
        "This is great, thanks for explaining!",
        "I love how helpful you are!",
        "You're the best NPC ever, thank you!",
        "Please help me find the treasure!",
        "Thanks for being so patient with me!",
        "You're wonderful, I'm grateful for your help!",
        "Great job explaining that, thanks!",
        
        # Angry messages  
        "This is useless, you're no help at all!",
        "You're stupid and waste my time!",
        "This is so annoying, move faster!",
        "You're slow and terrible at your job!",
        "I hate dealing with useless NPCs like you!",
        "This is the worst service ever!",
        "You're a waste of space, get out of my way!",
        "Stop being so annoying and help me!",
        "This is garbage, you're no good!",
        "You suck at this, do better!",
        
        # Neutral messages
        "Where can I find the blacksmith?",
        "How do I get to the next town?",
        "What items do you have for sale?",
        "Can you tell me about this area?",
        "I need information about the quest.",
        "What are the rules of this game?",
        "How do I level up my character?",
        "Where is the nearest inn?",
        "What can you tell me about magic?",
        "I'm looking for rare materials."
    ]
    
    messages = []
    base_time = datetime.now(timezone.utc)
    
    # Generate 100 messages from 50 different players
    for i in range(100):
        player_id = (i % 50) + 1  # 50 different players
        message_text = random.choice(sample_messages)
        
        # Add some variation to messages
        if random.random() < 0.3:  # 30% chance to modify message
            variations = [
                f"{message_text} Please respond quickly.",
                f"Hey, {message_text.lower()}",
                f"{message_text} I'm in a hurry!",
                f"Um, {message_text.lower()}",
                f"{message_text} Can you help?"
            ]
            message_text = random.choice(variations)
        
        # Generate timestamp (some out of order)
        time_offset = timedelta(seconds=i * 30 + random.randint(-60, 60))
        timestamp = (base_time + time_offset).isoformat().replace('+00:00', 'Z')
        
        messages.append({
            "player_id": player_id,
            "text": message_text,
            "timestamp": timestamp
        })
    
    # Shuffle messages to simulate out-of-order delivery
    random.shuffle(messages)
    
    # Save to file
    data = {"messages": messages}
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created sample data file: {file_path} with {len(messages)} messages")


def main():
    """Main function to run the NPC Chat System."""
    print("🤖 AI-Agent NPC Chat System v1.0.0")
    print("Using Groq LLaMA3 API for intelligent responses")
    print("="*60)
    
    # Enhanced environment variable handling
    try:
        from dotenv import load_dotenv, find_dotenv
        env_file = find_dotenv(raise_error_if_not_found=True)
        load_dotenv(env_file)
        print(f"✅ Loaded environment variables from: {env_file}")
    except Exception as e:
        print(f"⚠️  Warning: Error loading .env file: {e}")
    
    # Check for Groq API key with more detailed feedback
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️  Warning: GROQ_API_KEY environment variable not set!")
        print("💡 Create a .env file with your API key:")
        print("   GROQ_API_KEY=your-api-key-here")
        print("🔄 Continuing with fallback responses only...\n")
    else:
        masked_key = f"{api_key[:4]}...{api_key[-4:]}"
        print(f"✅ Groq API key found: {masked_key}")
        if not GROQ_AVAILABLE:
            print("❌ Groq library not installed. Install with: pip install groq")
            print("🔄 Continuing with fallback responses only...\n")
        else:
            print("✅ Groq library available")

    # Check if players.json exists, create if not
    json_file = "players.json"
    if not Path(json_file).exists():
        print(f"📝 {json_file} not found. Creating sample data...")
        create_sample_data(json_file)
        print()
    
    try:
        # Initialize and run the NPC system
        npc_system = NPCChatSystem(api_key=api_key)
        npc_system.process_all_messages(json_file)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Processing interrupted by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logging.error(f"Fatal error in main: {e}", extra={
            "component": "main",
            "action": "fatal_error",
            "exception": str(e)
        })


if __name__ == "__main__":
    main()
