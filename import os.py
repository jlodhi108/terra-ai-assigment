import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# Attempt to import the Groq library
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

class GroqAIClient:
    """Handles communication with Groq AI API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3-70b-8192"):
        # Enhanced API key validation and initialization
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.client = None
        
        if not GROQ_AVAILABLE:
            logging.error("Groq library not installed. Run: pip install groq")
            return
            
        if not self.api_key:
            logging.error("No API key provided. Ensure GROQ_API_KEY is set in .env")
            return
            
        if not self.api_key.startswith('gsk_'):
            logging.error(f"Invalid API key format: {self.api_key[:4]}... - Should start with 'gsk_'")
            return
            
        try:
            # Initialize client
            self.client = Groq(api_key=self.api_key)
            
            # Test API connection
            test_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model=self.model,
                max_tokens=1,
                temperature=0.7
            )
            
            if test_completion and hasattr(test_completion, 'choices'):
                logging.info("✅ Successfully tested Groq API connection")
                print("✅ Groq API connection successful!")
            else:
                raise ValueError("Invalid API response format")
                
        except Exception as e:
            logging.error(f"Failed to initialize Groq client: {str(e)}", extra={
                "component": "groq_client",
                "action": "initialize",
                "error": str(e),
                "api_key_length": len(self.api_key) if self.api_key else 0
            })
            self.client = None
            print(f"❌ Error initializing Groq client: {str(e)}")

def main():
    """Main function to run the NPC Chat System."""
    print("🤖 AI-Agent NPC Chat System v1.0.0")
    print("Using Groq LLaMA3 API for intelligent responses")
    print("="*60)
    
    # Enhanced environment variable loading
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        try:
            load_dotenv(env_path, override=True)
            print(f"✅ Loaded environment from: {env_path}")
        except Exception as e:
            print(f"❌ Error loading .env: {e}")
    else:
        print(f"❌ No .env file found at: {env_path}")

    # Initialize the Groq AI client
    api_key = os.getenv("GROQ_API_KEY")
    client = GroqAIClient(api_key=api_key)
    
    # ... existing code for the chat system ...