# terra-ai-assigment
In this assignment, you’ll build a simple **AI-powered NPC chat system**.  (NPC = Non-playing character)

---

## 📝 Task  

Write a Python program that:  

1. Reads a JSON file of **100 player chat messages** (`players.json`).  
   - Each message has:  
     - `player_id` (integer)  
     - `text` (string)  
     - `timestamp` (ISO 8601 string, e.g. `"2025-08-26T15:01:10"`)  

2. For each incoming message:  
   - Call the **OpenAI GPT API (`gpt-3.5-turbo`)** to generate a short NPC reply. (You can use any other AI model too)
   - Maintain **conversation state per player**:  
     - Keep the last 3 messages for each player.  
     - Pass this state along with the current message so replies feel consistent.  
   - Track an NPC “mood” per player that can shift over time (start as `"neutral"`, then switch to `"friendly"` or `"angry"` depending on what the player says).  
     - Example: If the player asks for help → `"friendly"`; if the player insults the NPC → `"angry"`.  
     - Include this mood in the prompt to GPT.  

3. Messages may arrive **out of order** (timestamps will not be sorted).  
   - Your program should process them in **chronological order**.  

4. Log the results (console or file):  
   - `player_id`  
   - message text  
   - NPC reply  
   - state used (last 3 messages)  
   - NPC mood  
   - timestamp   

---

## 📂 Input Example (`players.json`)  

```json
[
  {"player_id": 1, "text": "Hello there!", "timestamp": "2025-08-26T15:01:10"},
  {"player_id": 2, "text": "Where should I go now?", "timestamp": "2025-08-26T15:01:05"},
  {"player_id": 1, "text": "Tell me more about this village.", "timestamp": "2025-08-26T15:01:20"},
  {"player_id": 2, "text": "You are useless!", "timestamp": "2025-08-26T15:01:25"},
  {"player_id": 3, "text": "Do you have a quest for me?", "timestamp": "2025-08-26T15:01:15"}
]
```
---

## Stretch Opportunities

- If you want to, you are free to add additional flavour or improvements beyond the basics.
- This is not required, but it’s a chance to show how you think creatively about the problem.

## ✅ Baseline Expectations
Your script should run end-to-end and:
- Process all 100 messages in chronological order.
- Maintain per-player state (last 3 messages).
- Track and update a mood variable for each NPC.
- Produce structured logs with all required fields.

This is the minimum requirement for the assignment.


## 🚀 Getting Started
You can use **any text generation model** you prefer — commercial (OpenAI, Claude, Gemini, etc.) or open-source (LLaMA, Mistral, etc.).  

Install the client library for your chosen model.  
For example, if you use OpenAI:  
```
pip install openai
```
Place your code in npc_chat.py and run it:

```
python npc_chat.py
```
## 📦 Deliverables

- Submit your Python code (`npc_chat.py`).  
- Your program should read `players.json` and output logs as described.  
- If you used AI tools (ChatGPT, Claude, Gemini, Copilot, or open-source models) to help, please also share the **prompt chain or conversation link** you used.  
  - If the tool supports shareable links, include the link.  
  - If not, copy the prompts/responses into a file (`ai_prompts.txt`) and include it in your submission.  

👉 Using AI is encouraged. We’re interested in seeing *how you think with AI*, how you structure prompts, and how you refine outputs.  
**Please don’t claim you avoided AI use to “look better” — this does not give extra credit.**

## 📤 How to Submit

- Clone this repository to your own GitHub account. (create one if you haven't)
- Add your solution (npc_chat.py and any supporting files).
    1. Include: Your code `npc_chat.py`
    2. Include: the output logs (sample run)
    3. Include: Links to all chatgpt / claude / any other AI tool conversations that you used or `ai_prompts.txt`
- Make your repository public.
- Share the repo link to us
