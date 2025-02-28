# API-key-router
how to access free AI APIs and leverage OpenRouter.ai
# OpenRouter CLI Chat

A powerful command-line interface for chatting with various AI models via the OpenRouter API.

![OpenRouter CLI Chat Demo](https://i.imgur.com/your-image-here.png)

## Features

- 🤖 Access to multiple free AI models through OpenRouter
- 💾 Save and load multiple conversations
- 📊 Track token usage and context window limits
- 🔄 Stream responses in real-time
- 📝 Different system prompts for various use cases
- 📤 Export conversations to markdown
- ⚙️ Persistent configuration
- 🎨 Rich text interface with syntax highlighting and markdown rendering

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get an API key from [OpenRouter](https://openrouter.ai/)

## Usage

### Quick Start

Run the script and follow the prompts:

```bash
python openrouter_chat.py
```

Enter your OpenRouter API key when prompted, select a model, and start chatting!

### Setting API Key

To avoid entering your API key each time, set it as an environment variable:

```bash
# Linux/Mac
export OPENROUTER_API_KEY="your-api-key-here"

# Windows
set OPENROUTER_API_KEY=your-api-key-here
```

### Command Line Options

```bash
python openrouter_chat.py --model 1 --temperature 0.7 --conversation my_chat --system coding --no-stream
```

Available options:
- `--model`: Model number to use (1-6)
- `--temperature`: Temperature setting (0.0-1.0)
- `--conversation`: Name of conversation to load
- `--system`: System prompt to use (general, coding, creative, academic, concise)
- `--no-stream`: Disable response streaming

## Commands

During a chat session, you can use these commands:

| Command | Description |
|---------|-------------|
| `/quit` | Exit the application |
| `/clear` | Clear the current conversation |
| `/switch` | Switch to a different model |
| `/save` | Save the current conversation |
| `/load <name>` | Load a different conversation |
| `/new <name>` | Start a new conversation |
| `/list` | List all saved conversations |
| `/system` | Change the system prompt |
| `/temp <0.0-1.0>` | Change the temperature |
| `/export <path>` | Export conversation to markdown |
| `/tokens` | Show token usage for current conversation |
| `/config` | Show current configuration |
| `/help` | Show this help message |

## Available Models

The script provides access to these free models via OpenRouter:

1. **Google Gemini**: Google's experimental model - good for creative tasks
2. **Mistral 7B**: Efficient open-source model - balanced performance
3. **Qwen Turbo**: Fast model from Alibaba - good for quick responses
4. **OpenAI O3-mini**: OpenAI's smaller model - good all-rounder
5. **Perplexity Sonar**: Focused on accurate information retrieval
6. **Claude 2**: Anthropic's model - strong reasoning capabilities

## System Prompts

Choose from different system prompts to tailor the AI's behavior:

- **general**: Helpful, harmless, and honest assistant
- **coding**: Specialized in providing clear, efficient code examples
- **creative**: More imaginative and original in responses
- **academic**: Provides well-researched, thorough responses
- **concise**: Gives brief, to-the-point answers

## Examples

### Having a creative conversation

```
$ python openrouter_chat.py --system creative

You: Write a short poem about programming

🤖 AI: 
# The Silent Symphony

In chambers of logic, fingers dance,
Across keys that speak in binary romance.
Functions bloom like flowers in spring,
Each line of code, a note that sings.

Bugs scatter like shadows from light,
As minds wrestle problems through the night.
In this world built of thought and will,
Time bends to those with craft and skill.

A universe created keystroke by keystroke,
Ideas once dreamed, now finally awoke.
```

### Getting coding help

```
$ python openrouter_chat.py --system coding

You: How do I create a simple HTTP server in Python?

🤖 AI: You can create a simple HTTP server in Python using the built-in `http.server` module. Here's how:

```python
# Simple HTTP Server in Python
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket

# Define server parameters
HOST_NAME = "localhost"
PORT = 8000

# Create the server
server = HTTPServer((HOST_NAME, PORT), SimpleHTTPRequestHandler)

print(f"Server started at http://{HOST_NAME}:{PORT}")

try:
    # Start the server
    server.serve_forever()
except KeyboardInterrupt:
    # Handle Ctrl+C gracefully
    print("\nShutting down the server...")
    server.server_close()
    print("Server shut down successfully")
```

To use this:
1. Save it as `simple_server.py`
2. Run it with `python simple_server.py`
3. Access http://localhost:8000 in your browser
4. It will serve files from the current directory
5. Press Ctrl+C to stop the server
```

## License

MIT License

## Acknowledgments

- [OpenRouter](https://openrouter.ai/) for providing access to various AI models
- [Rich](https://github.com/Textualize/rich) for the beautiful terminal interface
