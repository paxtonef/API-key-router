from openai import OpenAI
import os
import json
import time
import tiktoken
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress
from rich import box
import argparse
import yaml

console = Console(width=100)

# Predefined list of free AI models from OpenRouter with additional info
AI_MODELS = {
    "1": {
        "id": "google/gemini-exp-1206:free",
        "name": "Google Gemini",
        "description": "Google's experimental model - good for creative tasks",
        "context_length": 32000
    },
    "2": {
        "id": "mistral/mistral-7b-instruct:free",
        "name": "Mistral 7B",
        "description": "Efficient open-source model - balanced performance",
        "context_length": 8000
    },
    "3": {
        "id": "qwen/qwen-turbo:free",
        "name": "Qwen Turbo",
        "description": "Fast model from Alibaba - good for quick responses",
        "context_length": 8000
    },
    "4": {
        "id": "openai/o3-mini:free",
        "name": "OpenAI O3-mini",
        "description": "OpenAI's smaller model - good all-rounder",
        "context_length": 8000
    },
    "5": {
        "id": "perplexity/sonar:free",
        "name": "Perplexity Sonar",
        "description": "Focused on accurate information retrieval",
        "context_length": 12000
    },
    "6": {
        "id": "anthropic/claude-2:free",
        "name": "Claude 2",
        "description": "Anthropic's model - strong reasoning capabilities",
        "context_length": 100000
    }
}

# Add system prompts for different purposes
SYSTEM_PROMPTS = {
    "general": "You are a helpful, harmless, and honest AI assistant.",
    "coding": "You are a helpful coding assistant. Provide clear, efficient, and well-commented code examples.",
    "creative": "You are a creative assistant. Be imaginative and original in your responses.",
    "academic": "You are an academic assistant. Provide well-researched, thorough responses with citations when possible.",
    "concise": "You are a concise assistant. Provide brief, to-the-point responses without unnecessary elaboration.",
}

# Token counters - approximation using tiktoken
def count_tokens(text, model="gpt-4"):
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback approximation if model-specific tokenizer not available
        return len(text.split()) * 1.3  # Rough approximation

def calculate_total_tokens(messages):
    """Calculate the total tokens used in the conversation."""
    total = 0
    for message in messages:
        total += count_tokens(message.get("content", ""))
    return int(total)

# Configuration management
def load_config():
    """Load configuration from file or create default."""
    config_path = os.path.expanduser("~/.openrouter_cli_config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            console.print(f"[yellow]Error loading config: {e}. Using defaults.[/yellow]")
    
    # Default configuration
    return {
        "default_model": "1",
        "temperature": 0.7,
        "max_tokens": 1000,
        "conversation_path": "conversations",
        "current_conversation": "default",
        "streaming": True,
        "system_prompt": "general"
    }

def save_config(config):
    """Save configuration to file."""
    config_path = os.path.expanduser("~/.openrouter_cli_config.yaml")
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config, f)
    except Exception as e:
        console.print(f"[red]Error saving config: {e}[/red]")

# Conversation management
def get_conversation_path(config, conversation_name=None):
    """Get the path for a specific conversation."""
    if conversation_name is None:
        conversation_name = config["current_conversation"]
    
    conversation_dir = os.path.expanduser(f"~/{config['conversation_path']}")
    os.makedirs(conversation_dir, exist_ok=True)
    return os.path.join(conversation_dir, f"{conversation_name}.json")

def load_conversation(config, conversation_name=None):
    """Load a conversation from file."""
    conversation_path = get_conversation_path(config, conversation_name)
    
    if os.path.exists(conversation_path):
        try:
            with open(conversation_path, "r") as f:
                return json.load(f)
        except Exception as e:
            console.print(f"[yellow]Error loading conversation: {e}. Starting fresh.[/yellow]")
    
    # Start with system message if specified
    system_prompt = SYSTEM_PROMPTS[config["system_prompt"]]
    return [{"role": "system", "content": system_prompt}]

def save_conversation(messages, config, conversation_name=None):
    """Save a conversation to file."""
    conversation_path = get_conversation_path(config, conversation_name)
    
    try:
        with open(conversation_path, "w") as f:
            json.dump(messages, f, indent=4)
    except Exception as e:
        console.print(f"[red]Error saving conversation: {e}[/red]")

def list_conversations(config):
    """List all saved conversations."""
    conversation_dir = os.path.expanduser(f"~/{config['conversation_path']}")
    if not os.path.exists(conversation_dir):
        return []
    
    conversations = []
    for file in os.listdir(conversation_dir):
        if file.endswith(".json"):
            conversations.append(file[:-5])  # Remove .json extension
    return conversations

def display_model_options():
    """Displays available AI models in a table."""
    table = Table(title="🔹 Available Free AI Models", show_header=True, header_style="bold cyan", box=box.ROUNDED)
    table.add_column("Option", style="bold yellow", justify="center", width=6)
    table.add_column("Model", style="bold green", width=15)
    table.add_column("Description", style="magenta")
    table.add_column("Context", style="cyan", justify="right", width=10)
    
    for key, model in AI_MODELS.items():
        table.add_row(
            key, 
            model["name"], 
            model["description"],
            f"{model['context_length']} tokens"
        )
    console.print(table)

def display_system_prompts():
    """Displays available system prompts in a table."""
    table = Table(title="📝 Available System Prompts", show_header=True, header_style="bold cyan", box=box.ROUNDED)
    table.add_column("Name", style="bold yellow")
    table.add_column("Description", style="magenta")
    
    for key, prompt in SYSTEM_PROMPTS.items():
        table.add_row(key, prompt)
    console.print(table)

def display_help():
    """Display available commands."""
    commands = [
        ("/quit", "Exit the application"),
        ("/clear", "Clear the current conversation"),
        ("/switch", "Switch to a different model"),
        ("/save", "Save the current conversation"),
        ("/load <name>", "Load a different conversation"),
        ("/new <name>", "Start a new conversation"),
        ("/list", "List all saved conversations"),
        ("/system", "Change the system prompt"),
        ("/temp <0.0-1.0>", "Change the temperature"),
        ("/export <path>", "Export conversation to markdown"),
        ("/tokens", "Show token usage for current conversation"),
        ("/config", "Show current configuration"),
        ("/help", "Show this help message")
    ]
    
    table = Table(title="💬 Available Commands", show_header=True, header_style="bold cyan", box=box.ROUNDED)
    table.add_column("Command", style="bold green")
    table.add_column("Description", style="magenta")
    
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    
    console.print(table)

def export_to_markdown(messages, filepath):
    """Export conversation to markdown file."""
    try:
        with open(filepath, "w") as f:
            f.write("# OpenRouter CLI Chat Export\n\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for message in messages:
                if message["role"] == "system":
                    continue  # Skip system messages
                    
                if message["role"] == "user":
                    f.write(f"## User\n\n{message['content']}\n\n")
                else:
                    f.write(f"## Assistant\n\n{message['content']}\n\n")
        
        console.print(f"[green]Conversation exported to {filepath}[/green]")
    except Exception as e:
        console.print(f"[red]Error exporting conversation: {e}[/red]")

def get_streaming_response(client, messages, model, config):
    """Get streaming response from the API."""
    # Start spinning animation
    with Progress() as progress:
        task = progress.add_task("[cyan]Thinking...", total=None)
        
        # Start response stream
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            stream=True
        )
        
        progress.update(task, visible=False)
        
        # Print response as it comes
        console.print("\n[bold cyan]🤖 AI:[/bold cyan]", end=" ")
        
        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                console.print(content, end="")
                full_response += content
        
        console.print("\n")
        return full_response

def process_command(input_text, messages, client, config, current_model):
    """Process command inputs starting with /."""
    parts = input_text.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    if command == "/quit":
        return True, messages, current_model
    
    elif command == "/clear":
        system_prompt = next((m for m in messages if m["role"] == "system"), None)
        if system_prompt:
            messages = [system_prompt]
        else:
            system_message = SYSTEM_PROMPTS[config["system_prompt"]]
            messages = [{"role": "system", "content": system_message}]
        console.print("[green]Conversation cleared, system prompt retained.[/green]")
    
    elif command == "/switch":
        display_model_options()
        model_choice = Prompt.ask("🟢 Choose a model by number", choices=AI_MODELS.keys())
        current_model = AI_MODELS[model_choice]["id"]
        config["default_model"] = model_choice
        save_config(config)
        console.print(f"[green]Switched to model: {AI_MODELS[model_choice]['name']}[/green]")
    
    elif command == "/save":
        name = args if args else config["current_conversation"]
        save_conversation(messages, config, name)
        console.print(f"[green]Conversation saved as '{name}'[/green]")
    
    elif command == "/load":
        if not args:
            conversations = list_conversations(config)
            if not conversations:
                console.print("[yellow]No saved conversations found.[/yellow]")
                return False, messages, current_model
            
            table = Table(title="Saved Conversations", show_header=True)
            table.add_column("Name", style="cyan")
            for conv in conversations:
                table.add_row(conv)
            console.print(table)
            
            args = Prompt.ask("Enter conversation name to load")
        
        loaded_messages = load_conversation(config, args)
        if loaded_messages:
            messages = loaded_messages
            config["current_conversation"] = args
            save_config(config)
            console.print(f"[green]Loaded conversation '{args}'[/green]")
        else:
            console.print(f"[yellow]Could not load conversation '{args}'[/yellow]")
    
    elif command == "/new":
        name = args if args else f"conversation_{int(time.time())}"
        system_message = SYSTEM_PROMPTS[config["system_prompt"]]
        messages = [{"role": "system", "content": system_message}]
        config["current_conversation"] = name
        save_config(config)
        console.print(f"[green]Started new conversation '{name}'[/green]")
    
    elif command == "/list":
        conversations = list_conversations(config)
        if not conversations:
            console.print("[yellow]No saved conversations found.[/yellow]")
            return False, messages, current_model
        
        table = Table(title="Saved Conversations", show_header=True)
        table.add_column("Name", style="cyan")
        table.add_column("Current", style="green")
        
        for conv in conversations:
            current = "✓" if conv == config["current_conversation"] else ""
            table.add_row(conv, current)
        
        console.print(table)
    
    elif command == "/system":
        if not args:
            display_system_prompts()
            args = Prompt.ask("Choose a system prompt", choices=list(SYSTEM_PROMPTS.keys()))
        
        if args in SYSTEM_PROMPTS:
            # Update or add system message
            system_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
            if system_idx is not None:
                messages[system_idx]["content"] = SYSTEM_PROMPTS[args]
            else:
                messages.insert(0, {"role": "system", "content": SYSTEM_PROMPTS[args]})
            
            config["system_prompt"] = args
            save_config(config)
            console.print(f"[green]System prompt changed to '{args}'[/green]")
        else:
            console.print(f"[yellow]Unknown system prompt '{args}'[/yellow]")
    
    elif command == "/temp":
        try:
            temp = float(args)
            if 0.0 <= temp <= 1.0:
                config["temperature"] = temp
                save_config(config)
                console.print(f"[green]Temperature set to {temp}[/green]")
            else:
                console.print("[yellow]Temperature must be between 0.0 and 1.0[/yellow]")
        except ValueError:
            console.print("[yellow]Please provide a valid number between 0.0 and 1.0[/yellow]")
    
    elif command == "/export":
        path = args if args else f"chat_export_{int(time.time())}.md"
        export_to_markdown(messages, path)
    
    elif command == "/tokens":
        total_tokens = calculate_total_tokens(messages)
        console.print(f"[cyan]Current conversation uses approximately {total_tokens} tokens[/cyan]")
        
        # Find current model's context limit
        for model_info in AI_MODELS.values():
            if model_info["id"] == current_model:
                context_limit = model_info["context_length"]
                percentage = (total_tokens / context_limit) * 100
                console.print(f"[cyan]Using {percentage:.1f}% of {context_limit} token context window[/cyan]")
                break
    
    elif command == "/config":
        table = Table(title="Current Configuration", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in config.items():
            if key == "default_model":
                model_name = AI_MODELS[value]["name"] if value in AI_MODELS else value
                table.add_row(key, f"{value} ({model_name})")
            else:
                table.add_row(key, str(value))
        
        console.print(table)
    
    elif command == "/help":
        display_help()
    
    else:
        console.print(f"[yellow]Unknown command: {command}[/yellow]")
        display_help()
    
    return False, messages, current_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="OpenRouter CLI Chat Interface")
    parser.add_argument("--model", type=str, help="Model ID to use")
    parser.add_argument("--temperature", type=float, help="Temperature setting (0.0-1.0)")
    parser.add_argument("--conversation", type=str, help="Conversation to load")
    parser.add_argument("--system", type=str, help="System prompt to use")
    parser.add_argument("--no-stream", action="store_true", help="Disable response streaming")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Override config with command line arguments
    if args.model:
        config["default_model"] = args.model
    if args.temperature is not None:
        config["temperature"] = max(0.0, min(1.0, args.temperature))
    if args.conversation:
        config["current_conversation"] = args.conversation
    if args.system and args.system in SYSTEM_PROMPTS:
        config["system_prompt"] = args.system
    if args.no_stream:
        config["streaming"] = False
    
    # Display welcome banner
    console.print(Panel.fit(
        "[bold green]OpenRouter CLI Chat[/bold green]\n"
        "[cyan]Access multiple AI models through a simple interface[/cyan]",
        title="Welcome", subtitle="v2.0"
    ))
    
    # Securely request API key
    api_key_env = os.environ.get("OPENROUTER_API_KEY")
    if api_key_env:
        YOUR_API_KEY = api_key_env
        console.print("[green]Using API key from environment variable[/green]")
    else:
        YOUR_API_KEY = Prompt.ask('🔑 Enter your OpenRouter API key', password=True)
    
    # OpenRouter API client setup
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=YOUR_API_KEY,
    )
    
    # Show available models and let the user choose one
    display_model_options()
    
    model_choice = config["default_model"]
    valid_choice = model_choice in AI_MODELS
    if not valid_choice:
        model_choice = Prompt.ask("🟢 Choose a model by number", choices=AI_MODELS.keys())
        config["default_model"] = model_choice
        save_config(config)
    
    selected_model = AI_MODELS[model_choice]["id"]
    console.print(f"\n[bold cyan]✅ Using model:[/bold cyan] {AI_MODELS[model_choice]['name']}\n")
    
    # Load conversation history
    messages = load_conversation(config)
    
    # Show startup info
    console.print(f"[green]Current conversation: {config['current_conversation']}[/green]")
    console.print(f"[green]System prompt: {config['system_prompt']}[/green]")
    console.print("\n[bold green]💬 Chatbot Ready! Type /help for commands or /quit to exit.[/bold green]")
    
    # Main chat loop
    while True:
        try:
            user_input = Prompt.ask("\n[bold magenta]You[/bold magenta]")
            
            # Check for commands
            if user_input.startswith("/"):
                should_quit, messages, selected_model = process_command(
                    user_input, messages, client, config, selected_model
                )
                if should_quit:
                    save_conversation(messages, config)
                    console.print("\n[bold red]👋 Exiting chatbot. See you next time![/bold red]")
                    break
                continue
            
            if not user_input.strip():
                console.print("[bold yellow]⚠️ Empty input detected. Please enter a message.[/bold yellow]")
                continue
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Check token count before making the API call
            total_tokens = calculate_total_tokens(messages)
            
            # Find current model's context limit
            context_limit = 8000  # Default
            for model_info in AI_MODELS.values():
                if model_info["id"] == selected_model:
                    context_limit = model_info["context_length"]
                    break
            
            # Warn if approaching context limit
            if total_tokens > context_limit * 0.8:
                console.print(
                    f"[yellow]Warning: Approaching context limit ({total_tokens}/{context_limit} tokens)[/yellow]"
                )
                
                # Trim conversation if needed and user agrees
                if total_tokens > context_limit * 0.9:
                    if Confirm.ask("Would you like to trim older messages to stay within context limits?"):
                        # Keep system message and last several exchanges
                        system_message = next((m for m in messages if m["role"] == "system"), None)
                        trimmed_messages = messages[-10:]  # Keep last 5 exchanges (10 messages)
                        
                        if system_message and system_message not in trimmed_messages:
                            trimmed_messages.insert(0, system_message)
                        
                        messages = trimmed_messages
                        console.print("[green]Conversation trimmed to fit context window[/green]")
            
            # Call OpenRouter API with selected model
            try:
                if config["streaming"]:
                    response = get_streaming_response(client, messages, selected_model, config)
                else:
                    completion = client.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        temperature=config["temperature"],
                        max_tokens=config["max_tokens"]
                    )
                    response = completion.choices[0].message.content
                    console.print(f"\n[bold cyan]🤖 AI:[/bold cyan]")
                    
                    # Parse and render any markdown in the response
                    try:
                        md = Markdown(response)
                        console.print(md)
                    except:
                        console.print(response)
                
                # Add response to history
                messages.append({"role": "assistant", "content": response})
                
                # Auto-save conversation after each exchange
                save_conversation(messages, config)
                
            except Exception as e:
                console.print(f"\n[bold red]❌ API Error:[/bold red] {e}")
                
        except KeyboardInterrupt:
            if Confirm.ask("\n[bold yellow]Do you want to exit?[/bold yellow]"):
                save_conversation(messages, config)
                console.print("\n[bold red]👋 Exiting chatbot. See you next time![/bold red]")
                break
            else:
                console.print("[green]Continuing...[/green]")
        
        except Exception as e:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
            continue

if __name__ == "__main__":
    main()
