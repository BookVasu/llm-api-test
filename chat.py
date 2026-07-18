import json
from pathlib import Path
from typing import Any

import requests


API_URL = "https://openrouter.ai/api/v1/chat/completions"
SCRIPT_FOLDER = Path(__file__).resolve().parent


def load_api_key() -> str:
    """Load the OpenRouter API key from api_key or api_key.txt."""
    possible_files = [
        SCRIPT_FOLDER / "api_key",
        SCRIPT_FOLDER / "api_key.txt",
    ]

    for key_file in possible_files:
        if key_file.exists():
            api_key = key_file.read_text(
                encoding="utf-8-sig"
            ).strip()

            if not api_key:
                raise ValueError(f"{key_file.name} is empty.")

            return api_key

    raise FileNotFoundError(
        "Could not find api_key or api_key.txt "
        "in the same folder as chat.py."
    )


def load_models() -> list[str]:
    """Load fallback model IDs from models.txt."""
    models_file = SCRIPT_FOLDER / "models.txt"

    if not models_file.exists():
        raise FileNotFoundError(
            "Could not find models.txt."
        )

    models: list[str] = []

    for line in models_file.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        line = line.strip()

        # Ignore empty lines and comment lines.
        if not line or line.startswith("#"):
            continue

        models.append(line)

    if not models:
        raise ValueError(
            "models.txt does not contain any model IDs."
        )

    return models


def load_system_prompt() -> str:
    """Load the system prompt from a TXT or Markdown file."""
    possible_files = [
        SCRIPT_FOLDER / "system_prompt.txt",
        SCRIPT_FOLDER / "system_prompt.md",
    ]

    for prompt_file in possible_files:
        if prompt_file.exists():
            system_prompt = prompt_file.read_text(
                encoding="utf-8-sig"
            ).strip()

            if not system_prompt:
                raise ValueError(
                    f"{prompt_file.name} is empty."
                )

            return system_prompt

    raise FileNotFoundError(
        "Could not find system_prompt.txt "
        "or system_prompt.md."
    )


def is_number(value: Any) -> bool:
    """Return True for an integer or float, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    )


def load_parameters() -> dict[str, Any]:
    """Load and validate API parameters from parameters.txt."""
    parameters_file = SCRIPT_FOLDER / "parameters.txt"

    if not parameters_file.exists():
        raise FileNotFoundError(
            "Could not find parameters.txt."
        )

    raw_content = parameters_file.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not raw_content:
        raise ValueError("parameters.txt is empty.")

    try:
        parameters = json.loads(raw_content)
    except json.JSONDecodeError as error:
        raise ValueError(
            "parameters.txt contains invalid JSON. "
            f"Line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from error

    if not isinstance(parameters, dict):
        raise ValueError(
            "parameters.txt must contain one JSON object."
        )

    # These fields are controlled directly by chat.py.
    reserved_keys = {
        "model",
        "models",
        "messages",
        "stream",
    }

    conflicting_keys = reserved_keys.intersection(
        parameters.keys()
    )

    if conflicting_keys:
        keys = ", ".join(sorted(conflicting_keys))

        raise ValueError(
            "Remove these reserved fields from "
            f"parameters.txt: {keys}"
        )

    temperature = parameters.get("temperature")

    if temperature is not None:
        if (
            not is_number(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError(
                "temperature must be a number "
                "between 0 and 2."
            )

    top_p = parameters.get("top_p")

    if top_p is not None:
        if (
            not is_number(top_p)
            or not 0 <= top_p <= 1
        ):
            raise ValueError(
                "top_p must be a number between 0 and 1."
            )

    seed = parameters.get("seed")

    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(
                "seed must be an integer."
            )

    max_tokens = parameters.get("max_tokens")

    if max_tokens is not None:
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError(
                "max_tokens must be an integer "
                "greater than or equal to 1."
            )

    max_completion_tokens = parameters.get(
        "max_completion_tokens"
    )

    if max_completion_tokens is not None:
        if (
            isinstance(max_completion_tokens, bool)
            or not isinstance(max_completion_tokens, int)
            or max_completion_tokens < 1
        ):
            raise ValueError(
                "max_completion_tokens must be an integer "
                "greater than or equal to 1."
            )

    return parameters


def load_input_text() -> str:
    """Load a user message from input.txt."""
    input_file = SCRIPT_FOLDER / "input.txt"

    if not input_file.exists():
        raise FileNotFoundError(
            "Could not find input.txt."
        )

    input_text = input_file.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not input_text:
        raise ValueError("input.txt is empty.")

    return input_text


def create_new_conversation(
    system_prompt: str,
) -> list[dict[str, str]]:
    """Create a new conversation containing the system prompt."""
    return [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]


def format_api_error(response: requests.Response) -> str:
    """Extract a readable error from an API response."""
    try:
        error_details = response.json()
        formatted_details = json.dumps(
            error_details,
            indent=2,
            ensure_ascii=False,
        )
    except ValueError:
        formatted_details = response.text.strip()

    if not formatted_details:
        formatted_details = "No error details returned."

    return (
        f"OpenRouter error {response.status_code}:\n"
        f"{formatted_details}"
    )


def stream_response(
    api_key: str,
    models: list[str],
    parameters: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, str]:
    """Send a streamed request and return the completed response."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Load generation settings from parameters.txt.
    payload = dict(parameters)

    # These required fields are controlled by this script.
    payload.update(
        {
            "models": models,
            "messages": messages,
            "stream": True,
        }
    )

    full_response = ""
    selected_model = "unknown"
    response_started = False

    with requests.post(
        API_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=(15, 300),
    ) as response:
        if not response.ok:
            raise RuntimeError(
                format_api_error(response)
            )

        for line in response.iter_lines(
            decode_unicode=True
        ):
            if not line:
                continue

            # Ignore SSE comments and other non-data lines.
            if not line.startswith("data:"):
                continue

            data = line.removeprefix("data:").strip()

            if not data:
                continue

            if data == "[DONE]":
                break

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            if event.get("model"):
                selected_model = event["model"]

            if "error" in event:
                error = event["error"]

                if isinstance(error, dict):
                    code = error.get("code", "unknown")
                    message = error.get(
                        "message",
                        str(error),
                    )
                else:
                    code = "unknown"
                    message = str(error)

                raise RuntimeError(
                    f"Provider error {code}: {message}"
                )

            choices = event.get("choices", [])

            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content")

            if not content:
                continue

            if not response_started:
                print("Assistant: ", end="", flush=True)
                response_started = True

            print(content, end="", flush=True)
            full_response += content

    if response_started:
        print()

    return full_response, selected_model


def show_configuration(
    models: list[str],
    parameters: dict[str, Any],
    system_prompt: str,
) -> None:
    """Display the currently loaded settings."""
    print("\nModel priority:")

    for number, model in enumerate(models, start=1):
        print(f"  {number}. {model}")

    print("\nParameters:")
    print(
        json.dumps(
            parameters,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nSystem prompt loaded: "
        f"{len(system_prompt):,} characters"
    )


def show_commands() -> None:
    """Display available terminal commands."""
    print("\nCommands:")
    print("  /input   Send the contents of input.txt")
    print("  /clear   Clear conversation history")
    print("  /reload  Reload configuration files")
    print("  /config  Show current configuration")
    print("  /help    Show available commands")
    print("  /exit    Exit the program")
    print()


def main() -> None:
    """Run the terminal chat program."""
    try:
        api_key = load_api_key()
        models = load_models()
        parameters = load_parameters()
        system_prompt = load_system_prompt()
    except (FileNotFoundError, ValueError) as error:
        print(f"Configuration error: {error}")
        return

    messages = create_new_conversation(
        system_prompt
    )

    print("OpenRouter terminal chat")
    show_configuration(
        models,
        parameters,
        system_prompt,
    )
    show_commands()

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in {
            "/exit",
            "/quit",
            "exit",
            "quit",
        }:
            print("Goodbye!")
            break

        if command == "/help":
            show_commands()
            continue

        if command == "/clear":
            messages = create_new_conversation(
                system_prompt
            )

            print("Conversation history cleared.\n")
            continue

        if command == "/config":
            show_configuration(
                models,
                parameters,
                system_prompt,
            )
            print()
            continue

        if command == "/reload":
            try:
                api_key = load_api_key()
                models = load_models()
                parameters = load_parameters()
                system_prompt = load_system_prompt()

                messages = create_new_conversation(
                    system_prompt
                )

                print("Configuration files reloaded.")
                print("Conversation history cleared.")

                show_configuration(
                    models,
                    parameters,
                    system_prompt,
                )
                print()

            except (
                FileNotFoundError,
                ValueError,
            ) as error:
                print(f"Reload error: {error}\n")

            continue

        if command == "/input":
            try:
                user_input = load_input_text()
            except (
                FileNotFoundError,
                ValueError,
            ) as error:
                print(f"Input error: {error}\n")
                continue

            print(
                "\nYou [input.txt] "
                f"({len(user_input):,} characters):"
            )
            print(user_input)
            print()

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            assistant_response, selected_model = (
                stream_response(
                    api_key=api_key,
                    models=models,
                    parameters=parameters,
                    messages=messages,
                )
            )

        except requests.Timeout:
            print(
                "\nNetwork error: The request timed out.\n"
            )
            messages.pop()
            continue

        except requests.ConnectionError as error:
            print(
                "\nNetwork connection error: "
                f"{error}\n"
            )
            messages.pop()
            continue

        except requests.RequestException as error:
            print(f"\nNetwork error: {error}\n")
            messages.pop()
            continue

        except RuntimeError as error:
            print(f"\nAPI error: {error}\n")
            messages.pop()
            continue

        if not assistant_response:
            print(
                "The model returned an empty response.\n"
            )
            messages.pop()
            continue

        messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )

        print(f"[Used model: {selected_model}]\n")


if __name__ == "__main__":
    main()