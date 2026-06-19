from schemas import RunState


def tools(state: RunState):
    permissions = state.config.environment.permissions

    def format_run_command(args):
        stdout, stderr, exitcode = state.sandbox.run(args["cmd"], timeout_seconds=120)
        output = f"OUTPUT:\n{stdout}"
        if exitcode != 0:
            output += f"\nERROR (code={exitcode}):\n{stderr}"
        return output

    def read_file(args):
        with state.sandbox.open(args["path"], "r") as f:
            return f.read()

    def write_file(args):
        with state.sandbox.open(args["path"], "w") as f:
            f.write(args["text"])

    all_tools = {
        "WEB_SEARCH": {
            "spec": {
                "type": "function",
                "function": {
                    "name": "WEB_SEARCH",
                    "description": "Use Google Search to find documentation, datasheets, example code, and existing solutions to problems you encounter.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The google search query",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            "fn": lambda args: state.sandbox.search_google(
                args["query"], num_results=5
            ),
        },
        "DOWNLOAD_MARKDOWN": {
            "spec": {
                "type": "function",
                "function": {
                    "name": "DOWNLOAD_MARKDOWN",
                    "description": "Use the urls returned by WEB_SEARCH to download the markdown of a website/PDF. Then use the READ_FILE tool to get the entire content or the RUN_COMMAND with 'grep' to search through the output.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The url returned by WEB_SEARCH or found linked in documentation",
                            },
                            "destination_path": {
                                "type": "string",
                                "description": "The path to write the markdown to",
                            },
                        },
                        "required": ["url", "destination_path"],
                    },
                },
            },
            "fn": lambda args: (
                "success"
                if state.sandbox.download_url_as_markdown(
                    args["url"], args["destination_path"], timeout_seconds=120
                )
                else "failure"
            ),
        },
        "READ_FILE": {
            "spec": {
                "type": "function",
                "function": {
                    "name": "READ_FILE",
                    "description": "Gets the entire text content of a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path to read",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            "fn": lambda args: read_file(args),
        },
        "WRITE_FILE": {
            "spec": {
                "type": "function",
                "function": {
                    "name": "WRITE_FILE",
                    "description": "Writes text content to a file. The parent directory must first exist.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path to read",
                            },
                            "text": {
                                "type": "string",
                                "description": "The content to place in the file",
                            },
                        },
                        "required": ["path", "text"],
                    },
                },
            },
            "fn": lambda args: write_file(args) or "success",
        },
        "RUN_COMMAND": {
            "spec": {
                "type": "function",
                "function": {
                    "name": "RUN_COMMAND",
                    "description": "Run a command in the sandbox container. The command is executed directly from an argv list, not through a shell, so shell syntax such as pipes, redirects, &&, and variable expansion only works if you explicitly run a shell. Examples: cmd=['python', 'test.py'] or cmd=['git', 'status']. You can write a python file first using WRITE_FILE.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {
                                "type": "array",
                                "description": "first element is the command, the rest are the arguments; the usual GNU Core Utilities are available as well as 'python' (3.9)",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["cmd"],
                    },
                },
            },
            "fn": lambda args: format_run_command(args),
        },
    }

    allowed = {}
    if permissions.network:
        allowed["WEB_SEARCH"] = all_tools["WEB_SEARCH"]
        allowed["DOWNLOAD_MARKDOWN"] = all_tools["DOWNLOAD_MARKDOWN"]
    if permissions.read_files or permissions.code:
        allowed["READ_FILE"] = all_tools["READ_FILE"]
    if permissions.write_files or permissions.code:
        allowed["WRITE_FILE"] = all_tools["WRITE_FILE"]
    if permissions.code:
        allowed["RUN_COMMAND"] = all_tools["RUN_COMMAND"]

    return allowed
