
from datetime import datetime
from pathlib import Path

def save_llm_response(
    response: str,
    filename: str | None = None,
    directory: str | Path = "./reports",
    include_timestamp: bool = True,
    metadata: dict | None = None
) -> str:
    """
    Save an LLM-generated response to a markdown file.
    
    Args:
        response (str): The LLM response text to save
        filename (str, optional): Name for the file (without extension). 
                                 If None, uses timestamp
        directory (str): Directory to save the file in. Defaults to "llm_responses"
        include_timestamp (bool): Whether to include timestamp in filename
        metadata (dict, optional): Additional metadata to include at the top of the file
    
    Returns:
        str: Path to the saved file
    """
    # create directory if it doesn't exist
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if filename is None:
        filename = f"response_{timestamp}"

    # ensure md extension
    if not filename.endswith('.md'):
        filename += '.md'

    # full file path
    filepath = directory / filename

    # prepare content
    content_parts = []

    # add metadata if provided
    if metadata:
        content_parts.append("---")
        for key, value in metadata.items():
            content_parts.append(f"{key}: {value}")
        content_parts.append(f"saved at: {datetime.now().isoformat()}")
        content_parts.append("---")
        content_parts.append("")

    # add the response
    content_parts.append(response)

    # write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content_parts))

    return str(filepath)

