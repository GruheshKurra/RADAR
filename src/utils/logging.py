"""Simple logging utilities for RADAR training."""


def print_section(title: str, body: str = None, width: int = 70):
    """Print a formatted section header with optional body."""
    print(f"\n{'=' * width}")
    print(title)
    print('=' * width)
    if body:
        print(body)


def print_subsection(title: str, width: int = 70):
    """Print a subsection separator."""
    print(f"\n{'-' * width}")
    print(title)
    print('-' * width)


def print_result(items: dict, prefix: str = ""):
    """Print key-value results in a clean format."""
    for key, value in items.items():
        if isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        elif isinstance(value, int):
            print(f"{prefix}{key}: {value:,}")
        else:
            print(f"{prefix}{key}: {value}")


def print_complete(title: str, summary: dict = None, width: int = 70):
    """Print a completion message with optional summary."""
    print(f"\n{'=' * width}")
    print(f"✓ {title}")
    print('=' * width)
    if summary:
        print_result(summary)
    print('=' * width + '\n')
