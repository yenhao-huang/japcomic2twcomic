"""
輸入格式檢查工具
"""
from pathlib import Path


def check_input_format(file_path: Path) -> bool:
    """
    Check if the input file is a .jpg file.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file is a .jpg file, False otherwise
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    return file_path.suffix.lower() in {'.jpg', '.jpeg'}
