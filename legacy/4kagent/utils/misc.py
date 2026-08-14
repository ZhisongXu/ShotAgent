from pathlib import Path
from base64 import b64encode


def encode_img(img_path: Path | str) -> str:
    """Encodes image to base64."""    
    with open(img_path, "rb") as img_file:
        b64code = b64encode(img_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{b64code}"
    

def sorted_glob(dir_path: Path, pattern: str = "*") -> list[Path]:
    assert dir_path.is_dir(), f"{dir_path} is not a directory."
    return sorted(list(dir_path.glob(pattern)))


def sorted_rglob(dir_path: Path, pattern: str = "*") -> list[Path]:
    assert dir_path.is_dir(), f"{dir_path} is not a directory."
    return sorted(list(dir_path.rglob(pattern)))
