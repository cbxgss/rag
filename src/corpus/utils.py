import hashlib, io, dill, base58


def extract_title(new_content: str):
    title = new_content.split("\n", 1)[0].strip().strip("#### ").strip("\"")
    return title


def hash_object(o) -> str:
    """Returns a character hash code of arbitrary Python objects."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()
