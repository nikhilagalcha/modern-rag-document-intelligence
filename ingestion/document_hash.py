
import hashlib


def calculate_file_hash(file_path):
    """
    Calculate SHA-256 hash of a file.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while True:
            data = file.read(4096)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()
