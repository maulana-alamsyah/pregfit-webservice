import hashlib
import sys

def compute_sha256_hash(data):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(data.encode('utf-8'))
    return sha256_hash.hexdigest()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 hash256.py <input_string>")
        sys.exit(1)
    
    input_string = sys.argv[1]
    hash_value = compute_sha256_hash(input_string)
    print(f"SHA-256 hash dari '{input_string}' adalah: {hash_value}")

