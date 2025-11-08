"""
Demo script for testing the LLM-powered etymology generation.

Usage:
1. Set up your .env file with API keys
2. Start the server: uvicorn api.main:app --reload --port 8000
3. Run this script: python demo_llm_query.py
"""
import requests
import json
from typing import List


BASE_URL = "http://localhost:8000"


def pretty_print(data):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2, ensure_ascii=False))


def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("1. HEALTH CHECK")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    pretty_print(response.json())


def test_generate_evolution(word: str, eras: List[str]):
    """Test generating word evolution using LLM."""
    print("\n" + "="*60)
    print(f"2. GENERATE EVOLUTION FOR '{word.upper()}'")
    print("="*60)
    
    payload = {
        "word": word,
        "eras": eras,
        "num_examples": 5
    }
    
    response = requests.post(
        f"{BASE_URL}/generate-evolution",
        json=payload
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nWord: {data['word']}")
        print(f"Source: {data['source']}")
        print(f"\nEvolution across eras:")
        
        for era in data['eras']:
            print(f"\n  {era}:")
            if era in data['evolution']:
                for i, example in enumerate(data['evolution'][era], 1):
                    print(f"    {i}. {example}")
            else:
                print("    (no data)")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_build_embeddings(word: str, eras: List[str]):
    """Test complete pipeline: generate evolution + create embeddings."""
    print("\n" + "="*60)
    print(f"3. BUILD EMBEDDINGS FOR '{word.upper()}'")
    print("="*60)
    
    payload = {
        "word": word,
        "eras": eras,
        "num_examples": 5
    }
    
    print(f"Generating evolution data and creating embeddings...")
    print(f"This may take 10-30 seconds depending on your LLM provider...")
    
    response = requests.post(
        f"{BASE_URL}/build-embeddings",
        json=payload
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Success!")
        print(f"Word: {data['word']}")
        print(f"Source: {data['source']}")
        print(f"\nEmbedding files created:")
        for file in data['embeddings_files']:
            print(f"  - {file}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


def test_timeline(concept: str):
    """Test timeline endpoint after embeddings are created."""
    print("\n" + "="*60)
    print(f"4. GET TIMELINE FOR '{concept.upper()}'")
    print("="*60)
    
    response = requests.get(
        f"{BASE_URL}/timeline",
        params={"concept": concept, "top_n": 3}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nConcept: {data['concept']}")
        print(f"\nTimeline:")
        
        for entry in data['timeline']:
            print(f"\n  {entry['era']}:")
            print(f"    Centroid shift: {entry['centroid_shift_from_prev']:.4f}")
            print(f"    Top matches:")
            for item in entry['top']:
                print(f"      - {item['text'][:80]}... (score: {item['score']:.3f})")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def main():
    """Run all demo tests."""
    # Test word and eras
    word = "science"
    eras = ["1900s", "1950s", "2020s"]
    
    print("\n" + "="*60)
    print("ECHOES API - LLM ETYMOLOGY DEMO")
    print("="*60)
    print(f"\nTesting with word: '{word}'")
    print(f"Eras: {', '.join(eras)}")
    
    # Run tests
    test_health()
    test_generate_evolution(word, eras)
    test_build_embeddings(word, eras)
    test_timeline(word)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Check the embeddings/ folder for generated files")
    print("2. Try the /timeline endpoint with different concepts")
    print("3. Experiment with different words and time periods")


if __name__ == "__main__":
    main()