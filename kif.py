from kif_lib import Store
from kif_lib.vocabulary import wd
import time

def run_wikidata_demo():
    """Two Wikidata examples using only kif-lib features."""

    # --- Setup: Create store with proper User-Agent ---
    unique_ua = f"KIF-Demo-{time.time()}/1.0 (your-email@example.com)"
    kb = Store('wikidata', user_agent=unique_ua)

    # --- Example 1: Alan Turing's doctoral advisor ---
    print("-" * 30)
    print("Example 1: Alan Turing's Doctoral Advisor")
    print("-" * 30)

    try:
        results = list(kb.filter(subject=wd.Alan_Turing, property=wd.doctoral_advisor))
        for stmt in results:
            advisor = stmt.snak.value
            print(f"Advisor item: {advisor}")

            # Get English label via another filter (the tutorial way)
            label_results = list(kb.filter(subject=advisor, property=wd.label, language='en'))
            label = label_results[0].snak.value.content if label_results else "No label"
            print(f"Readable name: {label}")
    except Exception as e:
        print(f"Error: {e}")

run_wikidata_demo()