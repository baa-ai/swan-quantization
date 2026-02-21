#!/usr/bin/env python3
"""
Long-context benchmark for SmartQuant models.

Tests model performance with 8K and 32K token contexts using synthetic
needle-in-haystack style prompts. Measures:
  - Whether the model can retrieve embedded facts at different positions
  - Generation speed under long-context KV cache pressure
  - Memory usage with large KV caches

Usage:
    python long_context_bench.py --model ~/smartquant/models/maverick-smartquant
    python long_context_bench.py --model ~/smartquant/models/maverick-smartquant --output results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Filler paragraphs about various topics (each ~200-300 tokens when tokenized)
FILLER_PARAGRAPHS = [
    """The history of cartography stretches back thousands of years, with the earliest known maps dating to ancient Babylon around 600 BCE. These early maps were carved on clay tablets and depicted local areas such as estates and cities. The Greeks made significant contributions to mapmaking, with Ptolemy's Geography providing a coordinate system that would influence cartographers for centuries. During the medieval period, European maps were often oriented with east at the top and featured religious symbolism, while Arab cartographers like al-Idrisi produced remarkably accurate world maps. The Age of Exploration in the 15th and 16th centuries drove rapid advances in cartographic techniques, as navigators needed accurate charts for ocean voyages. Mercator's projection, introduced in 1569, became the standard for nautical navigation because it represented lines of constant bearing as straight lines. Modern cartography has been transformed by satellite imagery, GPS technology, and geographic information systems that can layer multiple types of data onto a single map.""",

    """Volcanic activity has shaped our planet's surface in profound ways throughout geological history. The Ring of Fire, encircling the Pacific Ocean, contains approximately 75 percent of the world's active and dormant volcanoes. When tectonic plates converge, one plate slides beneath another in a process called subduction, creating conditions for magma to rise and form volcanic arcs. Shield volcanoes like Mauna Loa in Hawaii are built by successive flows of relatively fluid basaltic lava, resulting in gently sloping profiles that can span enormous areas. In contrast, stratovolcanoes such as Mount Fuji are composed of alternating layers of lava and pyroclastic material, giving them steep, symmetrical cones. Supervolcanic eruptions, though extremely rare, can eject hundreds of cubic kilometers of material and alter global climate patterns for years. The Yellowstone caldera, which last erupted catastrophically about 640,000 years ago, sits atop a massive magma chamber that continues to be monitored by geologists.""",

    """The development of antibiotics represents one of the most transformative achievements in medical history. Alexander Fleming's accidental discovery of penicillin in 1928 opened the door to treating bacterial infections that had previously been death sentences. The mass production of penicillin during World War II saved countless lives among wounded soldiers and established the pharmaceutical industry's role in antibiotic development. Throughout the mid-20th century, scientists discovered numerous classes of antibiotics including tetracyclines, aminoglycosides, macrolides, and fluoroquinolones, each targeting different aspects of bacterial physiology. However, the widespread and often indiscriminate use of antibiotics in both human medicine and agriculture has accelerated the evolution of resistant bacteria. Methicillin-resistant Staphylococcus aureus, commonly known as MRSA, and extensively drug-resistant tuberculosis are just two examples of superbugs that pose serious public health threats. Researchers are now exploring novel approaches including bacteriophage therapy, antimicrobial peptides, and CRISPR-based techniques to combat antibiotic resistance.""",

    """The physics of superconductivity has fascinated scientists since Heike Kamerlingh Onnes first observed the phenomenon in mercury cooled to 4.2 Kelvin in 1911. Below a critical temperature, certain materials exhibit zero electrical resistance and expel magnetic fields through the Meissner effect, allowing electric current to flow indefinitely without energy loss. The BCS theory, developed by Bardeen, Cooper, and Schrieffer in 1957, explained conventional superconductivity through the formation of Cooper pairs, where electrons with opposite spins and momenta bind together via interactions with the crystal lattice. The discovery of high-temperature superconductors in 1986, particularly copper oxide ceramics that superconduct above the boiling point of liquid nitrogen, sparked intense research activity. Applications of superconductivity include MRI machines in hospitals, particle accelerators like the Large Hadron Collider, and SQUID magnetometers capable of detecting incredibly faint magnetic fields. The quest for room-temperature superconductivity continues, with recent claims involving hydrogen-rich materials under extreme pressures generating both excitement and controversy.""",

    """Ocean currents function as a global conveyor belt, transporting heat, nutrients, and marine organisms across vast distances. The thermohaline circulation, driven by differences in water temperature and salinity, connects the world's ocean basins in a continuous loop that takes approximately 1,000 years to complete. Warm surface waters from the tropics flow poleward, gradually cooling and becoming denser until they sink in the North Atlantic near Greenland and Iceland. This deep water then flows southward along the ocean floor, eventually upwelling in the Southern Ocean and Indian Ocean before being warmed again and completing the cycle. The Gulf Stream, a powerful western boundary current, carries warm water from the Gulf of Mexico northeastward toward Europe, significantly moderating the climate of the British Isles and Scandinavia. Climate change threatens to disrupt these circulation patterns by introducing large volumes of freshwater from melting ice sheets, potentially weakening the thermohaline circulation and triggering cascading effects on weather patterns worldwide.""",

    """The evolution of written language marks a pivotal transition in human civilization. The earliest writing systems emerged independently in Mesopotamia, Egypt, China, and Mesoamerica, each developing from pictographic or ideographic origins into more abstract symbolic systems. Sumerian cuneiform, which began as simple pictographs pressed into clay tablets around 3400 BCE, evolved over centuries into a complex system of wedge-shaped marks capable of representing the full range of spoken language. The Phoenician alphabet, developed around 1050 BCE, introduced the revolutionary concept of using individual symbols to represent consonant sounds rather than whole words or syllables. The Greeks adapted this alphabet by adding vowel symbols, creating a fully phonemic writing system that became the ancestor of Latin, Cyrillic, and many other scripts. The invention of printing, first with woodblock printing in China around 200 CE and later with Gutenberg's movable type press in 1440, dramatically accelerated the spread of literacy and knowledge. Today, Unicode encompasses over 150,000 characters from writing systems spanning human history.""",

    """Quantum computing represents a paradigm shift in computational capability. Unlike classical bits, which exist in definite states of 0 or 1, quantum bits or qubits can exist in superpositions of both states simultaneously, enabling quantum computers to explore vast solution spaces in parallel. Entanglement, another quantum mechanical property, allows qubits to be correlated in ways that have no classical analogue, enabling certain algorithms to achieve exponential speedups over their classical counterparts. Shor's algorithm for integer factorization and Grover's algorithm for database search are among the most well-known quantum algorithms. Current quantum processors from companies like IBM, Google, and Quantinuum operate with tens to hundreds of noisy qubits, a regime known as Noisy Intermediate-Scale Quantum computing. Error correction, which requires encoding a single logical qubit across many physical qubits, remains one of the greatest challenges facing the field. Despite these obstacles, quantum computing shows promise for applications in drug discovery, materials science, optimization, and cryptography.""",

    """The Amazon rainforest, spanning approximately 5.5 million square kilometers across nine South American countries, is the largest tropical rainforest on Earth and harbors an estimated ten percent of all species on the planet. Its canopy, reaching heights of 40 meters or more, creates a complex vertical ecosystem where different layers support distinct communities of plants, animals, and microorganisms. The forest floor receives only about two percent of sunlight, yet supports a rich decomposer community that rapidly recycles nutrients from fallen leaves and wood. Indigenous peoples have inhabited the Amazon for at least 11,000 years, developing sophisticated knowledge of medicinal plants and sustainable resource management practices. The forest plays a critical role in global climate regulation, absorbing roughly two billion tons of carbon dioxide annually and generating about half of its own rainfall through transpiration. Deforestation, primarily driven by cattle ranching, soy cultivation, and logging, threatens this vital ecosystem, with satellite data showing that approximately 17 percent of the Amazon's forest cover has been lost in the past 50 years.""",

    """The architecture of Gothic cathedrals represents a remarkable fusion of engineering innovation and artistic expression. Beginning in 12th-century France, Gothic builders developed structural techniques that allowed them to construct buildings of unprecedented height and luminosity. The pointed arch, which distributes weight more efficiently than the rounded Roman arch, enabled wider spans and taller openings. Flying buttresses transferred the lateral thrust of the vaulted ceiling to external supports, freeing the walls from their load-bearing function and allowing them to be replaced with expansive stained glass windows. The ribbed vault system divided the ceiling into discrete structural units, making construction more flexible and enabling the creation of complex geometric patterns overhead. Chartres Cathedral, completed in the early 13th century, exemplifies the mature Gothic style with its 37-meter-high nave, elaborate sculpture programs, and 176 stained glass windows that transform the interior into a kaleidoscope of colored light. The construction of these cathedrals often spanned generations, serving as focal points of community identity and devotion.""",

    """Photosynthesis is the biochemical process that forms the foundation of nearly all life on Earth. In the light-dependent reactions, chlorophyll molecules in the thylakoid membranes of chloroplasts absorb photons of light energy, exciting electrons to higher energy states. These energized electrons pass through an electron transport chain, driving the synthesis of ATP and NADPH while splitting water molecules to release oxygen as a byproduct. In the Calvin cycle, which occurs in the stroma, the enzyme RuBisCO catalyzes the fixation of atmospheric carbon dioxide into three-carbon organic molecules, which are subsequently reduced using the ATP and NADPH generated in the light reactions. C4 and CAM plants have evolved alternative carbon fixation pathways that minimize photorespiration in hot, arid environments. Globally, photosynthetic organisms fix approximately 258 billion tons of carbon dioxide per year, producing the oxygen that makes up 21 percent of Earth's atmosphere. Research into artificial photosynthesis aims to mimic these natural processes to produce clean fuels directly from sunlight and water.""",

    """The study of prime numbers has captivated mathematicians for over two millennia. Euclid proved that there are infinitely many primes around 300 BCE, and the Sieve of Eratosthenes provided an efficient method for finding them. The distribution of primes among the integers follows patterns that are both regular and mysterious. The Prime Number Theorem, proved independently by Hadamard and de la Vallée Poussin in 1896, describes the asymptotic distribution of primes, showing that the number of primes less than n is approximately n divided by the natural logarithm of n. The Riemann Hypothesis, proposed in 1859 and still unproven, posits that all nontrivial zeros of the Riemann zeta function lie on the critical line with real part one-half, and its truth would provide the most precise known description of prime distribution. Modern cryptographic systems including RSA rely on the computational difficulty of factoring large numbers into their prime components. The Great Internet Mersenne Prime Search, a distributed computing project, has found several record-setting primes, with the largest known prime as of recent years having tens of millions of digits.""",

    """The human immune system is a remarkably sophisticated defense network comprising both innate and adaptive components. The innate immune system provides immediate, non-specific responses through physical barriers like skin and mucous membranes, antimicrobial proteins, and phagocytic cells including neutrophils and macrophages. When pathogens breach these initial defenses, the adaptive immune system mounts a targeted response through T cells and B cells that recognize specific molecular signatures called antigens. Helper T cells coordinate the immune response by releasing cytokines that activate other immune cells, while cytotoxic T cells directly destroy infected cells. B cells produce antibodies that bind to specific antigens, neutralizing pathogens or marking them for destruction by other immune cells. Immunological memory, mediated by long-lived memory T and B cells, allows the immune system to mount faster and stronger responses upon subsequent encounters with the same pathogen, forming the basis of vaccination. Autoimmune diseases arise when the immune system mistakenly attacks the body's own tissues, while immunodeficiency conditions leave individuals vulnerable to infections that healthy immune systems easily control.""",
]

# Needle facts to embed at different positions
NEEDLE_FACTS = [
    {
        "needle": "The secret password for Project Aurora is 'crystalline-harbinger-7492'. This password was established on March 15th and must be used for all secure communications.",
        "question": "What is the secret password for Project Aurora?",
        "expected_substring": "crystalline-harbinger-7492",
    },
    {
        "needle": "Dr. Helena Vasquez discovered that the optimal temperature for synthesizing compound XR-7 is exactly 347.2 degrees Celsius. This finding was published in the Journal of Advanced Materials on October 3rd.",
        "question": "What is the optimal temperature for synthesizing compound XR-7, and who discovered it?",
        "expected_substring": "347.2",
    },
    {
        "needle": "The annual budget allocated to the Meridian Research Initiative is precisely $42,876,103. This figure was approved by the board of directors during the quarterly review meeting held in Stockholm.",
        "question": "What is the exact annual budget for the Meridian Research Initiative?",
        "expected_substring": "42,876,103",
    },
    {
        "needle": "The winning coordinates for the geocaching competition are latitude 38.8977 and longitude -77.0365. These coordinates point to a location near the reflecting pool where the final prize is hidden beneath a bronze plaque.",
        "question": "What are the winning coordinates for the geocaching competition?",
        "expected_substring": "38.8977",
    },
]


def build_context(target_tokens: int, needle: str, position: str, tokenizer) -> str:
    """Build a context of approximately target_tokens with a needle embedded.

    Args:
        target_tokens: Target context length in tokens
        needle: The fact to embed
        position: Where to place the needle: 'beginning', 'middle', 'end'
        tokenizer: Tokenizer for token counting
    """
    # Estimate tokens per paragraph
    para_tokens = []
    for p in FILLER_PARAGRAPHS:
        t = len(tokenizer.encode(p))
        para_tokens.append(t)
    avg_tokens_per_para = sum(para_tokens) / len(para_tokens)

    # Calculate how many paragraphs we need
    needle_tokens = len(tokenizer.encode(needle))
    needed_filler_tokens = target_tokens - needle_tokens - 100  # leave room for instruction
    num_paragraphs = max(1, int(needed_filler_tokens / avg_tokens_per_para))

    # Build filler by cycling through paragraphs
    paragraphs = []
    for i in range(num_paragraphs):
        paragraphs.append(FILLER_PARAGRAPHS[i % len(FILLER_PARAGRAPHS)])

    # Insert needle at position
    if position == "beginning":
        insert_idx = min(2, len(paragraphs))
    elif position == "middle":
        insert_idx = len(paragraphs) // 2
    else:  # end
        insert_idx = max(0, len(paragraphs) - 2)

    paragraphs.insert(insert_idx, needle)

    context = "\n\n".join(paragraphs)

    # Trim if too long
    actual_tokens = len(tokenizer.encode(context))
    while actual_tokens > target_tokens * 1.1 and len(paragraphs) > 3:
        # Remove a filler paragraph (not the needle)
        for i in range(len(paragraphs) - 1, -1, -1):
            if paragraphs[i] != needle:
                paragraphs.pop(i)
                break
        context = "\n\n".join(paragraphs)
        actual_tokens = len(tokenizer.encode(context))

    return context, actual_tokens


def get_memory_info():
    """Get MLX Metal memory usage in GB."""
    import mlx.core as mx
    try:
        return {
            "allocated_gb": mx.get_active_memory() / 1e9,
            "peak_gb": mx.get_peak_memory() / 1e9,
        }
    except Exception:
        return {"allocated_gb": 0.0, "peak_gb": 0.0}


def run_long_context_bench(
    model_path: str,
    context_lengths: List[int],
    max_response_tokens: int = 256,
    output_path: str = None,
) -> Dict[str, Any]:
    """Run the long-context benchmark suite."""
    from mlx_lm import load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    print(f"Loading model from {model_path} ...")
    start = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - start
    mem_after_load = get_memory_info()
    print(f"Model loaded in {load_time:.1f}s, memory: {mem_after_load['allocated_gb']:.1f} GB")

    sampler = make_sampler(temp=0.0)  # Greedy for reproducibility

    results = {
        "model_path": model_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "load_time_seconds": load_time,
        "memory_after_load_gb": mem_after_load["allocated_gb"],
        "tests": [],
        "summary": {},
    }

    positions = ["beginning", "middle", "end"]
    total_correct = 0
    total_tests = 0
    all_speeds = []

    for ctx_len in context_lengths:
        print(f"\n{'='*60}")
        print(f"CONTEXT LENGTH: ~{ctx_len // 1000}K tokens")
        print(f"{'='*60}")

        for needle_info in NEEDLE_FACTS:
            for pos in positions:
                total_tests += 1
                needle = needle_info["needle"]
                question = needle_info["question"]
                expected = needle_info["expected_substring"]

                # Build context
                context, actual_tokens = build_context(
                    ctx_len, needle, pos, tokenizer
                )

                prompt = (
                    f"Read the following document carefully, then answer the question at the end.\n\n"
                    f"--- DOCUMENT START ---\n{context}\n--- DOCUMENT END ---\n\n"
                    f"Question: {question}\nAnswer:"
                )

                prompt_tokens = len(tokenizer.encode(prompt))
                print(f"\n  Needle: '{expected}' at {pos} ({prompt_tokens} prompt tokens)")

                # Generate
                start = time.time()
                try:
                    response = mlx_generate(
                        model,
                        tokenizer,
                        prompt=prompt,
                        max_tokens=max_response_tokens,
                        sampler=sampler,
                    )
                    elapsed = time.time() - start
                    response_tokens = len(tokenizer.encode(response))
                    tps = response_tokens / elapsed if elapsed > 0 else 0

                    # Check if the needle was found
                    found = expected.lower() in response.lower()
                    if found:
                        total_correct += 1

                    mem = get_memory_info()
                    all_speeds.append(tps)

                    test_result = {
                        "context_length": ctx_len,
                        "actual_prompt_tokens": prompt_tokens,
                        "needle_position": pos,
                        "needle_keyword": expected,
                        "question": question,
                        "response": response[:500],
                        "response_tokens": response_tokens,
                        "elapsed_seconds": elapsed,
                        "tokens_per_second": tps,
                        "found_needle": found,
                        "memory_allocated_gb": mem["allocated_gb"],
                        "memory_peak_gb": mem["peak_gb"],
                    }
                    results["tests"].append(test_result)

                    status = "FOUND" if found else "MISSED"
                    print(f"    [{status}] {tps:.1f} tok/s, {elapsed:.1f}s, mem: {mem['peak_gb']:.1f} GB peak")
                    print(f"    Response: {response[:150]}...")

                except Exception as e:
                    print(f"    [ERROR] {e}")
                    results["tests"].append({
                        "context_length": ctx_len,
                        "actual_prompt_tokens": prompt_tokens,
                        "needle_position": pos,
                        "needle_keyword": expected,
                        "error": str(e),
                        "found_needle": False,
                    })

    # Summary
    accuracy = total_correct / total_tests if total_tests > 0 else 0
    results["summary"] = {
        "total_tests": total_tests,
        "total_correct": total_correct,
        "accuracy": accuracy,
        "avg_tokens_per_second": sum(all_speeds) / len(all_speeds) if all_speeds else 0,
        "min_tokens_per_second": min(all_speeds) if all_speeds else 0,
        "max_tokens_per_second": max(all_speeds) if all_speeds else 0,
        "memory_peak_gb": get_memory_info()["peak_gb"],
    }

    # Per context-length summary
    for ctx_len in context_lengths:
        ctx_tests = [t for t in results["tests"] if t.get("context_length") == ctx_len]
        ctx_correct = sum(1 for t in ctx_tests if t.get("found_needle", False))
        ctx_speeds = [t["tokens_per_second"] for t in ctx_tests if "tokens_per_second" in t]
        results["summary"][f"{ctx_len // 1000}k"] = {
            "tests": len(ctx_tests),
            "correct": ctx_correct,
            "accuracy": ctx_correct / len(ctx_tests) if ctx_tests else 0,
            "avg_tps": sum(ctx_speeds) / len(ctx_speeds) if ctx_speeds else 0,
        }

    # Per position summary
    for pos in positions:
        pos_tests = [t for t in results["tests"] if t.get("needle_position") == pos]
        pos_correct = sum(1 for t in pos_tests if t.get("found_needle", False))
        results["summary"][f"position_{pos}"] = {
            "tests": len(pos_tests),
            "correct": pos_correct,
            "accuracy": pos_correct / len(pos_tests) if pos_tests else 0,
        }

    # Print summary
    print(f"\n{'='*60}")
    print(f"LONG-CONTEXT BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Overall accuracy: {total_correct}/{total_tests} ({accuracy:.1%})")
    for ctx_len in context_lengths:
        k = f"{ctx_len // 1000}k"
        s = results["summary"][k]
        print(f"  {k}: {s['correct']}/{s['tests']} ({s['accuracy']:.1%}), avg {s['avg_tps']:.1f} tok/s")
    for pos in positions:
        s = results["summary"][f"position_{pos}"]
        print(f"  {pos}: {s['correct']}/{s['tests']} ({s['accuracy']:.1%})")
    print(f"Speed: avg {results['summary']['avg_tokens_per_second']:.1f} tok/s")
    print(f"Peak memory: {results['summary']['memory_peak_gb']:.1f} GB")

    # Save
    if output_path is None:
        output_path = str(Path.home() / "smartquant" / "results" / "long_context_results.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SmartQuant Long-Context Benchmark")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the quantized MLX model",
    )
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[8000, 32000],
        help="Context lengths to test (default: 8000 32000)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum response tokens (default: 256)",
    )
    parser.add_argument(
        "--output",
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    run_long_context_bench(
        model_path=args.model,
        context_lengths=args.context_lengths,
        max_response_tokens=args.max_tokens,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
