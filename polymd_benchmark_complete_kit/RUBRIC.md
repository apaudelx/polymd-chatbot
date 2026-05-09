# PolyMD Benchmark Rubric

## Automatic checks

The included evaluator computes:

1. **Pass rate**  
   A combined lightweight pass metric. For answerable questions, the answer should include the expected value/source evidence. For abstention questions, it should refuse without inventing values.

2. **Chunk recall**  
   Whether at least one expected chunk ID appears in `retrieved_chunk_ids`.

3. **DOI recall**  
   Whether an expected DOI appears in `retrieved_dois` or in the answer text.

4. **Value mention rate**  
   Fraction of expected values explicitly present in the final answer.

5. **Must-mention rate**  
   Fraction of key strings, such as polymer, force field, value, DOI, or paper title, present in the final answer.

6. **Abstention accuracy**  
   Whether unsupported questions are refused and answerable questions are not refused.

## Manual checks recommended

Automatic string checks are intentionally transparent but not perfect. Manually inspect:

- **Comparison questions:** verify same-property comparison only. The model should not compare Tg to density or Young's modulus.
- **Multi-value questions:** verify that the answer does not collapse multiple records into a single fake average unless explicitly asked.
- **Condition-aware questions:** verify that conditions in `extra_info` are preserved when scientifically relevant.
- **Contextual follow-ups:** verify that the second answer uses the source/value from the first turn.
- **Abstention questions:** verify no fabricated DOI, force field, or numeric value appears.

## Suggested grade bands

- **Excellent:** pass rate ≥ 85%, chunk recall ≥ 85%, DOI recall ≥ 90%, abstention accuracy ≥ 90%.
- **Good:** pass rate ≥ 75%, chunk recall ≥ 75%, DOI recall ≥ 80%, abstention accuracy ≥ 85%.
- **Needs work:** pass rate < 70% or abstention accuracy < 80%.
