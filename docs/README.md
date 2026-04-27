# FractureX documentation

- **FractureX overall architecture and routes (中文)**: [HUZHANG_PHASEFIELD_ARCHITECTURE.md](HUZHANG_PHASEFIELD_ARCHITECTURE.md)
- **Hu–Zhang + phase-field architecture (English, focused version)**: [HUZHANG_PHASEFIELD_ARCHITECTURE.en.md](HUZHANG_PHASEFIELD_ARCHITECTURE.en.md)

After moving or adding modules that these documents reference, sync both language files and run from the repository root:

```bash
python scripts/verify_huzhang_docs.py
```

The script only checks that listed paths still exist; it does not rewrite the prose.
