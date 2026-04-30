# FractureX documentation

- **FractureX overall architecture and routes (中文)**: [HUZHANG_PHASEFIELD_ARCHITECTURE.md](HUZHANG_PHASEFIELD_ARCHITECTURE.md)
- **Hu–Zhang + phase-field architecture (English, focused version)**: [HUZHANG_PHASEFIELD_ARCHITECTURE.en.md](HUZHANG_PHASEFIELD_ARCHITECTURE.en.md)
- **HuZhang 相场接口测试手册（中文）**: [HUZHANG_INTERFACE_TEST_MANUAL.md](HUZHANG_INTERFACE_TEST_MANUAL.md)

After moving or adding modules that these documents reference, sync both language files and run from the repository root:

```bash
python scripts/verify_huzhang_docs.py
```

The script only checks that listed paths still exist; it does not rewrite the prose.
