# FractureX documentation

- **FractureX overall architecture and routes (中文)**: [huzhang_phasefield_architecture.md](huzhang_phasefield_architecture.md)
- **Hu–Zhang + phase-field architecture (English, focused version)**: [huzhang_phasefield_architecture.en.md](huzhang_phasefield_architecture.en.md)
- **HuZhang 相场接口测试手册（中文）**: [huzhang_interface_test_manual.md](huzhang_interface_test_manual.md)
- **D12 块预条件论文计划（中文）**: [D12_PRECONDITIONER_PAPER_PLAN.md](D12_PRECONDITIONER_PAPER_PLAN.md)

After moving or adding modules that these documents reference, sync both language files and run from the repository root:

```bash
python scripts/verify_huzhang_docs.py
```

The script only checks that listed paths still exist; it does not rewrite the prose.
