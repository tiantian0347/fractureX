#!/usr/bin/env bash
# Map CLI aliases (model1, square_tension, ...) to paper_huzhang case ids.

# Slug for logs / pid files (model0, model1, model2).
_paper_log_slug() {
  case "${1:-}" in
    model0) echo model0 ;;
    model0_aux|model0-aux|model0aux) echo model0_aux ;;
    model1|square|square_tension) echo model1 ;;
    model2) echo model2 ;;
    *)
      echo "unknown"
      return 1
      ;;
  esac
}

# Case id passed to run_case.py (model0, square, model2).
_normalize_paper_case() {
  case "${1:-}" in
    model0) echo model0 ;;
    model1|square|square_tension) echo square ;;
    model2) echo model2 ;;
    all) echo all ;;
    *)
      echo "unknown"
      return 1
      ;;
  esac
}
