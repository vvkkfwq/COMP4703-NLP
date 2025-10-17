import json
from pathlib import Path
from config import RANKERS, OUTPUT_PATH, is_STAGING


def test_paths():
    """验证 RAGA.py 路径配置"""
    print("=" * 60)
    print(f"STAGING Mode: {is_STAGING}")
    print("=" * 60)

    print(f"\nOUTPUT_PATH: {OUTPUT_PATH}")
    print(f"  Exists: {OUTPUT_PATH.exists()}")

    if not OUTPUT_PATH.exists():
        print(f"  Creating: {OUTPUT_PATH}")
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    print("\nRANKERS Input Paths:")
    print("-" * 60)

    all_valid = True
    for ranker_name, ranker_path in RANKERS.items():
        path_obj = Path(ranker_path) if isinstance(ranker_path, str) else ranker_path
        exists = path_obj.exists()
        status = "✅" if exists else "❌"

        print(f"{status} {ranker_name:20s} -> {path_obj}")

        if exists:
            try:
                with open(path_obj, "r") as f:
                    data = json.load(f)
                print(f"   Items: {len(data)}")

                if len(data) > 0:
                    first = data[0]
                    required_keys = [
                        "query",
                        "answer",
                        "question_type",
                        "retrieval_list",
                    ]
                    missing = [k for k in required_keys if k not in first]

                    if missing:
                        print(f"   ⚠️  Missing keys: {missing}")
                        all_valid = False
                    else:
                        print(
                            f"   ✅ Schema valid, retrieval_list length: {len(first['retrieval_list'])}"
                        )
            except json.JSONDecodeError:
                print(f"   ❌ Invalid JSON")
                all_valid = False
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
                all_valid = False
        else:
            all_valid = False

    print("\n" + "=" * 60)
    if all_valid:
        print("✅ All paths valid - ready to run RAGA.py")
    else:
        print("❌ Some paths invalid - fix config.py before running")
    print("=" * 60)


if __name__ == "__main__":
    test_paths()
