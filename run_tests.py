"""Run test_cos.py tests directly without pytest."""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── import all test classes ──────────────────────────────────────
from tests.test_cos import (
    TestEmailGrader, TestCalendarGrader, TestDelegationGrader,
    TestCalendarModule, TestDelegationModule,
    TestChiefOfStaffEnv, TestTaskFiles, TestModels,
)

passed = 0
failed = 0
errors = []


def run_class(cls):
    global passed, failed
    instance = cls()
    methods = [m for m in dir(cls) if m.startswith("test_")]
    for name in methods:
        # parametrize manually for TestTaskFiles
        method = getattr(instance, name)
        try:
            if hasattr(method, "pytestmark") or "parametrize" in str(getattr(method, "__wrapped__", "")):
                method()
            else:
                method()
            print(f"  PASS  {cls.__name__}::{name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {cls.__name__}::{name}  — {e}")
            errors.append((cls.__name__, name, traceback.format_exc()))
            failed += 1


def run_parametrized(cls, param_name, params):
    """Run parametrized test methods manually."""
    global passed, failed
    methods = [m for m in dir(cls) if m.startswith("test_")]
    for name in methods:
        for param in params:
            instance = cls()
            method = getattr(instance, name)
            try:
                method(param)
                print(f"  PASS  {cls.__name__}::{name}[{param}]")
                passed += 1
            except TypeError:
                # method doesn't take param — skip
                break
            except Exception as e:
                print(f"  FAIL  {cls.__name__}::{name}[{param}]  — {e}")
                errors.append((cls.__name__, name, traceback.format_exc()))
                failed += 1


TASK_IDS = ["easy_cos", "medium_cos", "hard_cos"]
TASK_COUNTS = [
    ("easy_cos",   5,  2, 2),
    ("medium_cos", 10, 3, 3),
    ("hard_cos",   15, 5, 5),
]

print("\n" + "="*60)
print("Running AI Chief of Staff Test Suite")
print("="*60)

print("\n── Email Grader ──")
run_class(TestEmailGrader)

print("\n── Calendar Grader ──")
run_class(TestCalendarGrader)

print("\n── Delegation Grader ──")
run_class(TestDelegationGrader)

print("\n── Calendar Module ──")
run_class(TestCalendarModule)

print("\n── Delegation Module ──")
run_class(TestDelegationModule)

print("\n── Task Files ──")
tf = TestTaskFiles()
for task_id, n_emails, n_conflicts, n_tasks in TASK_COUNTS:
    for method_name in [m for m in dir(TestTaskFiles) if m.startswith("test_")]:
        method = getattr(tf, method_name)
        try:
            if method_name == "test_task_file_counts":
                method(task_id, n_emails, n_conflicts, n_tasks)
            else:
                method(task_id)
            print(f"  PASS  TestTaskFiles::{method_name}[{task_id}]")
            passed += 1
        except TypeError:
            pass
        except Exception as e:
            print(f"  FAIL  TestTaskFiles::{method_name}[{task_id}]  — {e}")
            errors.append(("TestTaskFiles", method_name, traceback.format_exc()))
            failed += 1

print("\n── Models ──")
run_class(TestModels)

print("\n── ChiefOfStaffEnv ──")
run_class(TestChiefOfStaffEnv)

print("\n" + "="*60)
print(f"Results: {passed} passed, {failed} failed")
print("="*60)

if errors:
    print("\nFailed tests:")
    for cls_name, test_name, tb in errors:
        print(f"\n  {cls_name}::{test_name}")
        print("  " + tb.splitlines()[-1])
    sys.exit(1)
else:
    print("\nALL TESTS PASSED")
