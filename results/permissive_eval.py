"""Wrapper that registers flexible tasks, then calls oe_eval"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and register flexible tasks FIRST
import flexible_tasks  # noqa: F401

if __name__ == "__main__":
    # Patch at the source module, not run_eval
    from deps.olmes.oe_eval.tasks import oe_eval_tasks
    from deps.olmes.oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY
    
    # Verify the registry has our tasks
    print(f"boolq_flexible in registry: {'boolq_flexible' in TASK_REGISTRY}")
    
    # Now oe_eval_tasks.TASK_REGISTRY points to the right dict
    from oe_eval.run_eval import run_eval
    from oe_eval import run_eval as run_eval_module
    
    args = run_eval_module._parser.parse_args()
    args_dict = vars(args)
    run_eval(args_dict)