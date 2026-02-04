"""Wrapper that registers flexible tasks, then calls oe_eval"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and register flexible tasks FIRST
import flexible_tasks  # noqa: F401
from deps.olmes.oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY as REGISTERED_TASKS

if __name__ == "__main__":
    # Patch run_eval to use the registry with flexible tasks
    import deps.olmes.oe_eval.run_eval as run_eval_module_actual
    run_eval_module_actual.TASK_REGISTRY = REGISTERED_TASKS
    
    from oe_eval.run_eval import run_eval
    from oe_eval import run_eval as run_eval_module
    
    args = run_eval_module._parser.parse_args()
    args_dict = vars(args)
    run_eval(args_dict)