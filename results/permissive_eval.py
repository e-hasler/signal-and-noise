"""Wrapper that registers flexible tasks, then calls oe_eval"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import flexible tasks to register them
import flexible_tasks  # noqa: F401

if __name__ == "__main__":
    from oe_eval.run_eval import run_eval
    from oe_eval import run_eval as run_eval_module
    
    args = run_eval_module._parser.parse_args()
    args_dict = vars(args)
    run_eval(args_dict)