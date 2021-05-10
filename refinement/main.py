from utils import apply_refine, save_json
from constant import submit, save_path

def main():
    submit_refine = apply_refine(submit.copy())
    save_json(submit_refine, save_path)
    return

if __name__ == "__main__":
    main()