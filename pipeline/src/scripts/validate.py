import argparse
import numpy as np
from validation.validator import Validator
from validation.statistics import Statistics

def main():
    parser = argparse.ArgumentParser(description="Validate extracted state vectors.")
    parser.add_argument("--input", default="state_vectors.npy",
                        help="Path to NumPy file containing extracted state vectors")
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration YAML")
    parser.add_argument("--report", default="validation_report.txt",
                        help="Path to write validation results")
    args = parser.parse_args()

    # Load state vectors
    states = np.load(args.input)
    validator = Validator(args.config)
    stats = Statistics()

    errors = []
    for idx, state in enumerate(states):
        try:
            validator.validate(state)
            stats.update(state)
        except (ValueError, TypeError) as e:
            errors.append(f"State {idx}: {str(e)}")

    # Write validation report
    with open(args.report, "w") as f:
        if errors:
            f.write("Validation failed\n")
            f.write(f"{len(errors)} invalid states out of {len(states)}\n\n")
            for line in errors:
                f.write(line + "\n")
        else:
            f.write("Validation passed\n")
            f.write(f"All {len(states)} states are valid\n\n")
        f.write("Summary statistics for valid states:\n")
        f.write(str(stats.summary()))

    print(f"Validation report saved to {args.report}")

if __name__ == "__main__":
    main()
