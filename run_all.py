"""Run the whole project in one process: load -> train/evaluate -> MongoDB demo."""
import load_data
import train_models
import mongo_operations
import make_diagrams
import extra_analysis

if __name__ == "__main__":
    load_data.load()
    train_models.main()
    mongo_operations.main()
    extra_analysis.main()
    make_diagrams.workflow(); make_diagrams.architecture()
    print("\nDone. See the outputs/ folder for charts, metrics_summary.csv and results.json")
