"""Create the workflow flowchart and system architecture diagram used in the report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from config import OUT_DIR


def box(ax, x, y, w, h, text, color="#DCE9F7", edge="#1F4E79", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec=edge, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4))


def workflow():
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.2); ax.axis("off")
    steps = [("CSV dataset\\n4,341 municipalities", "#FFF2CC"), ("Load into MongoDB\\n(PyMongo insert_many)", "#DCE9F7"),
             ("Read from MongoDB\\ninto pandas", "#DCE9F7"), ("Preprocessing\\nclean, impute, encode,\\nlog-transform", "#E2F0D9")]
    steps2 = [("Train / test split\\n80% / 20%", "#E2F0D9"), ("Train models\\nRegression + Classification", "#F8CBAD"),
              ("Evaluate\\nR2, MAE, Accuracy,\\nPrecision, Recall, F1", "#F8CBAD"), ("Save to MongoDB\\n+ models + charts\\n-> Streamlit dashboard", "#DCE9F7")]
    for i, (t, c) in enumerate(steps):
        box(ax, 0.2 + i * 2.75, 2.6, 2.3, 1.3, t.replace("\\n", "\n"), c)
        if i < 3: arrow(ax, 0.2 + i * 2.75 + 2.3, 3.25, 0.2 + (i + 1) * 2.75, 3.25)
    for i, (t, c) in enumerate(steps2):
        box(ax, 0.2 + i * 2.75, 0.4, 2.3, 1.3, t.replace("\\n", "\n"), c)
        if i < 3: arrow(ax, 0.2 + i * 2.75 + 2.3, 1.05, 0.2 + (i + 1) * 2.75, 1.05)
    arrow(ax, 0.2 + 3 * 2.75 + 1.15, 2.6, 1.35, 1.7)
    ax.set_title("End-to-end project workflow", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_DIR / "12_workflow.png", dpi=150); plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(10, 5.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis("off")
    box(ax, 0.3, 3.6, 2.4, 1.1, "Data layer\nwaste_fee.csv", "#FFF2CC")
    box(ax, 3.8, 3.6, 2.6, 1.1, "MongoDB Atlas / local\nwaste_db", "#DCE9F7")
    box(ax, 7.2, 3.6, 2.5, 1.1, "PyMongo\n(Python driver)", "#DCE9F7")
    box(ax, 0.3, 1.9, 2.4, 1.1, "Preprocessing\n(pandas, scikit-learn\nPipelines)", "#E2F0D9")
    box(ax, 3.8, 1.9, 2.6, 1.1, "ML models\nRandom Forest, Decision Tree,\nLinear / Logistic Regression", "#F8CBAD")
    box(ax, 7.2, 1.9, 2.5, 1.1, "Evaluation\nR2, MAE, Accuracy,\nPrecision, Recall, F1", "#F8CBAD")
    box(ax, 0.3, 0.2, 4.4, 1.1, "Collections: municipalities | predictions | model_metrics", "#DCE9F7")
    box(ax, 5.3, 0.2, 4.4, 1.1, "Streamlit dashboard\n(explore, predict, metrics, MongoDB view)", "#E4D9F0")
    arrow(ax, 2.7, 4.15, 3.8, 4.15); arrow(ax, 7.2, 4.15, 6.4, 4.15)
    arrow(ax, 7.2 + 1.25, 3.6, 1.5 + 0.0, 3.0)
    arrow(ax, 2.7, 2.45, 3.8, 2.45); arrow(ax, 6.4, 2.45, 7.2, 2.45)
    arrow(ax, 8.45, 1.9, 4.2, 1.3); arrow(ax, 4.7, 0.75, 5.3, 0.75)
    ax.set_title("System architecture", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT_DIR / "13_architecture.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    workflow(); architecture(); print("[diagrams] saved")
