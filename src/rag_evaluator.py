import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pathlib import Path
from omegaconf import DictConfig
import hydra

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def evaluate(cfg: DictConfig):
    c = cfg.rag_evaluator

    # Load files
    log_df = pd.read_csv(c.log_file)
    gt_df = pd.read_csv(c.ground_truth_file)
    df = pd.merge(log_df, gt_df, on="question")

    # Load sentence transformer
    model = SentenceTransformer(c.sentence_transformer_model, device=c.device)

    def compute_cosine_similarity(a, b):
        emb = model.encode([a, b], convert_to_tensor=True)
        return cosine_similarity([emb[0].cpu().numpy()], [emb[1].cpu().numpy()])[0][0]

    def compute_bleu(reference, hypothesis):
        smoothie = SmoothingFunction().method4
        return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)

    # Compute metrics
    df["cosine_similarity"] = df.apply(lambda row: compute_cosine_similarity(row["answer"], row["ground_truth"]), axis=1)
    df["bleu_score"] = df.apply(lambda row: compute_bleu(row["ground_truth"], row["answer"]), axis=1)

    # Save results
    output_path = Path(c.output_file)
    df.to_csv(output_path, index=False)
    print(f"Evaluation saved to: {output_path}")

if __name__ == "__main__":
    evaluate()
