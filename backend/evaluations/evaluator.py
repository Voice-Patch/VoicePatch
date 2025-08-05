import pandas as pd
import numpy as np
import ast
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import evaluate
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("ignore")


class CompositeMLMEvaluator:
    def __init__(self, csv_file_path: str, device: str = None):
        self.df = pd.read_csv(csv_file_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {"samples_evaluated": len(self.df)}
        self._load_models()

    def _load_models(self):
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.word_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.word_model = AutoModel.from_pretrained("microsoft/deberta-v3-base").to(
            self.device
        )

        self.rouge = evaluate.load("rouge")
        self.meteor = evaluate.load("meteor")

        print("models loaded")

    def parse_masks_and_predictions(self, row):
        try:
            original_masks = ast.literal_eval(row["Replacement Masks"])
            llm_predictions = ast.literal_eval(row["LLM Replaced masks"])
            return original_masks, llm_predictions
        except:
            return [], []

    def exact_accuracy(self) -> float:
        total_masks = 0
        correct_predictions = 0

        for _, row in self.df.iterrows():
            original_masks, llm_predictions = self.parse_masks_and_predictions(row)

            for orig, pred in zip(original_masks, llm_predictions):
                total_masks += 1
                if orig.strip().lower() == pred.strip().lower():
                    correct_predictions += 1

        accuracy = correct_predictions / total_masks if total_masks > 0 else 0
        self.results["exact_accuracy"] = accuracy
        return accuracy

    def contextual_word_similarity(self) -> float:
        similarities = []
        successful_comparisons = 0
        total_comparisons = 0

        for idx, row in self.df.iterrows():
            original_masks, llm_predictions = self.parse_masks_and_predictions(row)

            if not original_masks or not llm_predictions:
                continue

            masked_context = row["Masked"]

            if pd.isna(masked_context):
                continue

            masked_context = str(masked_context).strip()
            if not masked_context:
                continue

            # Process each mask-prediction pair
            temp_context = masked_context
            for orig_word, pred_word in zip(original_masks, llm_predictions):
                total_comparisons += 1

                # Replace first occurrence of [MASK] with the words
                orig_context = temp_context.replace("[MASK]", orig_word, 1)
                pred_context = temp_context.replace("[MASK]", pred_word, 1)

                # Get contextualized embeddings
                orig_embedding = self._get_word_embedding(orig_context, orig_word)
                pred_embedding = self._get_word_embedding(pred_context, pred_word)

                if orig_embedding is not None and pred_embedding is not None:
                    # Check for zero vectors which can cause issues
                    orig_norm = np.linalg.norm(orig_embedding)
                    pred_norm = np.linalg.norm(pred_embedding)

                    if orig_norm == 0 or pred_norm == 0:
                        continue

                    try:
                        similarity = cosine_similarity(
                            orig_embedding.reshape(1, -1), pred_embedding.reshape(1, -1)
                        )[0][0]

                        if np.isnan(similarity) or np.isinf(similarity):
                            continue

                        similarities.append(similarity)
                        successful_comparisons += 1
                    except Exception as sim_error:
                        print(
                            f"Error calculating similarity for row {idx}, mask '{orig_word}': {sim_error}"
                        )
                        continue

                temp_context = temp_context.replace("[MASK]", orig_word, 1)
        average_similarity = np.mean(similarities) if similarities else 0.0
        self.results["contextual_word_similarity"] = average_similarity
        return average_similarity

    def _get_word_embedding(self, context: str, target_word: str) -> np.ndarray | None:
        try:
            if not context or not target_word:
                return None

            context = str(context).strip()
            target_word = str(target_word).strip()

            if not context or not target_word:
                return None

            inputs = self.word_tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.word_model(**inputs)
                embeddings = outputs.last_hidden_state[0]

            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                return None

            token_ids = inputs["input_ids"][0].cpu().tolist()
            tokens = self.word_tokenizer.convert_ids_to_tokens(token_ids)

            target_tokens = self.word_tokenizer.tokenize(target_word.lower())

            if not target_tokens:
                return None

            word_embedding = self._find_word_embedding(
                tokens, target_tokens, embeddings
            )

            if word_embedding is not None:
                embedding_np = word_embedding.cpu().numpy()

                if np.isnan(embedding_np).any() or np.isinf(embedding_np).any():
                    return None

                return embedding_np
            else:
                print(f"Could not find word '{target_word}' in context")
                return None

        except Exception as e:
            print(f"Error getting embedding for '{target_word}': {e}")
            return None

    def _find_word_embedding(
        self, tokens, target_tokens, embeddings: torch.Tensor
    ) -> torch.Tensor | None:
        target_tokens_lower = [t.lower() for t in target_tokens]
        tokens_lower = [t.lower() for t in tokens]

        for i in range(len(tokens_lower) - len(target_tokens_lower) + 1):
            if tokens_lower[i : i + len(target_tokens_lower)] == target_tokens_lower:
                word_embeddings = embeddings[i : i + len(target_tokens_lower)]
                return word_embeddings.mean(dim=0)

        for i, token in enumerate(tokens_lower):
            for target_token in target_tokens_lower:
                if target_token in token or token in target_token:
                    return embeddings[i]

        return None

    def sentence_transformer_similarity(self) -> float:
        similarities = []

        originals = self.df["Original"].tolist()
        llm_replaced = self.df["LLM Replaced text"].tolist()

        orig_embeddings = self.sentence_model.encode(originals)
        llm_embeddings = self.sentence_model.encode(llm_replaced)

        for orig_emb, llm_emb in zip(orig_embeddings, llm_embeddings):
            similarity = cosine_similarity(
                orig_emb.reshape(1, -1), llm_emb.reshape(1, -1)
            )[0][0]
            similarities.append(similarity)

        avg_similarity = np.mean(similarities)
        self.results["sentence_transformer_similarity"] = avg_similarity
        return avg_similarity

    # def bert_score_evaluation(self) -> dict:
    #     """Calculate BERTScore - SOTA metric for text generation quality."""
    #     originals = self.df["Original"].tolist()
    #     llm_replaced = self.df["LLM Replaced text"].tolist()

    #     # Calculate BERTScore
    #     P, R, F1 = bert_score(llm_replaced, originals, lang="en", verbose=False)

    #     bert_scores = {
    #         "bertscore_precision": P.mean().item(),
    #         "bertscore_recall": R.mean().item(),
    #         "bertscore_f1": F1.mean().item(),
    #     }

    #     self.results.update(bert_scores)
    #     return bert_scores

    # def rouge_scores(self) -> dict:
    #     """Calculate ROUGE scores for text overlap quality."""
    #     originals = self.df["Original"].tolist()
    #     llm_replaced = self.df["LLM Replaced text"].tolist()

    #     rouge_results = self.rouge.compute(
    #         predictions=llm_replaced, references=originals
    #     )

    #     rouge_scores = {
    #         "rouge1_f1": rouge_results["rouge1"],
    #         "rouge2_f1": rouge_results["rouge2"],
    #         "rougeL_f1": rouge_results["rougeL"],
    #         "rougeLsum_f1": rouge_results["rougeLsum"],
    #     }

    #     self.results.update(rouge_scores)
    #     return rouge_scores

    # def meteor_score(self) -> float:
    #     """Calculate METEOR score - considers synonyms and paraphrases."""
    #     originals = self.df["Original"].tolist()
    #     llm_replaced = self.df["LLM Replaced text"].tolist()

    #     meteor_result = self.meteor.compute(
    #         predictions=llm_replaced, references=originals
    #     )

    #     meteor_score = meteor_result["meteor"]
    #     self.results["meteor_score"] = meteor_score
    #     return meteor_score

    def run_composite_evaluation(self) -> dict:
        exact_acc = self.exact_accuracy()
        contextual_sim = self.contextual_word_similarity()
        sentence_sim = self.sentence_transformer_similarity()
        # bert_scores = self.bert_score_evaluation()
        # rouge_results = self.rouge_scores()
        # meteor = self.meteor_score()
        print("Number of samples evaluated:", self.results["samples_evaluated"])
        print(f"Exact Accuracy:{exact_acc:.4f}")
        print(f"Contextual Word Similarity:{contextual_sim:.4f}")
        print(f"Sentence Semantic Similarity:{sentence_sim:.4f}")
        # print(f"BERTScore F1:{bert_scores['bertscore_f1']:.4f}")
        # print(f"BERTScore Precision:{bert_scores['bertscore_precision']:.4f}")
        # print(f"BERTScore Recall:{bert_scores['bertscore_recall']:.4f}")
        # print(f"ROUGE-L F1:{rouge_results['rougeL_f1']:.4f}")
        # print(f"ROUGE-1 F1:{rouge_results['rouge1_f1']:.4f}")
        # print(f"ROUGE-2 F1:{rouge_results['rouge2_f1']:.4f}")
        # print(f"METEOR Score:{meteor:.4f}")

        return self.results


# Example usage
if __name__ == "__main__":
    evaluator = CompositeMLMEvaluator("./out.csv")
    results = evaluator.run_composite_evaluation()

    # Write results to a text file
    with open("results.txt", "w") as f:
        f.write(f"Number of samples evaluated: {results['samples_evaluated']}\n\n")
        f.write(f"- Exact Accuracy: {results['exact_accuracy']:.4f}\n")
        f.write(
            f"- Contextual Word Similarity: {results['contextual_word_similarity']:.4f}\n\n"
        )
        f.write(
            f"- Sentence Transformer Similarity: {results['sentence_transformer_similarity']:.4f}\n"
        )
        # f.write(f"- METEOR Score: {results['meteor_score']:.4f}\n\n")

        # f.write(f"BERTScore Metrics:\n")
        # f.write(f"- Precision: {results['bertscore_precision']:.4f}\n")
        # f.write(f"- Recall: {results['bertscore_recall']:.4f}\n")
        # f.write(f"- F1: {results['bertscore_f1']:.4f}\n\n")

        # f.write(f"ROUGE Metrics:\n")
        # f.write(f"- ROUGE-1 F1: {results['rouge1_f1']:.4f}\n")
        # f.write(f"- ROUGE-2 F1: {results['rouge2_f1']:.4f}\n")
        # f.write(f"- ROUGE-L F1: {results['rougeL_f1']:.4f}\n")
        # f.write(f"- ROUGE-Lsum F1: {results['rougeLsum_f1']:.4f}\n")
