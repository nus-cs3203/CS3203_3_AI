import pandas as pd
import numpy as np
from transformers import pipeline
from lime.lime_text import LimeTextExplainer
from insight_generator.base_decorator import InsightDecorator

class TopAdverseSentimentsDecoratorLIME(InsightDecorator):
    def __init__(self, wrapped, sentiment_col="sentiment", category_col="category", text_col="title_with_description",
                 top_k=5, log_file="top_adverse_sentiments.txt", output_csv_dir="files/",
                 use_fast_model=True):
        """
        Identifies top adverse sentiment posts per category and explains them using LIME.

        Args:
        - wrapped: Base insight generator.
        - sentiment_col: Column for sentiment scores.
        - category_col: Column for category labels.
        - text_col: Column for post text.
        - top_k: Number of top positive and negative samples per category.
        - log_file: File to log explanations.
        - output_csv_dir: Directory to store category-wise CSVs of explanations.
        - use_fast_model: Use default fast sentiment model instead of cardiffnlp.
        """
        super().__init__(wrapped)
        self.sentiment_col = sentiment_col
        self.category_col = category_col
        self.text_col = text_col
        self.top_k = top_k
        self.log_file = log_file
        self.output_csv_dir = output_csv_dir

        # Load a faster default model or the cardiffnlp one
        self.sentiment_model = pipeline("sentiment-analysis", model="common_components/singlish_classifier_2", truncation=True, max_length=512)

        # Setup LIME
        self.explainer = LimeTextExplainer(class_names=["negative", "positive"])

    def extract_insights(self, df):
        insights = self._wrapped.extract_insights(df)

        if self.category_col not in df.columns or self.sentiment_col not in df.columns:
            raise ValueError(f"Missing {self.category_col} or {self.sentiment_col} in dataframe.")

        top_sentiments_outputs = []
        df[self.text_col] = df[self.text_col].fillna("")

        with open(self.log_file, "w", encoding="utf-8") as log_file:
            for category, group in df.groupby(self.category_col):
                if len(group) < 10:
                    print(f"[INFO] Skipping category '{category}' as it has fewer than 10 posts.")
                    continue

                top_positive = group.nlargest(1, self.sentiment_col)
                top_negative = group.nsmallest(1, self.sentiment_col)

                if top_positive.empty and top_negative.empty:
                    continue

                combined = pd.concat([top_positive, top_negative])
                texts = combined[self.text_col].tolist()
                indices = combined.index.tolist()

                print(f"[INFO] Explaining {len(texts)} texts for category: {category}")
                explanations_df = self.explain_sentiments(texts, indices)
                explanations_df["category"] = category
                top_sentiments_outputs.append(explanations_df)
                print(top_sentiments_outputs)

                # Logging
                log_file.write(f"\nCategory: {category}\n")
                for _, row in top_positive.iterrows():
                    log_file.write(f"[+] Score: {row[self.sentiment_col]:.3f}, Text: {row[self.text_col][:100]}...\n")
                for _, row in top_negative.iterrows():
                    log_file.write(f"[-] Score: {row[self.sentiment_col]:.3f}, Text: {row[self.text_col][:100]}...\n")

                for _, row in explanations_df.iterrows():
                    log_file.write(f"[LIME] idx {row['index']}: {row['feature_1']}({row['weight_1']:.3f}), "
                                   f"{row['feature_2']}({row['weight_2']:.3f}), {row['feature_3']}({row['weight_3']:.3f})\n")

                # Save to CSV
                explanations_df.to_csv(f"{self.output_csv_dir}{category}_lime_explanations.csv", index=False)

        if top_sentiments_outputs:
            combined_df = pd.concat(top_sentiments_outputs).reset_index(drop=True)
            insights["explainer_words"] = combined_df.to_dict(orient="records")

            # Create a DataFrame with category, explainer chosen words, and their scores
            category_explainer_df = combined_df.groupby("category").apply(
            lambda group: pd.DataFrame({
                "category": group["category"],
                "explainer_chosen_words": group.apply(
                lambda row: [
                    (row["feature_1"], row["weight_1"]),
                    (row["feature_2"], row["weight_2"]),
                    (row["feature_3"], row["weight_3"])
                ], axis=1
                )
            })
            ).reset_index(drop=True)
        else:
            insights["explainer_words"] = []
            category_explainer_df = pd.DataFrame(columns=["category", "explainer_chosen_words"])

        return category_explainer_df
    
    def explain_sentiments(self, texts, indices, visualize=True, save_html=True):
        """
        Run LIME on each text to extract top 3 features contributing to sentiment.
        Optionally visualizes and/or saves HTML explanations.

        Args:
            texts (list): List of texts to explain.
            indices (list): Corresponding indices of texts in the original DataFrame.
            visualize (bool): If True, shows LIME explanation in a browser.
            save_html (bool): If True, saves HTML explanation files per instance.

        Returns:
            DataFrame: LIME explanations with top 3 features.
        """
        explanations = []

        def predict_proba(samples):
            results = self.sentiment_model(samples)
            probas = []
            for r in results:
                if r["label"].lower() == "positive":
                    probas.append([0, r["score"]])
                else:
                    probas.append([r["score"], 0])
            return np.array(probas)

        for i, text in enumerate(texts):
            idx = indices[i]
            try:
                print(f"[LIME] Explaining idx={idx}")
                exp = self.explainer.explain_instance(text, predict_proba, num_features=3, num_samples=500)

                if save_html:
                    html_path = f"{self.output_csv_dir}lime_explanation_{idx}.html"
                    exp.save_to_file(html_path)
                    print(f"[LIME] Saved HTML explanation to {html_path}")

                top_3 = exp.as_list()[:3]
                explanations.append({
                    "index": idx,
                    "text": text,
                    "feature_1": top_3[0][0],
                    "weight_1": top_3[0][1],
                    "feature_2": top_3[1][0],
                    "weight_2": top_3[1][1],
                    "feature_3": top_3[2][0],
                    "weight_3": top_3[2][1],
                })

            except Exception as e:
                print(f"[ERROR] LIME explanation failed at idx={idx}: {str(e)}")
                explanations.append({
                    "index": idx,
                    "text": text,
                    "feature_1": "Error",
                    "weight_1": str(e),
                    "feature_2": None,
                    "weight_2": None,
                    "feature_3": None,
                    "weight_3": None,
                })

        return pd.DataFrame(explanations)
