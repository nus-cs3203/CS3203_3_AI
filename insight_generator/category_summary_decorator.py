import pandas as pd
from transformers import pipeline
from insight_generator.base_decorator import InsightDecorator

class CategoryWiseSummaryDecorator(InsightDecorator):
    def __init__(
        self, wrapped_insight_generator, 
        summarizer=None, 
        column_mappings=None
    ):
        super().__init__(wrapped_insight_generator)
        self.summarizer = summarizer or pipeline("summarization", model="facebook/bart-large-cnn")
        self.column_mappings = column_mappings if column_mappings else {
            "title": "title",
            "selftext": "selftext",
            "comments": "comments",
            "intent": "Intent Category",
            "domain": "Domain Category",
            "sentiment": "title_with_desc_label"
        }
        self.report_data = []

    def extract_insights(self, posts_df):
        insights_list = []

        for _, row in posts_df.iterrows():
            insights = super().extract_insights(row)

            title = row.get(self.column_mappings["title"], "")
            selftext = row.get(self.column_mappings["selftext"], "")
            comments = row.get(self.column_mappings["comments"], [])
            comments_text = " ".join(comments) if isinstance(comments, list) else str(comments)
            full_text = f"Title: {title} Selftext: {selftext} Comments: {comments_text}"
            
            summary = self.summarizer(full_text, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]
            
            category = row.get(self.column_mappings["intent"], "")
            domain = row.get(self.column_mappings["domain"], "")
            sentiment = row.get(self.column_mappings["sentiment"], "neutral")

            insights["category_summary"] = {
                "summary": summary,
                "category": category,
                "domain": domain,
                "sentiment": sentiment
            }
            
            self.report_data.append({
                "Category": category,
                "Domain": domain,
                "Sentiment": sentiment,
                "Summary": summary
            })
            
            insights_list.append(insights)
        
        return pd.DataFrame(insights_list)

    def generate_report(self, report_path="category_report.txt"):
        category_grouped = {}

        for entry in self.report_data:
            key = (entry["Category"], entry["Domain"])

            if key not in category_grouped:
                category_grouped[key] = []
            category_grouped[key].append(entry)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Category-wise Summary Report ===\n\n")
            for (category, domain), summaries in category_grouped.items():
                f.write(f"Category: {category} | Domain: {domain}\n")
                f.write(f"Total Entries: {len(summaries)}\n")
                sentiment_counts = {}

                for entry in summaries:
                    sentiment = entry["Sentiment"]
                    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

                f.write("Sentiment Distribution: " + str(sentiment_counts) + "\n")
                f.write("Sample Summaries:\n")
                for sample in summaries[:3]:
                    f.write(f"- {sample['Summary']}\n")
                f.write("\n" + "-"*50 + "\n\n")

        print(f"Category-wise summary report saved to {report_path}")
