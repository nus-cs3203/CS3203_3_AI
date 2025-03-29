import pandas as pd
import os
from categorizer.data_preprocessing import preprocess_data 

from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.length_validator import LengthValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.validator_logger import ValidatorLogger
from categorizer.news_filter import filter_for_opinions
from categorizer.r1_categorizer import categorize_complaints
from categorizer.post_process_data import post_process_data
from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.emotion.distilroberta import DistilRobertaClassifier
from sentiment_analyser.emotion.roberta import RobertaClassifier
from sentiment_analyser.polarity.bert import BERTClassifier
from sentiment_analyser.polarity.vader import VaderSentimentClassifier

def process_pipeline(input_file, output_folder):
    """
    Complete pipeline for processing posts
    """
    # Create output directory and subdirectories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "preprocessing"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "news_filter_results"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "complaint_categorizer_results"), exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(input_file)
    
    # Step 0: Preprocessing
    print("\nStep 0: Data Preprocessing...")
    df_preprocessed = preprocess_data(df, os.path.join(output_folder, "preprocessing"))
    
    # Step 1: News and Opinion Filtering
    print("\nStep 1: Filtering news posts and identifying opinions...")
    df_filtered = filter_for_opinions(
        df=df_preprocessed,
        output_folder=os.path.join(output_folder, "news_filter_results")
    )
    df_filtered.to_csv(os.path.join(output_folder, "2_after_news_filter.csv"), index=False)
    
    # Step 2: First Round Categorization
    print("\nStep 2: First round complaint categorization...")
    df_categorized = categorize_complaints(
        df=df_filtered,
        output_csv=os.path.join(output_folder, "complaint_categorizer_results/first_round_categorized.csv"),
        is_second_round=False
    )
    
    # Filter complaints from first round
    df_first_round_complaints = df_categorized[
        df_categorized['Intent Category'].str.lower() == 'yes'
    ].copy()
    
    print(f"\nFound {len(df_first_round_complaints)} complaints in first round")
    df_first_round_complaints.to_csv(os.path.join(output_folder, "3_first_round_complaints.csv"), index=False)
    
    # Step 3: Second Round Verification 
    print("\nStep 3: Second round verification...")
    df_verified = categorize_complaints(
        df=df_first_round_complaints,
        output_csv=os.path.join(output_folder, "complaint_categorizer_results/second_round_verified.csv"),
        is_second_round=True 
    )
    
    # Final filtering
    df_final = df_verified[
        df_verified['Intent Category'].str.lower() == 'yes'
    ].copy()
    df_final.to_csv(os.path.join(output_folder, "4_final_verified_complaints.csv"), index=False)
    
    # Step 4: Sentiment Analysis
    print("\nStep 4: Running sentiment analysis...")
    classifiers = [
        #("BERT", BERTClassifier()),
        ("VADER", VaderSentimentClassifier()),
        #("DistilRoberta Emotion", DistilRobertaClassifier()),
        #("Roberta Emotion", RobertaClassifier()),
    ]
    
    for name, classifier in classifiers:
        print(f"\n===== Running {name} Sentiment Analysis =====")
        context = SentimentAnalysisContext(classifier)
        df_final = context.analyze(df_final, text_cols=["combined_text"])
        df_final.to_csv(os.path.join(output_folder, f"5_{name}_sentiment_analysis.csv"), index=False)
    
    # Step 5: Post-processing
    print("\nStep 5: Post-processing data...")
    df_final = post_process_data(df=df_final)
    
    # Save final results
    output_path = os.path.join(output_folder, "final_processed_data.csv")
    df_final.to_csv(output_path, index=False)
    
    # Generate summary statistics
    stats = {
        'Initial posts': len(df),
        'Posts after news filter': len(df_filtered),
        'First round complaints': len(df_first_round_complaints),
        'Final verified complaints': len(df_final),
        'Domain categories distribution': df_final['category'].value_counts().to_dict(),
        'Average confidence (final)': df_final['confidence'].mean(),
        'Average sentiment (final)': df_final['sentiment'].mean(),
        'Average importance (final)': df_final['importance'].mean()
    }
    
    # Save statistics
    with open(os.path.join(output_folder, "processing_summary.txt"), 'w') as f:
        f.write("Processing Statistics:\n\n")
        for key, value in stats.items():
            f.write(f"{key}:\n{value}\n\n")
    
    print("\n===== Pipeline Complete =====")
    print(f"Final results saved to: {output_path}")
    print(f"Number of final complaints: {len(df_final)}")
    return df_final

if __name__ == "__main__":
    input_file = "last_round_files/all_posts_2022_2025.csv"
    output_folder = "pipeline_results"
    
    df_final = process_pipeline(input_file, output_folder)
