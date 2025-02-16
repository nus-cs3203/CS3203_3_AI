from text_processor import TextProcessorComponent
from director import TextProcessorDirector
from builder import TextProcessorBuilder

# Usage Example
if __name__ == "__main__":
    builder = TextProcessorBuilder()
    director = TextProcessorDirector(builder)
    director.construct_basic_pipeline()
    
    text_processor = TextProcessorComponent(builder)
    sample_text = "Hello, this is a test! 😊"
    processed_text = text_processor.process(sample_text)
    print("Processed Text:", processed_text)
