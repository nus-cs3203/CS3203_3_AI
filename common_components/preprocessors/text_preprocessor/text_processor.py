class TextProcessorComponent:
    def __init__(self, builder):
        self.steps = builder.get_steps()

    def process(self, text):
        for step in self.steps:
            text = step.process(text)  # Each component has a .process() method
        return text
