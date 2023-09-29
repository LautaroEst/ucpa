
class ContentFreeInputPrompt:

    def __init__(self, content_free_inputs):
        self.content_free_inputs = content_free_inputs
    
    def construct_prompt(self, query):
        return