class HookCloser:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
    
    def __call__(self, module, input_, output_):
        self.model_wrapper.curr_embedding = output_
        output_.retain_grad()