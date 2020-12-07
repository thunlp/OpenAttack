import language_tool_python
class LanguageTool:
    def __init__(self):
        self.language_tool = language_tool_python.LanguageTool('en-US')
    def __call__(self, sent):
        return len(self.language_tool.check(sent))

