import gradio
import inspect
print('gradio', gradio.__version__)
try:
    sig = inspect.signature(gradio.Interface.__init__)
    print('Interface signature:', sig)
except Exception as e:
    print('Interface signature error:', e)
