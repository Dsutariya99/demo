# from django.shortcuts import render

# # Create your views here.


from django.shortcuts import render
from .lstm_utils import TextGenerator

def index(request):
    generated_text = None
    seed_input = ""

    if request.method == 'POST':
        seed_input = request.POST.get('seed', '')
        if seed_input:
            generator = TextGenerator()
            try:
                # Generate 200 characters
                generated_text = generator.generate_text(seed_input, length=200)
            except Exception as e:
                generated_text = f"Error generating text: {str(e)}"

    return render(request, 'exam/index.html', {
        'generated_text': generated_text,
        'seed_input': seed_input
    })