# exam/management/commands/train_lstm.py
from django.core.management.base import BaseCommand
from exam.lstm_utils import TextGenerator
import os

class Command(BaseCommand):
    help = 'Trains the LSTM model on a dataset'

    def handle(self, *args, **kwargs):
        # Path to your dataset
        data_path = os.path.join(os.getcwd(), 'data', 'shakespeare.txt')
        
        if not os.path.exists(data_path):
            self.stdout.write(self.style.ERROR(f'Dataset not found at {data_path}'))
            return

        self.stdout.write('Starting training... this may take a while.')
        generator = TextGenerator()
        # Use low epochs (e.g., 1 or 5) for "demo" purposes. Use 20+ for real results.
        generator.build_and_train(data_path, epochs=5) 
        
        self.stdout.write(self.style.SUCCESS('Training finished successfully!'))