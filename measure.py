from jiwer import wer

def calculate_metrics(reference, hypothesis):
    # Calculate Word Error Rate (WER)
    wer_score = wer(reference, hypothesis)

    # Calculate Character Error Rate (CER)
    cer_score = wer(reference, hypothesis, truth_transform=lambda x: x.split(), hypothesis_transform=lambda x: x.split())

    # Calculate Word Accuracy
    word_accuracy = 1 - wer_score

    return wer_score, cer_score, word_accuracy

def main():
    # Load reference transcript
    with open('reference_transcript.txt', 'r') as file:
        reference_transcript = file.read().replace('\n', '')

    # Transcribe audio
    hypothesis_transcript = ds.stt(audio)

    # Calculate metrics
    wer_score, cer_score, word_accuracy = calculate_metrics(reference_transcript, hypothesis_transcript)

    # Print metrics
    print(f'Word Error Rate (WER): {wer_score}')
    print(f'Character Error Rate (CER): {cer_score}')
    print(f'Word Accuracy: {word_accuracy}')

if __name__ == '__main__':
    main()