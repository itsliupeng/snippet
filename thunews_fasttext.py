import fasttext

model = fasttext.train_supervised(input="thunews.train")
model.test("thunews.valid", k=1)

model = fasttext.train_supervised(input='thunews.train', autotuneValidationFile='thunews.valid', autotuneDuration=600)



def compute_precision_per_category(model_path, test_data_path):
    # Load the trained FastText model
    model = fasttext.load_model(model_path)

    # Load the test data
    with open(test_data_path, 'r') as f:
        lines = f.readlines()

    # Initialize counters
    category_counts = {}
    correct_counts = {}

    # Predict and compare
    for line in lines:
        true_label = line.split(' ')[0].replace('__label__', '')
        text = ' '.join(line.split(' ')[1:])
        text = text.replace("\n", "\\n")
        predicted_label = model.predict(text, k=1)[0][0].replace('__label__', '')

        # Update counters
        category_counts[true_label] = category_counts.get(true_label, 0) + 1
        if true_label == predicted_label:
            correct_counts[true_label] = correct_counts.get(true_label, 0) + 1

    # Compute precision for each category
    precisions = {}
    for category, count in category_counts.items():
        correct = correct_counts.get(category, 0)
        precisions[category] = correct / count

    return precisions


# Example usage
model_path = 'thunews_model.bin'
test_data_path = 'thunews.valid'
precisions = compute_precision_per_category(model_path, test_data_path)
print(precisions)



# with the previously trained `model` object, call :
model.quantize(input='thunews.train', retrain=True)

# then display results and save the new model :
print_results(*model.test("thunews.valid"))
model.save_model("model_filename.ftz")

# (167215, 0.9387375534491523, 0.9387375534491523)
# (167215, 0.9551236432138265, 0.9551236432138265)
