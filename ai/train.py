'''
Disturbance Tracker - Model Trainer
'''
import logging
import pathlib
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.model_selection

# DTrack
import ai.model
import ai.options


def main():
    '''
    Entry point for `python3 -m ai.train`:
    Create a trained model for each configured set of tagged audio
    '''
    options = ai.options.bootstrap()
    workspace = pathlib.Path(options['workspace'])
    tags = workspace / 'tags'

    for model_name in options['inspect_models']:

        # Train model against dataset
        logging.info('Training %s ...', model_name)
        model = train(
                model_name,
                match_dir=tags / model_name,
                empty_dir=tags / 'empty')

        # Test trained model against full dataset
        logging.info('TRAINING COMPLETE! Validating %s...', model_name)
        validate(
                model,
                match_dir=tags / model_name,
                empty_dir=tags / 'empty')

        # Save trained model
        ai.model.save(model, workspace / f'{model_name}.pkl')


def train(model_name, match_dir, empty_dir):
    '''
    Returns a NEW trained model from tagged audio segments
    '''
    (x, y) = ai.model.prepare_dataset(match_dir, empty_dir, True)
    if len(x) == 0:
        raise ValueError('No training data loaded!')

    # Split dataset into training and validation set
    logging.debug('Splitting dataset for validation')
    x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y)
    logging.trace('Training set has %d samples', len(x_train))
    logging.trace('Test set has %d samples', len(x_test))

    logging.debug('TRAINING IN PROGRESS ...')
    model = sklearn.ensemble.RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1)
    model.fit(x_train, y_train)

    logging.debug('Evaluating model against test set')
    y_pred = model.predict(x_test)

    # Summarize training results
    logging.debug(
            'Classification Report ...\n%s',
            sklearn.metrics.classification_report(
                y_test, y_pred, target_names=['empty', model_name]))
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    logging.debug(
            'Confusion Matrix ...\n\t\tPredicted_Empty\tPredicted_%s\n'
            'Actual_empty\t%s\t\t%s\nActual_%s\t%s\t\t%s',
            model_name, cm[0][0], cm[0][1], model_name, cm[1][0], cm[1][1])
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    logging.debug(
            'Training accuracy: %.2f%% ; Test accuracy: %.2f%%',
            train_score * 100, test_score * 100)

    return model


def validate(model, match_dir, empty_dir):
    '''
    Run model against entire set of tagged data to verify trained model
    '''
    (x, y) = ai.model.prepare_dataset(match_dir, empty_dir, False)
    if len(x) == 0:
        raise ValueError('No training data loaded!')

    # Run inference on all training data
    predictions = model.predict(x)
    match_accuracy = np.sum(predictions[y == 1] == 1) / np.sum(y == 1)
    empty_accuracy = np.sum(predictions[y == 0] == 0) / np.sum(y == 0)
    overall_accuracy = np.sum(predictions == y) / len(y)

    # Summarize validation results
    logging.debug(
            'Match accuracy: %.2f%% ; Non-Match accuracy: %.2f%%',
            match_accuracy * 100, empty_accuracy * 100)
    if match_accuracy == 1.0 and empty_accuracy == 1.0:
        logging.info('SUCCESS: 100% accuracy on all audio samples!')
    else:
        logging.warning(
                'FAILED: Final accuracy is %.2f%%',
                overall_accuracy * 100)


if __name__ == '__main__':
    main()
