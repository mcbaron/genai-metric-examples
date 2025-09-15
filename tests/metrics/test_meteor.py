from promptx_core.metric.meteor import Meteor


def test_meteor_score():
    meteor = Meteor()

    predictions = [
        "It is a guide to action which ensures that the \
            military always obeys the commands of the party"
    ]
    references = [
        "It is a guide to action that ensures that \
            the military will forever heed Party commands"
    ]

    results = meteor(references[0], predictions[0], dspy_example=False)
    results_batch = meteor.batch(references, predictions, dspy_example=False)
    assert round(results, 4) == 0.6944
    assert round(results_batch, 4) == 0.6944


def test_meteor_multiple_references():
    meteor = Meteor()

    predictions = [
        "It is a guide to action which ensures that the military \
            always obeys the commands of the party"
    ]
    references = [
        [
            "It is a guide to action that ensures that the \
                military will forever heed Party commands",
            "It is a guide that makes sure the military \
                follows party orders",
        ]
    ]

    results = meteor.batch(references, predictions, dspy_example=False)
    assert isinstance(results, float)


def test_meteor_parameters():
    meteor = Meteor()

    predictions = ["Test prediction"]
    references = ["Test reference"]

    results = meteor(
        references[0], predictions[0], dspy_example=False, alpha=0.8, beta=2, gamma=0.4
    )
    assert isinstance(results, float)
