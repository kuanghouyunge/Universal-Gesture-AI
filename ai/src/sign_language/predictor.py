class CSLPredictor:

    def __init__(self, model=None):
        self.model = model

    def predict(self, sequence):

        if self.model is None:
            return {
                "label": None,
                "confidence": 0.0,
                "status": "model_not_ready"
            }

        # TODO:
        # sequence -> normalization
        # -> temporal model
        # -> CSL vocabulary mapping

        return self.model.predict(sequence)