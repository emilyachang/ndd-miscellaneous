
from proglearn.deciders import SimpleArgmaxAverage

# redefine simpleargmaxaverage class to be
class NOTAVERAGE(SimpleArgmaxAverage):
    def predict_proba(self, X, transformer_ids=None):
        vote_per_transformer_id = []
        for transformer_id in (
            transformer_ids
            if transformer_ids is not None
            else self.transformer_id_to_voters.keys()
        ):
            if not self.is_fitted():
                msg = (
                    "This %(name)s instance is not fitted yet. Call 'fit' with "
                    "appropriate arguments before using this decider."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})

            vote_per_bag_id = []
            for bag_id in range(len(self.transformer_id_to_transformers[transformer_id])):
                transformer = self.transformer_id_to_transformers[transformer_id][bag_id]
                X_transformed = transformer.transform(X)
                voter = self.transformer_id_to_voters[transformer_id][bag_id]
                #
                print(voter.finite_sample_correction)
                #
                print(voter.predict(X_transformed))
                #
                vote = voter.predict_proba(X_transformed)
                vote_per_bag_id.append(vote)
            vote_per_transformer_id.append(np.mean(vote_per_bag_id, axis=0))
            decider_vote = np.mean(vote_per_transformer_id, axis=0)
            
        return decider_vote #, vote_per_transformer_id, vote_per_bag_id
