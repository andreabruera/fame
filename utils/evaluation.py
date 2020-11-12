import itertools
import scipy
import sklearn

def cos(v1, v2):
    similarity = sklearn.metrics.pairwise.cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1))
    return similarity

def pearson(v1, v2):
    corr = scipy.stats.pearsonr(v1, v2)[0]
    return corr

def predict_ranking(model, test_sample, gold_samples, similarity_measure='cos'):

    current_entity = test_sample[1]
    ranking_length = len([k for k in gold_samples.keys()])
    predictions = model.predict(test_sample[0].reshape(1, -1))[0]

    if similarity_measure == 'cos':
        sims = {k : cos(predictions, v) for k, v in gold_samples.items()}  
    elif similarity_measure == 'corr':
        sims = {k : pearson(predictions, v) for k, v in gold_samples.items()}  

    rankings = {k : sims[k] for k in sorted(sims, key = sims.get, reverse = True)}

    rank = 0
    for k, sim in rankings.items():
        rank += 1
        if k == current_entity:
            break
    accuracy = 1 - ((rank - 1) / (ranking_length - 1))

    return accuracy

def predict_pairwise(model, test_samples, gold_samples, similarity_measure='cos'):

    predictions_one = model.predict(test_samples[0][0].reshape(1, -1))[0]
    predictions_two = model.predict(test_samples[1][0].reshape(1, -1))[0]

    if similarity_measure == 'cos':
        right_sim = cos(predictions_one, gold_samples[test_samples[0][1]]) + cos(predictions_two, gold_samples[test_samples[1][1]])
        wrong_sim = cos(predictions_one, gold_samples[test_samples[1][1]]) + cos(predictions_two, gold_samples[test_samples[0][1]])
    elif similarity_measure == 'corr':
        right_sim = pearson(predictions_one, gold_samples[test_samples[0][1]]) + pearson(predictions_two, gold_samples[test_samples[1][1]])
        wrong_sim = pearson(predictions_one, gold_samples[test_samples[1][1]]) + pearson(predictions_two, gold_samples[test_samples[0][1]])
    if right_sim > wrong_sim:
        return 1.0
    else:
        return 0.0

