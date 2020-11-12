
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
svc = SVC(kernel='linear')
mlp = MLPClassifier()
mlp_2_layers = MLPClassifier(hidden_layer_sizes = (100, 100))
mlp_regressor = MLPRegressor(hidden_layer_sizes = (100))
logistic_regression = LogisticRegression(solver='liblinear')

constant_features_remover = VarianceThreshold(threshold=0)
def category_decoding():
    condition_combinations = [p for p in itertools.combinations(range(len(all_conditions)), 2)]
    for p in condition_combinations:
        conditions = [all_conditions[i] for i in p]
        logging.info('Current conditions: {} and {}'.format(conditions[0], conditions[1]))
        
        #for left_out in range(0, 11, 5):
        #for left_out in range(5, 11, 5):
        for left_out in range(10, 11, 5):
            if left_out == 0:
                left_out = 2
            results = []
            # Finding the constant features

            full_data = [v for v in sub_images[conditions[0]]] + [v for v in sub_images[conditions[1]]]
            constant_features_remover.fit(full_data)

            for i in tqdm(range(100)):
          
                half_test_length = min(len(sub_images[conditions[0]]), len(sub_images[conditions[1]]))
                random_indices = numpy.random.choice(half_test_length, half_test_length, replace = False)
                train_data = []
                test_data = []
            
                for i, v in enumerate(random_indices):
                    if half_test_length*2 - i*2 > left_out:
                        train_data.append([conditions[0], sub_images[conditions[0]][v]])
                        train_data.append([conditions[1], sub_images[conditions[1]][v]])
                    else:
                        test_data.append([conditions[0], sub_images[conditions[0]][v]])
                        test_data.append([conditions[1], sub_images[conditions[1]][v]])

                # Removing the constant features
                train_data_filtered = constant_features_remover.transform([k[1] for k in train_data])
                test_data_filtered = constant_features_remover.transform([k[1] for k in test_data])

                # Performing the classification and selecting the most relevant features used for classification

                feature_selection = SelectPercentile(f_classif, percentile = 95)
                #feature_selection = SelectPercentile(f_classif, percentile = 5)
                anova_m = Pipeline([('anova', feature_selection), ('classifier', m)])
                #m.fit([k[1] for k in train_data], [k[0] for k in train_data])
                anova_m.fit([k for k in train_data_filtered], [k[0] for k in train_data])
                #prediction = m.predict([k[1] for k in test_data])
                prediction = anova_m.predict([k for k in test_data_filtered])
                result = (prediction == [k[0] for k in test_data]).sum() / float(len([k[0] for k in test_data]))
                results.append(result)
                histogram_results[model_name].append(result)

                # We draw the brain maps using only the leave-10 out classifications
                if left_out == 10:
                    coef_orig = m.coef_
                    coef = feature_selection.inverse_transform(coef_orig)
                    collection.append(coef.tolist()[0])

            final_results[model_name].append(numpy.average(results))
            if args.write_to_file:
                with open(os.path.join(model_path, 'classification_results.txt'), 'a') as o:

                    o.write('Conditions: {} vs {}\n\nEvaluation leaving {} out\n\nResults after 100 iterations: {} - Standard deviation: {}\n\n{}\n\n\n'.format(conditions[0], conditions[1], left_out, numpy.average(results), numpy.std(results), results))
            #logging.info('Conditions: {} vs {}\n\nEvaluation leaving {} out\n\nResults after 100 iterations: {} - Standard deviation: {}\n\n{}\n\n\n'.format(conditions[0], conditions[1], left_out, numpy.average(results), numpy.std(results), results))
            logging.info('Results after 100 iterations of a leave-{} out evaluation: average {}, median {} - Standard deviation: {}'.format(left_out, numpy.average(results), numpy.median(results), numpy.std(results)))

        ### Voxel analysis
        voxel_analysis(args, collection, constant_features_remover, masker, aal_atlas, talairach_atlas, model_path, conditions)
