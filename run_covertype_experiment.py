"""
Experiment: DynaGradTree Performance Evaluation

🆕 New Feature: Auto-run full comparison after Grid Search completion
    - No need to manually switch RUN_MODE
    - Suitable for parallel execution on multiple datasets
    - Complete Grid Search + comparison in one run

Design Philosophy:
Based on successful design patterns from phase5_river_trees_test.py, implementing a clear,
reproducible experimental workflow for evaluating DynaGradTree

Core Features:
1. Manual prequential evaluation loop (test-then-train)
2. Fixed feature space + missing value patterns
3. Detailed data collection (CER per batch)
4. CSV format result saving for downstream analysis
5. Auto-generate learning curves
6. 🆕 Auto comparison after Grid Search

Reference: Validated successful patterns from phase5_river_trees_test.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
from datetime import datetime


# ========================================================================
# Utility Functions: Data Generation and Preprocessing
# ========================================================================

def generate_synthetic_data(n_samples=10000, n_features=20, n_classes=2, random_seed=42):
    """
    Generate synthetic binary classification dataset

    This function creates linearly separable data and adds noise to make the problem more challenging.
    This data generation method has been validated through all tests in phase1-phase6 and is reliable.

    Parameter Description:
        n_samples: Number of samples
        n_features: feature count
        n_classes: number of classes (currently only supports 2)
        random_seed: random seed for reproducibility

    Returns:
        X: Feature matrix with shape(n_samples, n_features)
        y: Label array with shape(n_samples,)
    """
    np.random.seed(random_seed)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate true weight vector
    true_weights = np.random.randn(n_features)
    true_weights = true_weights / np.linalg.norm(true_weights)

    # Calculate scores and generate labels (with noise)
    scores = X @ true_weights
    noise = np.random.randn(n_samples) * 0.3
    y = (scores + noise > 0).astype(int)

    print(f"[Data Generation] {n_samples} samples  × {n_features}features")
    print(f"[Data Generation] Class distribution: class 0={np.sum(y == 0)}, class 1={np.sum(y == 1)}")

    return X, y


def ensure_first_batch_has_all_classes(X, y, batch_size=100, random_seed=42):
    """
    Reorder data to ensure first batch contains all classes

    This function solves a classic online learning problem: if model infers class count from first batch,
    and first batch happens to not contain all classes, it will cause "class index out of bounds" errors later.

    Why do we need this function?
    In online learning, models typically need to determine model structure (e.g., number of output neurons) when seeing the first batch of data.
    If first batch happens to miss a class, model will think there are N-1 classes instead of N.
    When the missed class appears in subsequent data, index out of bounds error will occur.

    Solution Strategy:
    1. Identify all unique classes
    2. Select at least one sample from each class
    3. Put these samples in first batch
    4. Fill first batch to specified size with other samples
    5. Randomly shuffle remaining data

    Parameter Description:
        X: featuresmatrix (n_samples, n_features)
        y: labelsarray (n_samples,)
        batch_size: abatch size (default 100)
        random_seed: random seed (ensures reproducibility)

    Returns:
        X_reordered: Reordered feature matrix
        y_reordered: Reordered label array
    """
    np.random.seed(random_seed)
    n_samples = len(y)
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    print(f"[Data Reorder] Ensuring first batch contains all{n_classes} classes...")

    # Step 1: Select at least one sample from each class
    first_batch_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        if len(cls_indices) > 0:
            # Randomly select one sample from this class
            selected = np.random.choice(cls_indices)
            first_batch_indices.append(selected)

    # Step 2: If first batch not large enough, randomly select from remaining samples
    remaining_indices = np.setdiff1d(np.arange(n_samples), first_batch_indices)
    if len(first_batch_indices) < batch_size and len(remaining_indices) > 0:
        n_additional = min(batch_size - len(first_batch_indices), len(remaining_indices))
        additional = np.random.choice(remaining_indices, size=n_additional, replace=False)
        first_batch_indices.extend(additional)
        remaining_indices = np.setdiff1d(remaining_indices, additional)

    # Step 3: Randomly shuffle remaining samples (preserve randomness)
    np.random.shuffle(remaining_indices)

    # Step 4: Combine into final index order
    final_indices = np.concatenate([first_batch_indices, remaining_indices])

    # Verify first batch indeed contains all classes
    first_batch_classes = np.unique(y[final_indices[:batch_size]])
    print(
        f"[Data Reorder] First batch ({batch_size} samples) contains {len(first_batch_classes)} classes: {sorted(first_batch_classes)}")

    if len(first_batch_classes) < n_classes:
        missing_classes = set(unique_classes) - set(first_batch_classes)
        print(f"[Warning] First batch still missing classes: {missing_classes}")
        print(f"       This may be because batch_size too small. Recommend increasing batch_size or checking class distribution")
    else:
        print(f"[Data Reorder] ✓ First batch contains all classes, model will initialize correctly")

    return X[final_indices], y[final_indices]


def create_fixed_features_with_missing(X_full, y, missing_ratio=0.3,
                                       pattern='trapezoidal', random_seed=42):
    """
    Create missing value patterns in fixed feature space

    This function is completely based on validated design from phase5_river_trees_test.py.
    We support two missing patterns:

    1. trapezoidal（trapezoidal）：earlysamplesmissingmany，afterphasesamplesmissingfew
       thissimulatefeaturesgradually become observabletrueactualscenario

    2. random（random）：eachvaluesindependently withmissing_ratioprobabilistic missing
       thissimulate sensor failuresetcrandommissingscenario

    Parameter Description:
        X_full: completefeaturesmatrix
        y: labelsarray
        missing_ratio: missing ratio（forattrapezoidal pattern，thisisinitialmissingrate）
        pattern: missing pattern，'trapezoidal'or'random'
        random_seed: random seed

    Returns:
        X_with_missing: withhasNaNmissingvaluesfeaturesmatrix
        y: originallabels（unchanged）
    """
    np.random.seed(random_seed)
    n_samples, n_features = X_full.shape
    X_with_missing = X_full.copy()

    if pattern == 'trapezoidal':
        # trapezoidal pattern：missingrate linearly decreases
        # thisisphase5validationsuccessdesign
        for i in range(n_samples):
            current_missing_ratio = missing_ratio * (1 - i / n_samples)
            n_missing = int(n_features * current_missing_ratio)

            if n_missing > 0:
                missing_features = np.random.choice(
                    n_features, size=n_missing, replace=False
                )
                X_with_missing[i, missing_features] = np.nan

    elif pattern == 'random':
        # random pattern：eachvaluesindependentmissing
        mask = np.random.rand(n_samples, n_features) < missing_ratio
        X_with_missing[mask] = np.nan

    else:
        raise ValueError(f"unknownmissing pattern: {pattern}")

    actual_missing_ratio = np.isnan(X_with_missing).sum() / X_with_missing.size
    print(f"[missing pattern] pattern={pattern}, targetmissingrate={missing_ratio:.1%}")
    print(f"[missing pattern] actualmissingrate={actual_missing_ratio:.1%}")

    return X_with_missing, y


# ========================================================================
# coreassessfunction
# ========================================================================

def run_prequential_evaluation(learner, X, y, batch_size=100,
                               learner_name="Algorithm"):
    """
    Run prequential evaluation

    thisisIsexperimentcorefunction，implementationstandardtest-then-trainassessloop。
    thisdesigncompletelybaseatphase5_river_trees_test.pyin366-394row，
    thatisathroughvalidationsuccesspattern。

    assessprocedure：
    1. abatchonlyuseatinitial training（nottest）
    2. aftercontinueper batch：
       a) firstusewhenbeforemodelprediction（test）
       b) recordpredictionwrong
       c) usetrueactuallabelsupdatemodel（train）
       d) computecumulative error rate

    thistest-then-trainsequenceguaranteeIsassessismodelfortruecurrentlynot yetsee
    datageneralizationcapability，thisisinonline learningassessgoldenstandard。

    Parameter Description:
        learner: learnerforobject，needhaspredictandupdatemethod
        X: featuresmatrix（maycontainNaN）
        y: labelsarray
        batch_size: per batchsize
        learner_name: algorithmname（useatshow）

    Returns:
        results_dict: containdetailed resultsdictionary
    """
    print(f"\n{'=' * 70}")
    print(f"assessalgorithm: {learner_name}")
    print(f"{'=' * 70}")

    n_samples = len(y)
    n_batches = n_samples // batch_size

    # initializestatisticsvariable
    total_predictions = 0
    total_errors = 0
    batch_results = []  # storageper batchdetailed info

    # computewhen
    total_train_time = 0
    total_predict_time = 0

    print(f"startPrequentialassess...")
    print(f"  totalNumber of samples: {n_samples}")
    print(f"  Batchsize: {batch_size}")
    print(f"  Batchcount: {n_batches}")

    # mainassessloop
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        # abatchspecialprocessing：onlytraining，nottest
        # thisisbecausemodelstillnohaslearningtoanywhatthing，testnohasmeaning
        if batch_idx < 10:
            print(f"  Batch 0: initial training（skiptest）")

            t_start = time.time()
            learner.update(X_batch, y_batch)
            train_time = time.time() - t_start
            total_train_time += train_time

            continue

        # aftercontinuebatch：firsttest，againtraining

        # Teststage：prediction
        t_start = time.time()
        predictions = learner.predict(X_batch)
        predict_time = time.time() - t_start
        total_predict_time += predict_time

        # computewrong
        batch_errors = np.sum(predictions != y_batch)
        total_errors += batch_errors
        total_predictions += len(y_batch)

        # computecumulative error rate
        cer = total_errors / total_predictions
        instant_accuracy = 1 - (batch_errors / len(y_batch))

        # Trainstage：updatemodel
        t_start = time.time()
        learner.update(X_batch, y_batch)
        train_time = time.time() - t_start
        total_train_time += train_time

        # recordthisbatchresults
        batch_results.append({
            'batch_idx': batch_idx,
            'cumulative_error_rate': cer,
            'instant_accuracy': instant_accuracy,
            'train_time': train_time,
            'predict_time': predict_time,
            'total_errors': total_errors,
            'total_predictions': total_predictions
        })

        # definephaseprintprogress
        if batch_idx % max(1, n_batches // 10) == 0:
            print(f"  Batch {batch_idx}/{n_batches}: CER={cer:.4f}, "
                  f"namelywhenaccuracy={instant_accuracy:.4f}")

    # assessdone，printfinalresults
    final_cer = total_errors / total_predictions if total_predictions > 0 else 0

    print(f"\nassessdone！")
    print(f"  finalCER: {final_cer:.4f}")
    print(f"  totalpredictioncount: {total_predictions}")
    print(f"  totalwrongcount: {total_errors}")
    print(f"  totaltrainingtime: {total_train_time:.2f}second")
    print(f"  totalpredictiontime: {total_predict_time:.2f}second")

    return {
        'algorithm': learner_name,
        'final_cer': final_cer,
        'total_predictions': total_predictions,
        'total_errors': total_errors,
        'total_train_time': total_train_time,
        'total_predict_time': total_predict_time,
        'batch_results': batch_results
    }


# ========================================================================
# resultssaveandvisualization
# ========================================================================

def save_results_to_csv(results_dict, output_dir='./results'):
    """
    saveexperimentresultstoCSVfile

    thisfunctionwillper batchdetailed resultssavetoCSVformat，convenientataftercontinuedivideanalysis
    andvisualization。CSVfilebenefitiscommonusestrong，canusepandas、Excelor
    othertooleasilyprocessing。

    filenamingformat：{algorithm_name}_results_{timestamp}.csv
    thislikecanavoid overwritingbeforeexperimentresults。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # generatefilename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    algorithm_name = results_dict['algorithm'].replace(' ', '_').replace('/', '_')
    filename = f"{algorithm_name}_results_{timestamp}.csv"
    filepath = output_path / filename

    # convert toDataFrameandsave
    df = pd.DataFrame(results_dict['batch_results'])
    df.to_csv(filepath, index=False)

    print(f"[resultssave] alreadysaveto: {filepath}")

    return filepath


def plot_learning_curve(results_dict, save_path=None):
    """
    plotlearning curve

    learning curvedisplaycumulative error ratehow toview anytimetomoremanydatawhilechange。
    thisisunderstandalgorithmlearningdynamicmostintuitiveway。

    Parameter Description:
        results_dict: run_prequential_evaluationReturnsresultsdictionary
        save_path: savepath，iftoNonethenonlyshownotsave
    """
    batch_results = results_dict['batch_results']

    if not batch_results:
        print("warn：nohasdatacanplot")
        return

    # extracteddata
    batch_indices = [r['batch_idx'] for r in batch_results]
    cer_values = [r['cumulative_error_rate'] for r in batch_results]

    # createcharttable
    plt.figure(figsize=(10, 6))
    plt.plot(batch_indices, cer_values,
             linewidth=2, color='steelblue', marker='o', markersize=4)

    plt.xlabel('Batch Index', fontsize=12)
    plt.ylabel('Cumulative Error Rate', fontsize=12)
    plt.title(f'Learning Curve: {results_dict["algorithm"]}',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # addfinalCERannotation
    final_cer = results_dict['final_cer']
    plt.axhline(y=final_cer, color='red', linestyle='--', alpha=0.5)
    plt.text(0.02, final_cer, f'Final CER: {final_cer:.4f}',
             transform=plt.gca().get_yaxis_transform(),
             verticalalignment='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[charttablesave] learning curvealreadysaveto: {save_path}")

    plt.show()
    plt.close()


def plot_comparison(results_list, save_path=None):
    """
    plotmanyalgorithmforcomparison chart

    thisfunctioncreatetwosubchart：
    1. learning curveforcomparison（hasalgorithminasheetcharton）
    2. finalperformanceforcomparison（barchart）

    Parameter Description:
        results_list: manyresults_dictgroupbecomelist
        save_path: savepath
    """
    if not results_list:
        print("warn：nohasresultscanforcomparison")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # subchart1：learning curveforcomparison
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    for results, color in zip(results_list, colors):
        batch_results = results['batch_results']
        if not batch_results:
            continue

        batch_indices = [r['batch_idx'] for r in batch_results]
        cer_values = [r['cumulative_error_rate'] for r in batch_results]

        ax1.plot(batch_indices, cer_values,
                 label=results['algorithm'],
                 linewidth=2, color=color, alpha=0.8)

    ax1.set_xlabel('Batch Index', fontsize=11)
    ax1.set_ylabel('Cumulative Error Rate', fontsize=11)
    ax1.set_title('Learning Curves Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # subchart2：finalCERforcomparison
    algorithms = [r['algorithm'] for r in results_list]
    final_cers = [r['final_cer'] for r in results_list]

    bars = ax2.bar(range(len(algorithms)), final_cers, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.set_ylabel('Final CER', fontsize=11)
    ax2.set_title('Final Performance Comparison', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # inbarsubonaddcountvalues
    for bar, cer in zip(bars, final_cers):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{cer:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[charttablesave] forcomparison chartalreadysaveto: {save_path}")

    plt.show()
    plt.close()


# ========================================================================
# algorithmcreatefunction
# ========================================================================

def create_gradtree_with_params(
    dataset_config,
    learning_rate_index=0.005,
    learning_rate_values=0.005,
    learning_rate_leaf=0.005,
    epochs=1,
    reserve_ratio=1.5,
    depth=6,
    optimizer='SWA'
):
    """
    useusespecifyparameterscreateGradTreeinstance（useatgrid search）

    Parameter Description:
        dataset_config: Datasetconfigdictionary（mustcontainn_featuresandn_classes）
        learning_rate_index/values/leaf: threelayerlearning rate
        epochs: per batchtrainingroundcount
        reserve_ratio: featuresemptybetweenreservedratio
        depth: treedeepdegree
        optimizer: optimizertype

    Returns:
        GradTreeLearnerinstance
    """
    from dynagradtree_learner import GradTreeLearner

    n_features = dataset_config['n_features']
    n_classes = dataset_config.get('n_classes', None)

    learner = GradTreeLearner(
        depth=depth,
        n_estimators=1,  # singletreepattern
        learning_rate_index=learning_rate_index,
        learning_rate_values=learning_rate_values,
        learning_rate_leaf=learning_rate_leaf,
        optimizer=optimizer,
        max_features_reserve_ratio=reserve_ratio,
        objective='classification',
        n_classes=n_classes,
        random_seed=42, #  Random Seed
        verbose=0,
        # trainingcontrolparameters
        epochs=epochs,
        batch_size=dataset_config['batch_size']
    )

    return learner


def grid_search_gradtree(X, y, dataset_config, param_grid, n_trials=1, verbose=1):
    """
    toDynaGradTreeenterrowgridsearchadjustparameter

    Parameter Description:
        X: feature countaccording
        y: labelsdata
        dataset_config: Datasetconfigdictionary
        param_grid: parametersgriddictionary
        n_trials: eachparameterarraycombineNumber of experiment repetitions（useatdecreaserandomnessimpact）
        verbose: detaileddegree（0=quiet，1=progress，2=detailed）

    Returns:
        best_params: bestparameterarraycombine
        results_df: hasexperimentresultsDataFrame
    """
    from itertools import product
    import pandas as pd
    import time
    from datetime import datetime

    if verbose >= 1:
        print("\n" + "=" * 70)
        print("DynaGradTree gridsearchadjustparameter")
        print("=" * 70)
        print(f"\nDataset: {dataset_config.get('name', 'Unknown')}")
        print(f"Number of samples: {len(X)}")
        print(f"feature count: {X.shape[1]}")
        print(f"Number of classes: {dataset_config.get('n_classes', 'Unknown')}")
        print(f"Batchsize: {dataset_config['batch_size']}")

    # generatehasparameterarraycombine
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))

    if verbose >= 1:
        print(f"\nparametersgrid：")
        for key, values in param_grid.items():
            print(f"  {key}: {values}")
        print(f"\nTotal {len(all_combinations)} parameterarraycombine")
        print(f"Repeat per combination {n_trials} times")
        print(f"estimatedtotalexperimenttimescount: {len(all_combinations) * n_trials}")

        # estimatetime
        estimated_time_per_trial = 10  # second（coarserough estimatecompute）
        total_estimated_time = len(all_combinations) * n_trials * estimated_time_per_trial
        print(f"estimatedtotaltime: ~{total_estimated_time / 60:.1f} divideminute")
        print("=" * 70)

    results = []
    best_cer = float('inf')
    best_params = None

    for idx, params in enumerate(all_combinations, 1):
        # buildparametersdictionary
        current_params = dict(zip(param_names, params))

        if verbose >= 1:
            print(f"\n[{idx}/{len(all_combinations)}] testparameterarraycombine:")
            for key, value in current_params.items():
                print(f"  {key} = {value}")

        # repeatexperimentn_trialstimes
        trial_cers = []
        trial_times = []

        for trial in range(n_trials):
            if verbose >= 1:
                print(f"  Trial {trial + 1}/{n_trials}...", end=" ", flush=True)

            # createGradTree learner
            learner = create_gradtree_with_params(
                dataset_config=dataset_config,
                **current_params
            )

            # runassess
            start_time = time.time()
            result = run_prequential_evaluation(
                learner=learner,
                X=X,
                y=y,
                batch_size=dataset_config['batch_size'],
                learner_name=f"DynaGradTree_trial{trial+1}"
            )
            elapsed_time = time.time() - start_time

            trial_cers.append(result['final_cer'])
            trial_times.append(elapsed_time)

            if verbose >= 1:
                print(f"CER={result['final_cer']:.4f}, Time={elapsed_time:.1f}s")

        # computeaverageperformance
        avg_cer = np.mean(trial_cers)
        std_cer = np.std(trial_cers)
        avg_time = np.mean(trial_times)

        if verbose >= 1:
            print(f"  Average CER: {avg_cer:.4f} (±{std_cer:.4f})")
            print(f"  averagetime: {avg_time:.1f}s")

        # Record results
        result_entry = current_params.copy()
        result_entry.update({
            'avg_cer': avg_cer,
            'std_cer': std_cer,
            'avg_time': avg_time,
            'min_cer': min(trial_cers),
            'max_cer': max(trial_cers),
            'all_cers': trial_cers  # savehastrialresults
        })
        results.append(result_entry)

        # updatebestparameters
        if avg_cer < best_cer:
            best_cer = avg_cer
            best_params = current_params.copy()
            if verbose >= 1:
                print(f"  ✓ newbestresults！")

    # createresultsDataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('avg_cer')

    # saveGrid Searchresults
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = dataset_config.get('name', 'unknown')
    results_file = f"./results/gradtree_gridsearch_{dataset_name}_{timestamp}.csv"

    # savewhendropall_cerscolumns（toolarge）
    results_df_to_save = results_df.drop(columns=['all_cers'])
    results_df_to_save.to_csv(results_file, index=False)

    if verbose >= 1:
        print("\n" + "=" * 70)
        print("gridsearchdone！")
        print("=" * 70)
        print(f"\nbestparameterarraycombine（Average CER={best_cer:.4f}）：")
        for key, value in best_params.items():
            print(f"  {key} = {value}")

        print(f"\nTop 5 parameterarraycombine：")
        print(results_df_to_save.head(5).to_string(index=False))

        print(f"\ndetailed resultsalreadysaveto: {results_file}")
        print("=" * 70)

    # ============================================================================
    # 🆕 New Feature: Auto-run full comparison after Grid Search completion
    # ============================================================================
    if verbose >= 1:
        print("\n" + "=" * 70)
        print("🚀 Grid Searchdone！automaticallystartforcomparisonexperiment...")
        print("=" * 70)
        print(f"\nuseusebestparametersruncompleteforcomparisonexperiment：")
        for key, value in best_params.items():
            print(f"  {key} = {value}")
        print()

    try:
        # 1. useusebestparameterscreateDynaGradTreeandassess
        print("📊 [1/4] assessDynaGradTree（bestparameters）...")
        
        best_learner = create_gradtree_with_params(
            dataset_config=dataset_config,
            **best_params
        )
        
        dyna_result = run_prequential_evaluation(
            learner=best_learner,
            X=X,
            y=y,
            batch_size=dataset_config['batch_size'],
            learner_name=f"DynaGradTree"
        )
        
        # saveDynaGradTreeresults
        save_results_to_csv(dyna_result, output_dir='./results')
        all_results = [dyna_result]
        
        # 2. runtraditionalinonline decision treeforcomparison
        print(f"\n{'=' * 70}")
        print("📊 [2/4] runtraditionalinonline decision treeforcomparison...")
        print(f"{'=' * 70}\n")
        
        try:
            from river_trees_learner import create_hoeffding_tree, create_efdt, create_hat
            
            n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
            
            # create3traditionaltree
            traditional_trees = [
                ('HT_mean', create_hoeffding_tree(n_features=n_features, missing_strategy='mean')),
                ('EFDT_mean', create_efdt(n_features=n_features, missing_strategy='mean')),
                ('HAT_mean', create_hat(n_features=n_features, missing_strategy='mean'))
            ]
            
            for tree_name, tree_learner in traditional_trees:
                result = run_prequential_evaluation(
                    learner=tree_learner,
                    X=X,
                    y=y,
                    batch_size=dataset_config['batch_size'],
                    learner_name=tree_name
                )
                all_results.append(result)
                save_results_to_csv(result, output_dir='./results')
            
            # 3. generateforcomparison charttable
            print(f"\n{'=' * 70}")
            print("📊 [3/4] generateforcomparison charttable...")
            print(f"{'=' * 70}\n")
            
            dataset_name = dataset_config.get('name', 'unknown')
            plot_comparison(
                all_results,
                save_path=f"./results/{dataset_name}_auto_comparison_{timestamp}.png"
            )
            
            # 4. printfinalranking
            print("\n" + "=" * 70)
            print("📊 [4/4] automaticallyforcomparisonExperiment complete！finalresultsranking：")
            print("=" * 70)
            
            sorted_results = sorted(all_results, key=lambda x: x['final_cer'])
            for rank, result in enumerate(sorted_results, 1):
                cer = result['final_cer']
                algo = result['algorithm']
                
                # toDynaGradTreeaddspecialMark
                if 'DynaGradTree' in algo:
                    marker = "🏆" if rank == 1 else "⭐"
                    print(f"{rank}. {marker} {algo:<35} CER={cer:.4f}")
                else:
                    print(f"{rank}.    {algo:<35} CER={cer:.4f}")
            
            print("=" * 70)
            
            # saveranking results
            ranking_file = f"./results/{dataset_name}_ranking_{timestamp}.txt"
            with open(ranking_file, 'w', encoding='utf-8') as f:
                f.write(f"experimenttime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Number of samples: {len(X)}\n")
                f.write(f"bestparameters:\n")
                for key, value in best_params.items():
                    f.write(f"  {key} = {value}\n")
                f.write(f"\nfinalranking:\n")
                for rank, result in enumerate(sorted_results, 1):
                    f.write(f"{rank}. {result['algorithm']:<35} CER={result['final_cer']:.4f}\n")
            
            print(f"\n💾 ranking resultsalreadysave: {ranking_file}")
            
        except ImportError as e:
            print(f"\n⚠️  nomethodimportRiverdecisiontree：{e}")
            print("onlysaveDynaGradTreeresults")
            
    except Exception as e:
        print(f"\n❌ automaticallyforcomparisonexperimenterror：{e}")
        import traceback
        traceback.print_exc()
        print("\nbutGrid Searchresultsalreadysave，canmanualrunforcomparisonexperiment。")
    
    # ReturnsGrid Searchresults（keeporiginalhasinterface）
    return best_params, results_df


def create_algorithms(n_features, n_classes=None, batch_size=100):
    """
    Create DynaGradTree algorithm instance
    
    Parameters:
        n_features: Number of features (for compatibility, not used by DynaGradTree)
        n_classes: Number of classes (optional but recommended)
        batch_size: Batch size for training
    
    Returns:
        algorithms: List of (name, learner) tuples
    """
    algorithms = []
    
    # Import DynaGradTree learner
    try:
        from dynagradtree_learner import create_gradtree_medium
        GRADTREE_AVAILABLE = True
        print("[Import Success] DynaGradTree adapter")
    except ImportError as e:
        GRADTREE_AVAILABLE = False
        print(f"[Import Failed] DynaGradTree adapter: {e}")
        print("            Will skip DynaGradTree")
    
    # Add DynaGradTree
    if GRADTREE_AVAILABLE:
        learner = create_gradtree_medium(
            max_features_reserve_ratio=1.5,
            n_classes=n_classes,
            batch_size=batch_size,
            epochs=3,
            depth=6,
            learning_rate_index=0.005,
            learning_rate_values=0.005,
            learning_rate_leaf=0.005,
        )
        algorithms.append(('DynaGradTree', learner))
        if n_classes:
            print(f"  ✓ Added DynaGradTree (medium config, reserve_ratio=1.5x, n_classes={n_classes}, batch_size={batch_size})")
        else:
            print(f"  ✓ Added DynaGradTree (medium config, reserve_ratio=1.5x, batch_size={batch_size}, classes inferred from data)")
    
    # Check if any algorithms available
    if not algorithms:
        print("\n[Error] No algorithms available!")
        print("       Please ensure dynadynagradtree_learner.py")
        print("       is in current directory with all dependencies installed.")
        return []
    
    print(f"\nTotal algorithms created: {len(algorithms)}")
    return algorithms


# ========================================================================
# mainexperimentfunction
# ========================================================================

def run_single_experiment(experiment_config):
    """
    runsingleDatasetoncompleteexperiment

    thisfunctionorchestrateswholeexperimentprocedure：
    1. generateorloaddata
    2. createmissing value patterns
    3. createhasalgorithm
    4. foreachalgorithmrunprequentialassess
    5. saveresultsandgeneratecharttable

    Parameter Description:
        experiment_config: experimentconfigdictionary，contain：
            - name: experimentname
            - dataset_type: 'synthetic' or 'real'
            - real_dataset_name: trueactualDatasetname（ifdataset_type='real'）
            - n_samples: Number of samples（onlyuseatsynthetic）
            - n_features: feature count（onlyuseatsynthetic）
            - batch_size: batchsize
            - missing_ratio: missingrate
            - missing_pattern: missing pattern
    """
    print("\n" + "=" * 70)
    print(f"experiment：{experiment_config['name']}")
    print("=" * 70)

    # step1：loaddata
    print("\n[step1] loaddata...")

    if experiment_config.get('dataset_type', 'synthetic') == 'real':
        # loadtrueactualDataset
        from dataset_loader import load_dataset

        real_dataset_name = experiment_config.get('real_dataset_name', 'breast_cancer')
        n_samples = experiment_config.get('n_samples', None)

        print(f"loadtrueactualDataset: {real_dataset_name}")
        X, y, dataset_info = load_dataset(
            dataset_name=real_dataset_name,
            n_samples=n_samples,
            random_seed=42
        )

        # updateexperiment_configwithreflectactualDatasetsize
        experiment_config['n_samples'] = len(y)
        experiment_config['n_features'] = X.shape[1]
        experiment_config['n_classes'] = dataset_info['n_classes']

        print(f"\nDatasetinfo:")
        print(f"  description: {dataset_info['description']}")
        print(f"  actualNumber of samples: {dataset_info['n_samples']}")
        print(f"  actualfeature count: {dataset_info['n_features']}")
        print(f"  Number of classes: {dataset_info['n_classes']}")
    else:
        # generatecombinebecomedata
        print("generatecombinebecomedata...")
        X, y = generate_synthetic_data(
            n_samples=experiment_config['n_samples'],
            n_features=experiment_config['n_features'],
            random_seed=42
        )

        # fromlabelsininferNumber of classes
        n_classes = len(np.unique(y))
        experiment_config['n_classes'] = n_classes
        print(f"  Number of samples: {len(y)}")
        print(f"  feature count: {X.shape[1]}")
        print(f"  Number of classes: {n_classes}")

    # step2：Reorder data to ensure first batch contains all classes
    print("\n[step2] reordercolumnsdata...")
    X, y = ensure_first_batch_has_all_classes(
        X, y,
        batch_size=experiment_config['batch_size'],
        random_seed=42
    )

    # step3：createmissing value patterns
    print("\n[step3] createmissing value patterns...")
    X_with_missing, y = create_fixed_features_with_missing(
        X, y,
        missing_ratio=experiment_config['missing_ratio'],
        pattern=experiment_config.get('missing_pattern', 'trapezoidal'),
        random_seed=42
    )

    # step4：createalgorithm
    print("\n[step4] createalgorithm...")
    algorithms = create_algorithms(
        n_features=experiment_config['n_features'],
        n_classes=experiment_config.get('n_classes', None),
        batch_size=experiment_config['batch_size'],
        test_all_strategies=experiment_config.get('test_all_strategies', False)
    )

    if not algorithms:
        print("[wrong] nohascanusealgorithm！pleaseimplementationcreate_algorithmsfunction")
        return None

    print(f"willtest {len(algorithms)} algorithm")

    # step5：assesshasalgorithm
    print("\n[step5] assesshasalgorithm...")
    all_results = []

    for algo_name, learner in algorithms:
        try:
            results = run_prequential_evaluation(
                learner=learner,
                X=X_with_missing,
                y=y,
                batch_size=experiment_config['batch_size'],
                learner_name=algo_name
            )
            all_results.append(results)

            # savesinglealgorithmresults
            save_results_to_csv(results, output_dir='./results')

        except Exception as e:
            print(f"[wrong] algorithm {algo_name} assessfail: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # step6：generateforcomparison charttable
    if len(all_results) > 1:
        print("\n[step6] generateforcomparison charttable...")
        plot_comparison(
            all_results,
            save_path=f"./results/{experiment_config['name']}_comparison.png"
        )
    elif len(all_results) == 1:
        print("\n[step6] generatelearning curve...")
        plot_learning_curve(
            all_results[0],
            save_path=f"./results/{experiment_config['name']}_learning_curve.png"
        )

    # printfinalranking
    if all_results:
        print("\n" + "=" * 70)
        print("finalresultsranking:")
        print("=" * 70)

        sorted_results = sorted(all_results, key=lambda x: x['final_cer'])

        for rank, results in enumerate(sorted_results, 1):
            print(f"{rank}. {results['algorithm']:<30} CER={results['final_cer']:.4f}")

    return all_results


# ========================================================================
# 🆕 parallelexperimentconfig（useatmanyDatasetsamewhenrun）
# ========================================================================

# config1: Covertypedifferentscale
experiment_config_covertype_10k = {
    'name': 'covertype_10k',
    'dataset_type': 'real',
    'real_dataset_name': 'covertype',
    'n_samples': 10000,
    'batch_size': 100,
    'missing_ratio': 0.3,
    'missing_pattern': 'trapezoidal', #random
    'test_all_strategies': False
}

experiment_config_covertype_20k = {
    'name': 'covertype_20k',
    'dataset_type': 'real',
    'real_dataset_name': 'random',
    'n_samples': 20000,
    'batch_size': 100,
    'missing_ratio': 0.3,
    'missing_pattern': 'random',
    'test_all_strategies': False
}

experiment_config_covertype_30k = {
    'name': 'covertype_30k',
    'dataset_type': 'real',
    'real_dataset_name': 'covertype',
    'n_samples': 30000,
    'batch_size': 100,
    'missing_ratio': 0.3,
    'missing_pattern': 'random',
    'test_all_strategies': False
}

experiment_config_covertype_50k = {
    'name': 'covertype_50k',
    'dataset_type': 'real',
    'real_dataset_name': 'covertype',
    'n_samples': 50000,
    'batch_size': 100,
    'missing_ratio': 0.3,
    'missing_pattern': 'random',
    'test_all_strategies': False
}

# config2: Electricitydifferentscale
experiment_config_electricity_20k = {
    'name': 'electricity_20k',
    'dataset_type': 'real',
    'real_dataset_name': 'electricity',
    'n_samples': 20000,
    'batch_size': 100,
    'missing_ratio': 0.3,
    'missing_pattern': 'random',
    'test_all_strategies': False
}

experiment_config_electricity_full = {
    'name': 'electricity_45k',
    'dataset_type': 'real',
    'real_dataset_name': 'electricity',
    'n_samples': 20000,  # all45312samples
    'batch_size': 100,
    'missing_ratio': 0.3,
    'missing_pattern': 'random',
    'test_all_strategies': False
}

# config3: Adultdifferentscale
experiment_config_adult_20k = {
    'name': 'adult_20k',
    'dataset_type': 'real',
    'real_dataset_name': 'adult',
    'n_samples': 20000,
    'batch_size': 100,
    'missing_ratio': 0.3,
    'missing_pattern': 'random',
    'test_all_strategies': False
}

experiment_config_adult_full = {
    'name': 'adult_48k',
    'dataset_type': 'real',
    'real_dataset_name': 'adult',
    'n_samples': 20000,  # None all48842samples
    'batch_size': 100,
    'missing_ratio': 0.3,
    'missing_pattern': 'random',
    'test_all_strategies': False
}

# config4: differentbatch_sizetest
experiment_config_covertype_20k_bs50 = {
    'name': 'covertype_20k_bs50',
    'dataset_type': 'real',
    'real_dataset_name': 'covertype',
    'n_samples': 20000,
    'batch_size': 50,
    'missing_ratio': 0.3,
    'missing_pattern': 'random',
    'test_all_strategies': False
}

experiment_config_covertype_20k_bs200 = {
    'name': 'covertype_20k_bs200',
    'dataset_type': 'real',
    'real_dataset_name': 'covertype',
    'n_samples': 20000,
    'batch_size': 200,
    'missing_ratio': 0.3,
    'missing_pattern': 'random',
    'test_all_strategies': False
}

# ============================================================================
# 🎯 parallelexperimentconfiglist（methodconvenientchoose）
# ============================================================================

PARALLEL_CONFIGS = [
    experiment_config_covertype_10k,      # [0] instance1
    experiment_config_covertype_20k,      # [1] instance2
    experiment_config_covertype_30k,      # [2] instance3
    experiment_config_covertype_50k,      # [3] instance4
    experiment_config_electricity_20k,    # [4] instance5
    experiment_config_electricity_full,   # [5] instance6
    experiment_config_adult_20k,          # [6] instance7
    experiment_config_adult_full,         # [7] instance8
    experiment_config_covertype_20k_bs50, # [8] instance9
    experiment_config_covertype_20k_bs200,# [9] instance10
]


# ========================================================================
# mainprogram entry
# ========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DynaGradTree vs traditionalinonline decision tree - automaticallyizeforcomparisonexperiment")
    print("=" * 70)
    print("\n🆕 newfunction：Grid SearchdoneafterautomaticallyrunNormalforcomparison")
    print("   - atimesrundoneparametersoptimize+performanceforcomparison")
    print("   - Suitable for parallel execution on multiple datasets")
    print("   - fastfind outDynaGradTreeadvantageDataset")

    # ======================================================================
    # runpattern（keepgridsearch，becauseitwill automaticallyrunnormalforcomparison）
    # ======================================================================
    RUN_MODE = 'gridsearch'
    
    # ======================================================================
    # 🎯 Datasetchoose（parallelrunwhenmodifythisin）
    # ======================================================================
    # 
    # method1: directchoosePARALLEL_CONFIGSinsome
    experiment_config = PARALLEL_CONFIGS[0]  # choose0config
    # experiment_config = PARALLEL_CONFIGS[1]  # choose1config
    # ... withthisclassinfer
    #
    # method2: manualspecify（useattestorspecialconfig）
    #experiment_config = experiment_config_covertype_20k
    # 
    # ======================================================================

    # printwhenbeforeexperimentconfiginfo
    print(f"\n{'=' * 70}")
    print(f"whenbeforechooseexperimentconfig: {experiment_config['name']}")
    print(f"{'=' * 70}")
    print(f"  Datasettype: {experiment_config['dataset_type']}")
    if experiment_config['dataset_type'] == 'real':
        print(f"  trueactualDataset: {experiment_config['real_dataset_name']}")
        if experiment_config['n_samples'] is None:
            print(f"  Number of samples: useuseallsamples")
        else:
            print(f"  Number of samples: {experiment_config['n_samples']} (adoptlike)")
    else:
        print(f"  Number of samples: {experiment_config['n_samples']}")
        print(f"  feature count: {experiment_config['n_features']}")
    print(f"  Batchsize: {experiment_config['batch_size']}")
    print(f"  missingrate: {experiment_config['missing_ratio']}")
    print(f"  missing pattern: {experiment_config['missing_pattern']}")
    print(f"{'=' * 70}")

    # ======================================================================
    # Grid Searchparametersgrid（hasDatasetuseusesameconfig）
    # ======================================================================
    
    # fastGrid Search（3combinations）- useattest
    param_grid_quick = {
        'learning_rate_index': [0.005],
        'learning_rate_values': [0.005],
        'learning_rate_leaf': [0.005],
        'epochs': [3],
        'depth': [6],
        'reserve_ratio': [1.5]
    }
    
    # standardGrid Search（9-27combinations）- useatcurrentlystyleexperiment
    param_grid_standard = {
        'learning_rate_index': [0.005],
        'learning_rate_values': [0.005],
        'learning_rate_leaf': [0.005],
        'epochs': [6],
        'depth': [8],
        'reserve_ratio': [1.1]
    }
    
    # chooseuseusewhichparametersgrid
    param_grid = param_grid_standard  # 🔧 modifythisinchoosedifferentparametersgrid

    # ======================================================================
    # startexperiment
    # ======================================================================
    
    print(f"\nreadystartexperiment...")
    print(f"  runpattern: Grid Search + automaticallyforcomparison")
    print(f"  parameterarraycombinecount: {len([1 for _ in __import__('itertools').product(*param_grid.values())])}")
    
    if RUN_MODE == 'gridsearch':
        # Grid Searchpattern（will automaticallyrunNormalforcomparison）
        print("\n" + "=" * 70)
        print("runpattern: Grid Search + automaticallyforcomparison")
        print("=" * 70)

        # loaddata
        print("\n[step1] loaddata...")
        if experiment_config['dataset_type'] == 'synthetic':
            from dataset_loader import generate_synthetic_dataset

            X, y, dataset_info = generate_synthetic_dataset(
                n_samples=experiment_config['n_samples'],
                n_features=experiment_config['n_features']
            )
        else:
            from dataset_loader import load_dataset

            X, y, dataset_info = load_dataset(
                dataset_name=experiment_config['real_dataset_name'],
                n_samples=experiment_config.get('n_samples')
            )

        # readydataset_config
        dataset_config = {
            'name': dataset_info['name'],
            'n_features': X.shape[1],
            'n_classes': dataset_info.get('n_classes'),
            'batch_size': experiment_config['batch_size']
        }

        # reordercolumnsdata
        print("\n[step2] reordercolumnsdata...")
        X, y = ensure_first_batch_has_all_classes(
            X, y,
            batch_size=dataset_config['batch_size']
        ) # TODO Random Seed

        # createmissing value patterns
        print("\n[step3] createmissing value patterns...")
        X_with_missing, y = create_fixed_features_with_missing(
            X, y,
            missing_ratio=experiment_config['missing_ratio'],
            pattern=experiment_config['missing_pattern'],
            random_seed=1
        ) # TODO Random Seed Origin:42

        # runGrid Search（will automaticallyrunforcomparisonexperiment）
        print("\n[step4] runGrid Search + automaticallyforcomparison...")
        best_params, results_df = grid_search_gradtree(
            X=X_with_missing,
            y=y,
            dataset_config=dataset_config,
            param_grid=param_grid,
            n_trials=1,  # ⚠️ keepto1，notneedmodify
            verbose=1
        )

        print("\n" + "=" * 70)
        print("✅ completeexperimentproceduredone！")
        print("=" * 70)
        print("\nresultsfile:")
        print("  📊 Grid Searchresults: ./results/gradtree_gridsearch_*.csv")
        print("  📈 forcomparison charttable: ./results/*_auto_comparison_*.png")
        print("  🏆 ranking results: ./results/*_ranking_*.txt")
        print("  📁 detailedCSV: ./results/*_results_*.csv")
        print("=" * 70)

    else:
        # Normalpattern（notshouldthebeexecute，becauseIssetRUN_MODE='gridsearch'）
        results = run_single_experiment(experiment_config)

        print("\n" + "=" * 70)
        print("Experiment complete！")
        print("=" * 70)
        print(f"resultsalreadysaveto ./results directory")
        print("=" * 70)
