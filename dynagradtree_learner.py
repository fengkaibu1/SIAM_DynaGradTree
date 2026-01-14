"""
GradTreeingonline learning adapter

thismoduleimplementationGradTreetounifiedexperimentframeworkadapt。criticaldesign considerations：

1. ingonline learningpattern：
   - firstbatch：initializeandtrainingmodel
   - subsequentbatch：continuetraining（no rebuildmodel，keepweight）
   - predict：useusewhencurrentmodelstatusprediction

2. feature spacedynamicchange：
   - GradTreealreadyalready built-indynamic featuresprocessingcapability
   - Isonly needscorrectPassdatanamelycan

3. optimizerstatuskeep：
   - especiallyimportant：notingeachtimesupdaterecompilemodel
   - useuserebuild_model=Falsekeepoptimizerstatus

4. parametersconfig：
   - eachbatchonlytraining1epoch（ingonline learning）
   - useuseskip_preprocessing=True（assumptiondataalreadystandardization）
   - batch_sizematchexperimentframeworkbatch size
"""

import numpy as np
import pandas as pd
import sys
import os

# addGradTreeingdirectorytopath
# directorystructure：
# GradTree/
# ├── DynaGradTree_Module/
# │   └── DynaGradTree.py  ← Isneedimportfile
# └── Experiment_For_DynaGradTree/
#     └── gradtree_learner.py  ← whencurrentfile

# getwhencurrentfileingdirectory（Experiment_For_DynaGradTree）
current_dir = os.path.dirname(os.path.abspath(__file__))

# getGradTreerootdirectory（Experiment_For_DynaGradTreeparentdirectory）
gradtree_root = os.path.dirname(current_dir)

# getDynaGradTree_Moduledirectory
gradtree_dynamic_dir = os.path.join(gradtree_root, 'DynaGradTree_Module')

# willDynaGradTree_ModuledirectoryaddtoPythonpath
if gradtree_dynamic_dir not in sys.path:
    sys.path.insert(0, gradtree_dynamic_dir)

print(f"[debug] attemptfrombelowpathimportGradTree: {gradtree_dynamic_dir}")

# importframeworkbaseclass
from vfs_experiment_framework import OnlineLearnerBase

# importGradTree
# Nowshouldcandirectlyimport，becauseIsalreadyalreadyDynaGradTree_Moduleaddtopath
try:
    from DynaGradTree import GradTree

    print("[success] GradTreeimportsuccess！")
except ImportError as e:
    print(f"[wrong] unableimportGradTree: {e}")
    print(f"[hint] pleasecheckbelowpathyesnoexist:")
    print(f"       {gradtree_dynamic_dir}")
    print(f"       {os.path.join(gradtree_dynamic_dir, 'DynaGradTree.py')}")
    GradTree = None


class GradTreeLearner(OnlineLearnerBase):
    """
    GradTreeingonline learning adapter

    thisclasswillGradTreeadapttoIsunifiedexperimentframeworkin。itprocessingbelowcriticalproblem：
    1. ingonline learningbatchupdatepattern
    2. dynamic feature spaceprocessing
    3. optimizerstatuspersistence
    4. predictionandupdatecoordinate

    useuseexample：
        learner = GradTreeLearner(
            depth=5,
            n_estimators=50,
            learning_rate_values=0.1,
            max_features_reserve_ratio=2.0
        )

        # ingdata streamperform oningonline learning
        for X_batch, y_batch in data_stream:
            predictions = learner.predict(X_batch)
            learner.update(X_batch, y_batch)
    """

    def __init__(self,
                 depth=6,
                 n_estimators=50,
                 learning_rate_index=0.01,
                 learning_rate_values=0.01,
                 learning_rate_leaf=0.01,
                 optimizer='SWA',
                 max_features_reserve_ratio=2.0,
                 objective='binary',
                 n_classes=None,  # added：optionalNumber of classes
                 random_seed=42,
                 verbose=0,
                 **kwargs):
        """
        initializeGradTreelearner

        Args:
            depth: treedepth
            n_estimators: treecount（ensemble size）
            learning_rate_*: eachlayerlearning rate
            optimizer: optimizertype ('SWA', 'Adam', 'SGD'etc)
            max_features_reserve_ratio: reserved spaceratio（important！）
            objective: 'binary', 'classification', or 'regression'
            n_classes: Number of classesamount（optional）。ifprovide，willuseusethisvaluesrather thanyesfromdatainfer。
                      stronglyrecommendingmulti-class classificationtaskinprovide，avoidclassesemptybetweeninfernotcompleteproblem。
            random_seed: random seed
            verbose: detailedfinedegree (0: silent, 1: basic, 2: detailedfine)
            **kwargs: otherGradTreeparameters
        """
        super().__init__(name="GradTree", **kwargs)

        # checkGradTreeyesnosuccessimport
        if GradTree is None:
            raise ImportError("GradTreenot yetsuccessimport。pleasecheckimportpath。")

        # saveparametersconfig
        # thisinIsprovideGradTreephaseexpectcompleteparametersset，ensurenotwillbecausemissingparameterswhileouterror
        # parametersdividetoseveralclass：
        # 1. treestructureparameters：depth, n_estimators
        # 2. learning rateparameters：learning_rate_*
        # 3. optimizerparameters：optimizer, cosine_decay_steps
        # 4. lossfunctionparameters：loss, focal_loss
        # 5. tasktypeparameters：objective, from_logits
        # 6. classesbalanceparameters：apply_class_balancing
        # 7. assessparameters：metrics
        # 8. trainingcontrolparameters：epochs, batch_size, early_stopping_epochs
        # 9. otherparameters：random_seed, verbose, initializer, temperature
        self.params = {
            # treestructureparameters
            'depth': depth,
            'n_estimators': n_estimators,

            # learning rateparameters
            'learning_rate_index': learning_rate_index,
            'learning_rate_values': learning_rate_values,
            'learning_rate_leaf': learning_rate_leaf,

            # optimizerparameters
            'optimizer': optimizer,
            'cosine_decay_steps': 0,  # defaultnot useusecosine decay

            # lossfunctionparameters
            'loss': 'crossentropy',  # classificationtaskuseusecross entropy
            'focal_loss': False,  # defaultnot useusefocal loss

            # dynamic feature spaceparameters
            'max_features_reserve_ratio': max_features_reserve_ratio,

            # tasktypeparameters
            'objective': objective,  # will be used inupdatewhenaccording toNumber of classesdynamicadjust
            'from_logits': True,  # outputlogitsrather thanyesprobability

            # classesbalanceparameters
            'apply_class_balancing': True,  # automaticallyprocessingclassesnotbalance

            # assessparameters
            'metrics': ['Accuracy'],  # useuseaccuracyastoassessmetric

            # trainingcontrolparameters
            'epochs': 1,  # ingonline learning：eachbatchonlytraining1epoch
            'batch_size': 64,  # thiswill be used inupdatewhenaccording toactualbatch sizeadjust
            'early_stopping_epochs': 10,  # early stopping patiencevalues

            # otherparameters
            'random_seed': random_seed,
            'verbose': verbose,
            'initializer': 'RandomNormal',  # params for initializationmethod
            'temperature': 0.0,  # Gumbel-Softmaxtemperature（0meansnot useuse）
        }

        # allowthroughkwargscoverdefaultparameters
        self.params.update(kwargs)

        # modelinstance（latencyinitialize）
        self.model = None
        self._is_first_batch = True
        self._n_features_seen = 0
        self._n_classes = n_classes  # ifproviden_classes，directlyuseuse；nothentoNone，willfromdatainfer

        # statisticsinfo
        self.n_batches_processed = 0
        self.feature_expansion_count = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        prediction

        Args:
            X: shapeto(n_samples, n_features)feature matrix

        Returns:
            predictions: shapeto(n_samples,)predictionlabels
        """
        if self.model is None:
            # ifmodelalsonoinitialize，Returnsdefaultprediction
            # thistypecaseonlyingfirstbatchteststagewilloccur
            if self._n_classes is not None and self._n_classes > 2:
                # multi-class classification：Returns0（firstclasses）
                return np.zeros(len(X), dtype=int)
            else:
                # twoclassification：Returns0
                return np.zeros(len(X), dtype=int)

        try:
            # useuseGradTreepredictmethod
            # Note：GradTreepredictwillautomaticallyprocessingdynamic feature space
            predictions = self.model.predict(X)

            # processingpredictionresultshape
            if len(predictions.shape) > 1:
                if predictions.shape[1] == 1:
                    # twoclassification：(n, 1) -> (n,)
                    predictions = predictions.flatten()
                    predictions = (predictions > 0.5).astype(int)
                else:
                    # multi-class classification：(n, n_classes) -> (n,)
                    predictions = np.argmax(predictions, axis=1)
            else:
                # Already(n,)shape
                if self.params['objective'] == 'binary':
                    predictions = (predictions > 0.5).astype(int)

            return predictions.astype(int)

        except Exception as e:
            print(f"predictionwhenouterror: {str(e)}")
            # Returnsdefaultprediction
            return np.zeros(len(X), dtype=int)

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        updatemodel

        thismethodimplementationingonline learningcriticallogic：
        - firsttimescall：initializemodelandtraining
        - subsequentcall：continuetraining（no rebuildmodel）

        Args:
            X: shapeto(n_samples, n_features)feature matrix
            y: shapeto(n_samples,)trueactuallabels
        """
        # updatefeature count
        current_n_features = X.shape[1]
        if current_n_features > self._n_features_seen:
            self._n_features_seen = current_n_features

        # inferNumber of classesamount（ifalsonohavesetthen）
        if self._n_classes is None:
            # nohavedefaultNumber of classes，fromdatainfer
            unique_classes = np.unique(y)
            self._n_classes = len(unique_classes)
            print(f"[GradTree] fromdatainferNumber of classes: {self._n_classes}")

            # according toNumber of classesamountadjustobjective
            # Note：GradTreecompilemethodphaseexpect'classification'rather thanyes'multiclass'
            if self._n_classes == 2:
                self.params['objective'] = 'binary'
            else:
                self.params['objective'] = 'classification'  # GradTreeuseuse'classification'meansmulti-class classification
        else:
            # useusedefaultNumber of classes
            # stillnaturallyneedaccording toNumber of classessetcorrectobjective（ifalsonosetthen）
            if 'objective' not in self.params or self.params['objective'] == 'binary':
                # ifobjectivealsoyesinitialvalues，according toNumber of classesadjust
                if self._n_classes == 2:
                    self.params['objective'] = 'binary'
                else:
                    self.params['objective'] = 'classification'

        # adjustbatch_sizeparameterswithmatchactualbatchsize
        self.params['batch_size'] = len(X)

        # ⚠️ Critical fix：willNumber of classesaddtoparamsin
        # GradTreeneedknowoutputlayershouldhavehow manyneuron
        # ifnotprovidethis parameter，GradTreewillfromtrainingdataininfer，maycauseNumber of classesnotaccurate
        if self._n_classes is not None:
            self.params['n_classes'] = self._n_classes

        if self._is_first_batch:
            # =============================================================
            # firstbatch：initializeandtrainingmodel
            # =============================================================
            print(f"[GradTree] initializemodel: feature count={current_n_features}, "
                  f"Number of classes={self._n_classes}, "
                  f"reservedratio={self.params['max_features_reserve_ratio']}")

            # createmodelinstance
            self.model = GradTree(params=self.params, args={})

            # firsttimestraining
            # Note：useuseskip_preprocessing=True，becauseIsassumptiondataalreadyalreadystandardization
            self.model.fit(
                X_train=X,
                y_train=y,
                rebuild_model=False,  # firsttimesnotneedrebuild
                skip_preprocessing=True,  # Skip preprocessing
                verbose=self.params['verbose']
            )

            self._is_first_batch = False
            print(f"[GradTree] modelinitializedone")

        else:
            # =============================================================
            # subsequentbatch：continuetraining（keepoptimizerstatus）
            # =============================================================

            # detectionfeature spaceyesnoexpansion
            if hasattr(self.model, '_max_reserved_features'):
                max_reserved = self.model._max_reserved_features
                if current_n_features > max_reserved:
                    self.feature_expansion_count += 1
                    print(f"[GradTree] detectiontofeature spaceexpansion: "
                          f"{max_reserved} -> {current_n_features}")

            # continuetraining
            # critical：rebuild_model=False ensuredo not recreateoptimizer
            self.model.fit(
                X_train=X,
                y_train=y,
                rebuild_model=False,  # keepoptimizerstatus
                skip_preprocessing=True,
                verbose=self.params['verbose']
            )

        self.n_batches_processed += 1

    def get_stats(self) -> dict:
        """Returnslearnerstatisticsinfo"""
        stats = super().get_stats()
        stats.update({
            'n_batches_processed': self.n_batches_processed,
            'n_features_seen': self._n_features_seen,
            'n_classes': self._n_classes,
            'feature_expansion_count': self.feature_expansion_count,
            'params': self.params.copy()
        })

        # ifmodelalreadyinitialize，addmoremanyinfo
        if self.model is not None:
            if hasattr(self.model, '_expansion_history'):
                stats['expansion_history'] = self.model._expansion_history
            if hasattr(self.model, '_current_active_features'):
                stats['current_active_features'] = self.model._current_active_features
            if hasattr(self.model, '_max_reserved_features'):
                stats['max_reserved_features'] = self.model._max_reserved_features

        return stats

    def reset(self):
        """resetlearnertoinitialstatus"""
        super().reset()
        self.model = None
        self._is_first_batch = True
        self._n_features_seen = 0
        self._n_classes = None
        self.n_batches_processed = 0
        self.feature_expansion_count = 0


# ============================================================================
# convenientfunction：createdifferentconfigGradTreelearner
# ============================================================================

def create_gradtree_small(**kwargs):
    """
    createsmalltypeGradTree（fasttraining，suitabledebug）

    Note：n_estimators=1becausewhencurrentexperimentnot useuseensemblepattern。
    ensemblefunctioncanyesoriginalasortoGRANDEreserved，butingwhencurrentingonline learningexperimentin
    Isonlyuseusesingle decisiontree。
    """
    defaults = {
        'depth': 4,
        'n_estimators': 1,  # single treepattern（not useuseensemble）
        'learning_rate_values': 0.1,
        'max_features_reserve_ratio': 2.0,
    }
    defaults.update(kwargs)
    return GradTreeLearner(**defaults)


def create_gradtree_medium(**kwargs):
    """
    createintypeGradTree（balanceperformanceandspeed）

    ⚠️ alreadyupdatetoGrid Searchoptimizeafterparameters（2024-12-10）
    baseatCovertypeDataset100timesparameterarraycombinetest，findtobestconfig：
    - learning_rate: 0.001
    - epochs: 5
    - depth: 6
    - reserve_ratio: 1.5
    - objective: 'classification'（multi-class classificationtask）
    - optimizer: 'SWA'（stableingonline learningoptimizer）

    ⚠️ Note：batch_sizeshouldbycallmethodfromdataset_configpropagateenter，
    rather thanyeshardencodingingthisin。thisguaranteeexperimentconfigflexibility。

    thisconfiging7classificationtaskonreachtoCER=0.55（accuracy45%），
    totraditional decisiontree（CER=0.51）gap shrinksmallto3-4hundreddividepoint。

    Note：n_estimators=1becausewhencurrentexperimentnot useuseensemblepattern。
    """
    defaults = {
        # treestructureparameters
        'depth': 6,  # Grid Searchbest：6
        'n_estimators': 1,  # single treepattern

        # learning rateparameters（havelayeruseusesamelearning rate）
        'learning_rate_index': 0.001,  # Grid Searchbest：0.001
        'learning_rate_values': 0.001,  # Grid Searchbest：0.001
        'learning_rate_leaf': 0.001,  # Grid Searchbest：0.001

        # feature spaceparameters
        'max_features_reserve_ratio': 1.5,  # Grid Searchbest：1.5

        # trainingcontrolparameters
        'epochs': 5,  # Grid Searchbest：5

        # tasktypeparameters（important！）
        'objective': 'classification',  # ✅ multi-class classificationtask，toGrid Searchconsistent

        # optimizerparameters
        'optimizer': 'SWA',  # ✅ toGrid Searchconsistent

        # otherparameters
        'random_seed': 42,  # ✅ ensurecanreproducibility
        'verbose': 0,  # ✅ quietpattern

        # Note：batch_sizenotingthisinset，shouldbycallmethod propagationenter
    }
    defaults.update(kwargs)
    return GradTreeLearner(**defaults)


def create_gradtree_original_medium(**kwargs):
    """
    createintypeGradTree（originalbeginconfig，retainuseatcontrast）

    thisyesGrid Searchoptimizebeforeconfig，retainastobaselinecontrast。
    ingCovertypeDataseton，thisconfigCERaboutto0.72（accuracy28%）。

    ifyouwantcontrastoptimizebefore and afterperformance，canuseusethisfunction。
    """
    defaults = {
        'depth': 5,
        'n_estimators': 1,
        'learning_rate_index': 0.05,
        'learning_rate_values': 0.05,
        'learning_rate_leaf': 0.05,
        'max_features_reserve_ratio': 2.0,
        'epochs': 1,
    }
    defaults.update(kwargs)
    return GradTreeLearner(**defaults)


def create_gradtree_large(**kwargs):
    """
    createlargetypeGradTree（bestperformance，slower）

    thisconfiguseusemoredeeptreeandmoresmalllearning rate，canobtainbetterperformance，
    buttrainingtimealsowillmorelong。suitableforperformancerequirementhigh、timesufficientscenario。

    Note：n_estimators=1becausewhencurrentexperimentnot useuseensemblepattern。
    """
    defaults = {
        'depth': 6,
        'n_estimators': 1,  # single treepattern（not useuseensemble）
        'learning_rate_index': 0.05,
        'learning_rate_values': 0.05,
        'learning_rate_leaf': 0.05,
        'max_features_reserve_ratio': 2.0,
    }
    defaults.update(kwargs)
    return GradTreeLearner(**defaults)


def create_gradtree_for_vfs(**kwargs):
    """
    createspecializedtoVFSoptimizeGradTree

    VFS（Variable Feature Space）scenarioneedlargerreserved spaceshouldforfeaturegrowth。
    thisconfiguseuse5.0reserve_ratio，ensurehaveenoughemptybetweenaccommodate newfeature。

    Note：n_estimators=1becausewhencurrentexperimentnot useuseensemblepattern。
    """
    defaults = {
        'depth': 5,
        'n_estimators': 1,  # single treepattern（not useuseensemble）
        'learning_rate_index': 0.05,
        'learning_rate_values': 0.05,
        'learning_rate_leaf': 0.05,
        'max_features_reserve_ratio': 5.0,  # largerreserved spaceuseatfeaturegrowth
        'optimizer': 'SWA',  # SWAforingonline learningmorefriendly
    }
    defaults.update(kwargs)
    return GradTreeLearner(**defaults)


def create_gradtree_for_vfs_custom(
        batch_size=100,
        max_features_reserve_ratio=5.0,
        epochs=3,
        depth=6,
        learning_rate_index=0.01,
        learning_rate_values=0.01,
        learning_rate_leaf=0.01,
        optimizer='SWA',
        **kwargs
):
    """
    createcustomparametersVFSversionGradTree（useatGrid Search）

    tocreate_gradtree_for_vfs()difference：
    - create_gradtree_for_vfs(): fixeddefaultparameters
    - create_gradtree_for_vfs_custom(): accept anycustomparameters

    Note：notneedn_classesparameters！
    becauseIsalreadyalreadyuseensure_first_batch_has_all_classesensure
    firstbatchcontainhaveclasses，Number of classesinferwillsuccess。
    """
    params = {
        'depth': depth,
        'n_estimators': 1,  # VFSexperimentusesingle tree
        'learning_rate_index': learning_rate_index,
        'learning_rate_values': learning_rate_values,
        'learning_rate_leaf': learning_rate_leaf,
        'max_features_reserve_ratio': max_features_reserve_ratio,
        'optimizer': optimizer,
        'epochs': epochs,
        'batch_size': batch_size,
        'verbose': 0,  # Grid Searchwhennotoutputdetailed info
    }
    params.update(kwargs)
    return GradTreeLearner(**params)


if __name__ == "__main__":
    # simpletest
    print("GradTreeadaptdevicemoduleloadsuccess！")

    if GradTree is not None:
        print("\navailabledefaultconfig：")
        print("1. create_gradtree_small()  - smalltype（fast）")
        print("2. create_gradtree_medium() - intype（balance）")
        print("3. create_gradtree_large()  - largetype（precise）")
        print("4. create_gradtree_for_vfs() - VFSoptimize")
    else:
        print("\nwarn: GradTreenot yetsuccessimport")
        print("pleaseensureDynaGradTree.pyingcorrectposition")