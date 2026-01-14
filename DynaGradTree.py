import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import sklearn
from copy import deepcopy
import category_encoders as ce
import pandas as pd
import math
import time  # 【Step 2 Added】useatrecordexpansion timestamp
from focal_loss import SparseCategoricalFocalLoss


# Phase completedPhase 1：reserved spacedynamic featuresprocessing
# 【steptwo】：Now implementingdynamicexpansion functionality

class GradTree(tf.keras.Model):
    def __init__(self, params, args):
        # ========================================================================
        # 【criticalfixcurrently】First callparent class initialization - Kerasrequires this to befirst step！
        # ========================================================================
        super(GradTree, self).__init__()

        # Now can safelyset ourcustom attributes
        params.update(args)
        self.config = None

        # Added: Trackmodel buildingstatus
        self._model_built = False
        # Added: Trackpreprocessorstatus
        self._preprocessors_fitted = False

        # 【Added】Trackoptimizercompilation status
        self._optimizer_compiled = False  # ← addthisline

        # 【modify】dynamic feature spacestate tracking
        self._current_active_features = 0  # currently activefeature count
        self._max_reserved_features = 0  # whencurrentreservedmaximumfeaturecapacity

        # 【Added】fromparamsreadreserved spaceconfig，with defaults
        # max_features_reserve_ratio: reserved spacemultiplication factor（e.g.2.0meansreserved2x space）
        self._feature_expansion_factor = params.get('max_features_reserve_ratio', 2.0)

        # min_features_reserve: optional，minimumreservedfeature count
        # If user providesthis parameter，will be used in_initialize_feature_space
        # If not provided，use fixed increment strategy（whencurrentfeature count + buffer）
        self._min_features_reserve = params.get('min_features_reserve', None)

        # Internal use：fixed incrementbuffer（whenreservedratiostrategy not applicable）
        self._feature_expansion_buffer = 1000  # Kept as fallback strategy Can set largenumber to remove limits

        # ========================================================================
        # 【Step 2 Added - modify1】expansionhistory tracking
        # ========================================================================
        # These variablestrackmodelexpansion history，for futureanalysis and debugging
        self._expansion_history = []  # Record eachexpansiondetailed info（listeach element isdictionary）
        self._last_expansion_step = 0  # recordlastexpansionoccurrencetraining step（currently unused，but interface reserved）

        self.set_params(**params)

    @tf.function(jit_compile=True)
    def call(self, inputs, training):
        output = self.output_layer(inputs)

        if self.objective == 'regression' or self.objective == 'binary':
            result = tf.einsum('be->b', output)
        else:
            result = tf.einsum('bec->bc', output)

        if self.objective == 'regression' or self.objective == 'binary':
            result = tf.expand_dims(result, 1)

        return result

    def fit(self,
            X_train,
            y_train,
            X_val=None,
            y_val=None,
            rebuild_model=False,
            skip_preprocessing=False,
            **kwargs):

        # =================================================================
        # Data preprocessingbranch logic（three modes）
        # =================================================================

        if skip_preprocessing:
            # ============= Mode 1: Skip preprocessing =============

            # 【Fix】Properly handlemisalignedlist data
            if not isinstance(X_train, pd.DataFrame):
                # Check ifmisalignedlist
                if isinstance(X_train, list):
                    # Pass directly aslistPass，let subsequent_preprocess_dynamic_featuresprocessingalignment
                    # Do not convert toDataFrame，becausepandascannot properly handlemisalignedlist
                    X_train_processed = X_train  # keeplistformat
                else:
                    # Ifnumpyarray or other format，convert toDataFramethen get values
                    X_train = pd.DataFrame(X_train)
                    X_train_processed = X_train.values.astype(np.float64)
            else:
                # AlreadyDataFrame，get values directly
                X_train_processed = X_train.values.astype(np.float64)

            # Apply same to validation setprocessing
            if X_val is not None:
                if not isinstance(X_val, pd.DataFrame):
                    if isinstance(X_val, list):
                        X_val_processed = X_val  # keeplistformat
                    else:
                        X_val = pd.DataFrame(X_val)
                        X_val_processed = X_val.values.astype(np.float64)
                else:
                    X_val_processed = X_val.values.astype(np.float64)
            else:
                X_val_processed = None

            # Set necessaryattributes（skipcomplexpreprocessorfitting）
            # Note：we do not know yetfinalfeature count，becauselistmaymisaligned
            # so do not set yetnumber_of_variables，let_preprocess_dynamic_featuresto handle
            if isinstance(X_train_processed, list):
                # listformat，do not set yetnumber_of_variables
                pass
            else:
                # arrayformat，can directly getfeature count
                self.number_of_variables = X_train_processed.shape[1]

            # Handle target variablestatistics
            self.mean = np.mean(y_train)
            self.std = np.std(y_train)

            # Handle class information
            if self.objective == 'classification' or self.objective == 'binary':
                # ⚠️ Critical fix：Prioritizeparamsinn_classes
                if hasattr(self, 'n_classes') and self.n_classes is not None:
                    self.number_of_classes = self.n_classes
                    #print(f"   ✓ Using pre-specified n_classes={self.n_classes}")
                else:
                    self.number_of_classes = len(np.unique(y_train))
                    print(f"   ⚠️ Inferring n_classes from data: {self.number_of_classes}")

                # ⚠️ Smart weight calculation：processingmissing classes
                if hasattr(self, 'n_classes') and self.n_classes is not None:
                    # Only for existingclassescalculate weights
                    existing_classes = np.unique(y_train)
                    existing_weights = sklearn.utils.class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=existing_classes,
                        y=y_train
                    )
                    # Create completearray（length equalsnumber_of_classes），default weight1.0
                    self.class_weights = np.ones(self.number_of_classes, dtype=np.float64)
                    # Fill calculatedweights to positions
                    for i, cls in enumerate(existing_classes):
                        self.class_weights[int(cls)] = existing_weights[i]
                else:
                    # Standard process：calculate directly
                    self.class_weights = sklearn.utils.class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(y_train),
                        y=y_train
                    )

                self.class_weight_dict = {}
                for i in range(self.number_of_classes):
                    self.class_weight_dict[i] = self.class_weights[i]
            else:
                self.number_of_classes = 1
                self.class_weights = np.ones_like(np.unique(y_train))
                self.class_weight_dict = None

        elif hasattr(self, '_preprocessors_fitted') and self._preprocessors_fitted and not rebuild_model:
            # ============= Mode 2: Reuse fittedpreprocessor =============
            print("🔄 Reusing fitted preprocessors for incremental training")

            # Data format conversion
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            if X_val is not None and not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)

            # useusefittedpreprocessortransform（do not refit）
            X_train = self.encoder_loo.transform(X_train)
            X_val = self.encoder_loo.transform(X_val) if X_val is not None else None

            X_train = self.encoder_ohe.transform(X_train)
            X_val = self.encoder_ohe.transform(X_val) if X_val is not None else None

            X_train = X_train.fillna(self.median_train)
            X_val = X_val.fillna(self.median_train) if X_val is not None else None

            X_train_processed = self.normalizer.transform(X_train.values.astype(np.float64))
            X_val_processed = self.normalizer.transform(
                X_val.values.astype(np.float64)) if X_val is not None else None

            # These statistics not updated，use original values
            # self.mean, self.std, self.number_of_classesetc. remain unchanged

        else:
            # ============= Mode 3: completepreprocessingprocedure =============
            print("🔧 Fitting new preprocessors for initial training")

            # Data format conversion
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            if X_val is not None and not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)

            # Detect numerical andclassescolumns
            num_columns = list(X_train.select_dtypes(include=[np.number]).columns.values)
            if len(num_columns) == 0:
                cat_columns = list(X_train.columns.values)
            else:
                cat_columns = list(X_train.select_dtypes(exclude=[np.number]).columns.values)

            # Categorical feature encoding
            low_card_threshold = math.floor(0.025 * X_train.shape[0])
            high_cardinality_indices = []
            low_cardinality_indices = []
            for col_index, col in enumerate(cat_columns):
                if X_train[col].nunique() > low_card_threshold:
                    high_cardinality_indices.append(col_index)
                else:
                    low_cardinality_indices.append(col_index)

            self.encoder_loo = ce.LeaveOneOutEncoder(cols=high_cardinality_indices, sigma=0.05)
            if self.objective == 'regression':
                self.mean = np.mean(y_train)
                self.std = np.std(y_train)
                self.encoder_loo.fit(X_train, (y_train - self.mean) / self.std)
            else:
                self.encoder_loo.fit(X_train, y_train)
            X_train = self.encoder_loo.transform(X_train)
            if X_val is not None:
                X_val = self.encoder_loo.transform(X_val)

            self.encoder_ohe = ce.OneHotEncoder(cols=low_cardinality_indices)
            self.encoder_ohe.fit(X_train)
            X_train = self.encoder_ohe.transform(X_train)
            if X_val is not None:
                X_val = self.encoder_ohe.transform(X_val)

            self.median_train = X_train.median(axis=0)
            X_train = X_train.fillna(self.median_train)
            if X_val is not None:
                X_val = X_val.fillna(self.median_train)

            self.cat_columns_preprocessed = []
            for column, column_index in enumerate(X_train.columns):
                if column not in num_columns:
                    self.cat_columns_preprocessed.append(column_index)

            quantile_noise = 1e-4
            quantile_train = np.copy(X_train.values).astype(np.float64)
            np.random.seed(42)
            stds = np.std(quantile_train, axis=0, keepdims=True)
            noise_std = quantile_noise / np.maximum(stds, quantile_noise)
            quantile_train += noise_std * np.random.randn(*quantile_train.shape)

            quantile_train = pd.DataFrame(quantile_train, columns=X_train.columns, index=X_train.index)

            self.normalizer = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=min(quantile_train.shape[0], 1000),
                output_distribution='normal',
            )

            self.normalizer.fit(quantile_train.values.astype(np.float64))
            X_train_processed = self.normalizer.transform(X_train.values.astype(np.float64))
            X_val_processed = self.normalizer.transform(X_val.values.astype(np.float64)) if X_val is not None else None

            self.mean = np.mean(y_train)
            self.std = np.std(y_train)

            self.number_of_variables = X_train_processed.shape[1]
            if self.objective == 'classification' or self.objective == 'binary':
                # ⚠️ Critical fix：Prioritizeparamsinn_classes
                if hasattr(self, 'n_classes') and self.n_classes is not None:
                    self.number_of_classes = self.n_classes
                    #print(f"   ✓ Using pre-specified n_classes={self.n_classes}")
                else:
                    self.number_of_classes = len(np.unique(y_train))
                    print(f"   ⚠️ Inferring n_classes from data: {self.number_of_classes}")

                # ⚠️ Smart weight calculation：processingmissing classes
                if hasattr(self, 'n_classes') and self.n_classes is not None:
                    # Only for existingclassescalculate weights
                    existing_classes = np.unique(y_train)
                    existing_weights = sklearn.utils.class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=existing_classes,
                        y=y_train
                    )
                    # Create completearray（length equalsnumber_of_classes），default weight1.0
                    self.class_weights = np.ones(self.number_of_classes, dtype=np.float64)
                    # Fill calculatedweights to positions
                    for i, cls in enumerate(existing_classes):
                        self.class_weights[int(cls)] = existing_weights[i]
                else:
                    # Standard process：calculate directly
                    self.class_weights = sklearn.utils.class_weight.compute_class_weight(
                        class_weight='balanced',
                        classes=np.unique(y_train),
                        y=y_train
                    )

                self.class_weight_dict = {}
                for i in range(self.number_of_classes):
                    self.class_weight_dict[i] = self.class_weights[i]
            else:
                self.number_of_classes = 1
                self.class_weights = np.ones_like(np.unique(y_train))
                self.class_weight_dict = None

            # Markpreprocessoras fitted
            self._preprocessors_fitted = True
            #print(f"   ✅ New preprocessors fitted: {X_train_processed.shape}")

        # =================================================================
        # 【modify】dynamic feature spacepreprocessing（Integratesalignment、detection、expansion）
        # =================================================================
        #print(f"\n🔄 Processing dynamic feature space...")

        # Preprocess trainingpreprocessing
        X_train_processed = self._preprocess_dynamic_features(X_train_processed, update_mask=True)

        # forvalidation dataperformpreprocessing（if exists）
        if X_val_processed is not None:
            # Note：Validation set does not triggermaskupdate，becausetrainingsetalready processed
            # validationsetonly needsalignmentandexpansiontototraining setdimensions
            X_val_processed = self._preprocess_dynamic_features(X_val_processed, update_mask=False)

        # updatenumber_of_variablestoreserved spacesize
        # So subsequentmodel buildingwill usecorrectdimensions
        self.number_of_variables = self._max_reserved_features

        # print(
        #     f"   🎯 Feature space ready: active={self._current_active_features}, reserved={self._max_reserved_features}")

        # =================================================================
        # Core modification：Conditionally buildmodel（original code continues）
        # =================================================================

        # =================================================================
        # model buildingand training logic（shared by all modes）
        # =================================================================

        # model buildinglogic
        if not self._model_built or rebuild_model:
            self.build_model()
            self._model_built = True
            if rebuild_model:
                self._preprocessors_fitted = False
                self._optimizer_compiled = False  # ← when rebuildresetcompilation flag
            print(f"Model built/rebuilt. Model structure initialized.")

            # 【Critical fix】Immediately triggerbuild()，Ensure weightstf.functioncreated outside
            # thisCritical for avoiding"tf.function only supports singleton tf.Variables"error
            print(f"🔧 Triggering build() to create weights...")
            dummy_input = np.zeros((1, self.number_of_variables), dtype=np.float32)
            try:
                _ = self.output_layer(dummy_input, training=False)
                print(f"   ✅ Weights created successfully")
            except Exception as e:
                print(f"   ⚠️ Build trigger encountered issue (may be okay): {e}")
        else:
            # updateconfigbut do not rebuildmodelstructure
            self.config['mean'] = self.mean
            self.config['std'] = self.std
            self.config['number_of_classes'] = self.number_of_classes
            self.config['number_of_variables'] = self.number_of_variables

        # ✅ Fix：only on first time orafter rebuildcompile
        if not self._optimizer_compiled or rebuild_model:
            print(f"🔧 Compiling model with optimizer...")
            self.compile(loss=self.loss_name, metrics=self.metrics_name,
                         mean=self.mean, std=self.std, class_weight=self.class_weights)
            self._optimizer_compiled = True
            print(f"   ✅ Optimizer initialized and compiled")
        # else:
        #     print(f"🔄 Reusing existing optimizer (preserving state)")

        # Create training dataset
        train_data = tf.data.Dataset.from_tensor_slices(
            (tf.dtypes.cast(tf.convert_to_tensor(X_train_processed), tf.float32),
             tf.dtypes.cast(tf.convert_to_tensor(y_train), tf.float32)))

        train_data = (train_data
                      .shuffle(32_768)
                      .cache()
                      .batch(batch_size=self.batch_size, drop_remainder=False)
                      .prefetch(tf.data.AUTOTUNE)
                      )

        # createvalidation dataset
        validation_data = None
        if X_val_processed is not None and y_val is not None:
            validation_data = tf.data.Dataset.from_tensor_slices(
                (tf.dtypes.cast(tf.convert_to_tensor(X_val_processed), tf.float32),
                 tf.dtypes.cast(tf.convert_to_tensor(y_val), tf.float32)))

            validation_data = (validation_data
                               .cache()
                               .batch(batch_size=self.batch_size, drop_remainder=False)
                               .prefetch(tf.data.AUTOTUNE)
                               )

            monitor = 'val_loss'
        else:
            monitor = 'loss'

        # Callback setup
        if 'callbacks' not in kwargs.keys():
            callbacks = []

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                          patience=self.early_stopping_epochs,
                                                          min_delta=1e-3,
                                                          restore_best_weights=True)
        callbacks.append(early_stopping)

        if 'reduce_lr' in kwargs.keys() and kwargs['reduce_lr']:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=self.early_stopping_epochs // 3)
            callbacks.append(reduce_lr)

        # Start training
        # ingfitmethod，callsuper().fit()before
        # fromkwargsinExtract potentially conflictingparameters，Prioritizeuser-providedvalues
        epochs_to_use = kwargs.pop('epochs', self.epochs)  # popremoves and returnsvalues
        verbose_to_use = kwargs.pop('verbose', self.verbose)

        # Start training
        history = super(GradTree, self).fit(train_data,
                                            validation_data=validation_data,
                                            epochs=epochs_to_use,  # useuseextractedvalues
                                            callbacks=callbacks,
                                            class_weight=self.class_weight_dict,
                                            verbose=verbose_to_use,  # useuseextractedvalues
                                            **kwargs)  # Nowkwargsno longer containsepochs

        return history

    def build_model(self):
        self.config['mean'] = self.mean
        self.config['std'] = self.std
        self.config['number_of_classes'] = self.number_of_classes
        self.config['number_of_variables'] = self.number_of_variables

        # 【Added】Passdynamic feature spaceinfo toGradTreeBlock
        self.config['active_features'] = self._current_active_features
        self.config['max_features'] = self._max_reserved_features

        # 【Step 2 Added】checkif need to use migratedparams for initialization
        if hasattr(self, '_use_migrated_params') and self._use_migrated_params:
            # Expansion scenario：use migratedparams for initialization
            self.config['use_migrated_params'] = True
            self.config['migrated_params'] = self._migrated_params
        else:
            # Normal scenario：useuseregular initialization
            self.config['use_migrated_params'] = False

        self.output_layer = GradTreeBlock(**self.config)

    def compile(self,
                loss,
                metrics,
                **kwargs):

        if self.objective == 'classification':

            if loss == 'crossentropy':
                if not self.focal_loss:
                    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=self.from_logits)  # tf.keras.losses.get('categorical_crossentropy')
                else:
                    loss_function = SparseCategoricalFocalLoss(gamma=2, class_weight=self.class_weights,
                                                               from_logits=self.from_logits)
            else:
                loss_function = tf.keras.losses.get(loss)

            if metrics == ['Accuracy']:
                metrics_function = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
            elif metrics == ['F1']:
                metrics_function = [tfa.metrics.F1Score(num_classes=self.number_of_classes, average='macro')]
            else:
                metrics_function = []

        elif self.objective == 'binary':

            if loss == 'crossentropy':
                loss_function = tf.keras.losses.BinaryCrossentropy(
                    from_logits=self.from_logits)  # tf.keras.losses.get('binary_crossentropy')
            elif loss == 'focal':
                loss_function = tfa.losses.SigmoidFocalCrossEntropy(from_logits=self.from_logits)
            else:
                loss_function = tf.keras.losses.get(loss)

            if metrics == ['Accuracy']:
                metrics_function = [tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5)]
            elif metrics == ['F1']:
                metrics_function = [tfa.metrics.F1Score(num_classes=1, average='macro', threshold=0.5)]
            else:
                metrics_function = []

        elif self.objective == 'regression':
            loss_function = tf.keras.losses.get(loss)

            if metrics == ['R2']:
                metrics_function = [tfa.metrics.r_square.RSquare(y_shape=(1,))]
            else:
                metrics_function = []

        super(GradTree, self).compile(optimizer=self.get_optimizer(), loss=loss_function, metrics=metrics_function)

    def predict(self, X_test, **kwargs):
        """
        【modify】predictionmethod：alsoneedsthroughdynamic feature spacepreprocessing
        """
        # preprocessingprocedure
        if hasattr(self, '_preprocessors_fitted') and self._preprocessors_fitted:
            # Mode 2：useuseas fittedpreprocessor
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test)

            X_test = self.encoder_loo.transform(X_test)
            X_test = self.encoder_ohe.transform(X_test)
            X_test = X_test.fillna(self.median_train)
            X_test_processed = self.normalizer.transform(X_test.values.astype(np.float64))

        else:
            # Mode 1：Skip preprocessing（assumptiondataalreadypreprocessing）
            if not isinstance(X_test, pd.DataFrame):
                if isinstance(X_test, list):
                    X_test_processed = X_test  # keeplistformat
                else:
                    X_test = pd.DataFrame(X_test)
                    X_test_processed = X_test.values.astype(np.float64)
            else:
                X_test_processed = X_test.values.astype(np.float64)

        # 【critical】dynamic feature spacepreprocessing
        # Note：thisinupdate_maskchooseyesdesign decision
        # Option A：update_mask=True  -> predictionwhenalsowillactivenewfeature（more aggressive）
        # Option B：update_mask=False -> predictionwhen notactivenewfeature（more conservative）
        # currentlyadoptuseOption A，totrainingkeepconsistent
        X_test_processed = self._preprocess_dynamic_features(X_test_processed, update_mask=True)

        # callparentclasspredict
        predictions = super(GradTree, self).predict(X_test_processed, **kwargs)

        return predictions

    def set_params(self, **kwargs):

        if self.config is None:
            self.config = {
                'objective': 'binary',

                'depth': 6,
                'n_estimators': 1,

                'learning_rate_index': 0.01,
                'learning_rate_values': 0.01,
                'learning_rate_leaf': 0.01,
                'temperature': 0.0,

                'optimizer': 'SWA',
                'cosine_decay_steps': 0,

                'initializer': 'RandomNormal',

                'from_logits': True,
                'apply_class_balancing': True,

                'metrics': ['F1'],  # F1, Accuracy, R2
                'random_seed': 42,
                'verbose': 0,
                'batch_size': 64,
                'epochs': 1,
                'early_stopping_epochs': 10,
            }

        self.config.update(kwargs)

        if 'loss' not in self.config.keys():
            if self.config['objective'] == 'classification' or self.config['objective'] == 'binary':
                self.config['loss'] = 'crossentropy'
                self.config['focal_loss'] = False
            elif self.config['objective'] == 'regression':
                self.config['loss'] = 'mse'
                self.config['focal_loss'] = False

        self.config['optimizer_name'] = self.config.pop('optimizer')
        self.config['loss_name'] = self.config.pop('loss')
        if 'metrics' in self.config.keys():
            self.config['metrics_name'] = self.config.pop('metrics')
        else:
            self.config['metrics_name'] = []
        for arg_key, arg_value in self.config.items():
            setattr(self, arg_key, arg_value)

        self.leaf_trainable = True
        self.split_index_trainable = True
        self.split_values_trainable = True
        self.weights_trainable = True

        tf.keras.utils.set_random_seed(self.random_seed)

    def get_params(self):
        return self.config

    def reset_model(self):
        """manuallyresetmodelstructure（ifneedsfrombeginningstart）"""
        self._model_built = False
        self._preprocessors_fitted = False
        self._optimizer_compiled = False  # ← addthisline
        if hasattr(self, 'output_layer'):
            self.output_layer = None
        print("Model reset. Next fit() will rebuild from scratch.")

    def _initialize_feature_space(self, initial_feature_count):
        """
        firsttimesinitializefeature space

        strategy：reservedextraemptybetweento supportnot yetfeaturegrowth

        reservedstrategy：
        1. ifuserspecifymin_features_reserve，useusemax(ratiostrategy, absoluteforminimumvalues)
        2. nothen，useusemin(ratiostrategy, fixed incrementstrategy)

        itsin：
        - ratiostrategy: whencurrentfeature count × max_features_reserve_ratio
        - fixed incrementstrategy: whencurrentfeature count + buffer (20)
        - absoluteforminimumvalues: min_features_reserve
        """
        self._current_active_features = initial_feature_count

        # computeratiostrategyreservedamount
        reserved_by_factor = int(initial_feature_count * self._feature_expansion_factor)

        if self._min_features_reserve is not None:
            # strategy1：userspecifyminimumreservedcount
            # takeratiostrategyandabsoluteforminimumvaluesincomparelargethat
            self._max_reserved_features = max(reserved_by_factor, self._min_features_reserve)
            print(f"🔧 Dynamic Feature Space Initialized (with min_features_reserve):")
            print(f"   - Reserve ratio: {self._feature_expansion_factor}x")
            print(f"   - Min reserve: {self._min_features_reserve}")
        else:
            # strategy2：useusetraditionalminstrategy（ratio vs fixed increment）
            reserved_by_buffer = initial_feature_count + self._feature_expansion_buffer
            self._max_reserved_features = min(reserved_by_factor, reserved_by_buffer)
            print(f"🔧 Dynamic Feature Space Initialized:")
            print(f"   - Reserve ratio: {self._feature_expansion_factor}x")

        # ensureat leastreservedsomeemptybetween（safecheck）
        if self._max_reserved_features <= initial_feature_count:
            self._max_reserved_features = initial_feature_count + 10
            print(f"   ⚠️ Adjusted: reserved capacity too small, increased to {self._max_reserved_features}")

        print(f"   - Active features: {self._current_active_features}")
        print(f"   - Reserved capacity: {self._max_reserved_features}")
        print(f"   - Available growth: {self._max_reserved_features - self._current_active_features}")

    def _handle_new_features(self, new_feature_count):
        """
        processingnewfeatureadd

        ifnewfeature countingreservedrangeinternal：simpleactive（removemask）
        ifexceedreservedrange：【steptwomodify6】reportwrongwhilenotyesdirectlyraise
        """
        if new_feature_count <= self._max_reserved_features:
            # ingreservedrangeinternal，activenewfeature
            added_count = new_feature_count - self._current_active_features
            print(f"✅ Activating {added_count} new features:")
            print(f"   - Previous: {self._current_active_features}")
            print(f"   - Current: {new_feature_count}")
            print(f"   - Remaining capacity: {self._max_reserved_features - new_feature_count}")

            # updateactivefeature count
            old_active = self._current_active_features
            self._current_active_features = new_feature_count

            # updateGradTreeBlockinmask
            if hasattr(self, 'output_layer') and self.output_layer is not None:
                self.output_layer.update_active_features(new_feature_count)
                print(f"   - Mask updated in GradTreeBlock")

        else:
            # ========================================================================
            # 【steptwomodify6】thisbranchnotshouldbeexecute
            # ifexecutetothisin，instructionabovelayerexpansionlogichaveproblemproblem
            # ========================================================================
            raise RuntimeError(
                f"INTERNAL ERROR: Feature count {new_feature_count} exceeds "
                f"reserved capacity {self._max_reserved_features}, but expansion "
                f"was not triggered. This indicates a bug in _preprocess_dynamic_features. "
                f"Please report this error."
            )

    def get_feature_space_info(self):
        """getwhencurrentfeature spacestatusinfo"""
        if self._model_built:
            return {
                'active_features': self._current_active_features,
                'reserved_features': self._max_reserved_features,
                'available_capacity': self._max_reserved_features - self._current_active_features,
                'utilization': f"{self._current_active_features / self._max_reserved_features * 100:.1f}%"
            }
        else:
            return {'status': 'Model not built yet'}

    def _align_batch_features(self, X):
        """
        willbatchinternaldataalignmenttobatchinternalmaximumfeature count

        thisfunctionprocessingtwo typescase：
        1. inputyesnotrulelist（eachsamplefeature countdifferent）
        2. inputAlreadyrulenumpyarray（havesamplefeature countsame）

        foratcase1，functionwillfindtomostlongsample，thenwillhavesamplefilltothislength。
        foratcase2，functionsimpleReturnsoriginalarrayanditfeaturedimensions。

        parameters:
            X: inputdata，canyeslist of lists、list of arrays，ornumpy array

        Returns:
            (X_aligned, max_features_in_batch)
            X_aligned: numpyarray，shapeto[n_samples, max_features_in_batch]
            max_features_in_batch: integer，batchinternalmaximumfeature count
        """
        # first step：convert tonumpy array（ifalsonotyesspeech）
        if not isinstance(X, np.ndarray):
            # inputmayyeslist of listsorclasssimilarstructure
            # firstattemptconvert toobjecttypearray，thislikecancontainacceptdifferentlengthsubarray
            try:
                X = np.array(X, dtype=object)
            except:
                # ifconvertfail，attemptdirectlywhenasfloatarray
                X = np.array(X, dtype=np.float64)

        # twostep：detectionyesnoneedsalignment
        if X.ndim == 1 or X.dtype == object:
            # case1：notrulearray，needsalignment
            # computeeachsamplefeature count
            feature_counts = [len(sample) if hasattr(sample, '__len__') else 1 for sample in X]
            max_features_in_batch = max(feature_counts)

            # createalignmentafterarray（use0fill）
            n_samples = len(X)
            X_aligned = np.zeros((n_samples, max_features_in_batch), dtype=np.float64)

            # gradualsamplecopy
            for i, sample in enumerate(X):
                if hasattr(sample, '__len__'):
                    # ensuresampleyesnumpyarray
                    sample_array = np.array(sample, dtype=np.float64)
                    X_aligned[i, :len(sample_array)] = sample_array
                else:
                    # singlescalarvalues
                    X_aligned[i, 0] = float(sample)

            return X_aligned, max_features_in_batch
        else:
            # case2：Alreadyrulenumpyarray
            max_features_in_batch = X.shape[1] if X.ndim > 1 else 1
            return X, max_features_in_batch

    def _expand_to_reserved_space(self, X_aligned, current_feature_count):
        """
        willalignmentafterdataexpansiontoreserved spacedimensions

        input:
            X_aligned: [n_samples, current_feature_count]
            current_feature_count: whencurrentfeature count（batchinternalmaximumvalues）

        output:
            X_expanded: [n_samples, max_reserved_features]
                        aftercolumnsuse0fill

        thisfunctionensuredatafeaturedimensionstomodelparametersmatrixdimensionsconsistent
        """
        if current_feature_count < self._max_reserved_features:
            # needsexpansion
            padding_size = self._max_reserved_features - current_feature_count
            X_expanded = np.pad(X_aligned,
                                ((0, 0), (0, padding_size)),  # onlyingcolumnsdimensionsendfill
                                mode='constant',
                                constant_values=0)
        else:
            # Alreadyreserved spacesize（orlarger，butthisnotshouldoccur）
            X_expanded = X_aligned

        return X_expanded

    def _preprocess_dynamic_features(self, X, update_mask=True):
        """
        unifieddynamic featurespreprocessingentry

        thisfunctionIntegratesprocessingdynamic feature spaceneedhavestep：
        1. alignmentbatchinternalfeaturedimensions
        2. detectionnewfeatureandupdatemask（ifneeds）
        3. expansiondatatoreserved spacedimensions

        ========================================================================
        【steptwomodify2】ingthisfunctioninaddexpansiondetectionandprocessing
        ========================================================================

        thisfunctionbesetcomputetofit()andpredict()togethersameentry，ensuredataprocessingconsistent。

        parameters:
            X: inputdata，canyesmisalignedlistoralreadyalignmentnumpyarray
            update_mask: booleanvalues，yesnoingdetectiontonewfeaturewhenupdatemask
                        fit()trainingsetwhentoTrue
                        fit()validationsetwhentoFalse（becausetrainingsetalreadythroughprocessing）
                        predict()whentoTrue（according todesign decision，predictalsoupdatemask）

        Returns:
            processingafternumpyarray，shapeto[n_samples, max_reserved_features]
        """
        # step1：alignmentbatchinternal
        X_aligned, max_features_in_batch = self._align_batch_features(X)

        #print(f"   📊 Batch features aligned: detected max {max_features_in_batch} features in this batch")

        # ========================================================================
        # step2：【steptwomodify2】detectionnewfeatureanddecidehow toprocessing
        # ========================================================================
        if update_mask:
            if self._model_built and max_features_in_batch > self._current_active_features:
                # modelalreadyexist，anddetectiontonewfeature
                print(f"   🆕 New features detected: {self._current_active_features} -> {max_features_in_batch}")

                # 【Step 2 Added】checkyesnoexceedreservedcapacity
                if max_features_in_batch > self._max_reserved_features:
                    # exceedcapacity，touchsendexpansion！
                    print(f"   ⚠️ Exceeds reserved capacity ({self._max_reserved_features})")
                    print(f"   🚀 Triggering dynamic expansion...")

                    # callexpansionfunction（steptwocorecore）
                    self._expand_feature_space(max_features_in_batch, X_aligned)
                else:
                    # ingreservedrangeinternal，simpleactive（steplogic）
                    self._handle_new_features(max_features_in_batch)

            elif not self._model_built:
                # firsttimesbuild，initializefeature space
                print(f"   🎯 Initializing feature space with {max_features_in_batch} features")
                self._initialize_feature_space(max_features_in_batch)
        else:
            # notupdatemask，onlyyesrecordinfo
            if max_features_in_batch > self._current_active_features:
                print(f"   ℹ️ Batch has {max_features_in_batch} features, but mask update is disabled")
            print(f"   📌 Using current active features: {self._current_active_features}")

        # step3：expansiontoreserved space
        X_expanded = self._expand_to_reserved_space(X_aligned, max_features_in_batch)

        # print(
        #     f"   ✅ Data expanded to reserved space: [{X_aligned.shape[0]}, {max_features_in_batch}] -> [{X_expanded.shape[0]}, {self._max_reserved_features}]")

        return X_expanded

    # ============================================================================
    # 【steptwomodify3】corecoreexpansionfunction
    # ============================================================================
    def _expand_feature_space(self, required_features, X_current=None):
        """
        dynamicexpansionfeature space

        whennewfeature countexceedreservedcapacitywhencallthismethod。executebelowstep：
        1. computenewcapacity
        2. savewhencurrentmodelparameters
        3. readymigrateafterparametersmatrix（ingnumpyindonemigrate）
        4. rebuildlargermodel，directlyusemigrateafterparams for initialization
        5. updatestatusandmask

        parameters:
            required_features: needscontainacceptfeature count
            X_current: whencurrentbatchdata（optional，useatnot yetwarmupimplementation）

        Note:
            - expansionwillrebuildmodel，optimizerstatuswillbereset
            - thisyestechnologyabovemustnaturally，butbecauseexpansionfrequencyratelowwhilecanconnectaffected
            - inspired bycustominitializeapproach，steptopositiondoneparametersmigrate
        """
        print(f"\n{'=' * 70}")
        print(f"🚀 DYNAMIC EXPANSION TRIGGERED")
        print(f"{'=' * 70}")
        print(f"   Current capacity: {self._max_reserved_features}")
        print(f"   Current active: {self._current_active_features}")
        print(f"   Required: {required_features}")

        # ============================================================
        # step1：computenewcapacity
        # ============================================================
        expansion_by_ratio = int(self._max_reserved_features * 1.5)
        new_capacity = max(required_features, expansion_by_ratio)

        if new_capacity <= self._max_reserved_features:
            new_capacity = self._max_reserved_features + 10

        if self._min_features_reserve is not None:
            new_capacity = max(new_capacity, self._min_features_reserve)

        growth = new_capacity - self._max_reserved_features

        print(f"\n📐 Capacity Planning:")
        print(f"   Strategy 1 (ratio ×1.5): {expansion_by_ratio}")
        print(f"   Strategy 2 (required): {required_features}")
        print(f"   Selected new capacity: {new_capacity}")
        print(f"   Growth: +{growth} features")

        # ============================================================
        # step2：savewhencurrentmodelstatus
        # ============================================================
        print(f"\n📦 Saving current model state...")
        old_capacity = self._max_reserved_features
        old_active = self._current_active_features

        try:
            old_state = self._save_model_state()
            print(f"   ✅ Model state saved successfully")
        except Exception as e:
            print(f"   ❌ Failed to save model state: {e}")
            raise RuntimeError(f"Expansion failed at state saving: {e}")

        # ============================================================
        # step3：ingnumpyinreadymigrateafterparametersmatrix
        # ============================================================
        print(f"\n🔄 Preparing migrated parameters...")
        try:
            migrated_params = self._prepare_migrated_parameters(
                old_state, old_capacity, new_capacity, required_features
            )
            print(f"   ✅ Parameters prepared for migration")
        except Exception as e:
            print(f"   ❌ Failed to prepare parameters: {e}")
            raise RuntimeError(f"Expansion failed at parameter preparation: {e}")

        # ============================================================
        # step4：updatecapacityconfig
        # ============================================================
        print(f"\n⚙️ Updating capacity configuration...")
        self._max_reserved_features = new_capacity
        self.number_of_variables = new_capacity
        self._current_active_features = required_features
        print(f"   ✅ Capacity updated: {old_capacity} → {new_capacity}")

        # ============================================================
        # step5：rebuildmodel，use migratedafterparams for initialization
        # ============================================================
        print(f"\n🔨 Rebuilding model with migrated parameters...")
        try:
            # MarkIsneeduse migratedparams for initialization
            self._use_migrated_params = True
            self._migrated_params = migrated_params

            # rebuildmodel（willcreatenewGradTreeBlock）
            self.build_model()

            # 【Critical fix】Immediately triggerbuild()，Ensure weightstf.functioncreated outside
            # createdummy batchtouchsendGradTreeBlock.build()
            print(f"   🔧 Triggering build() with dummy forward pass...")
            dummy_input = np.zeros((1, new_capacity), dtype=np.float32)
            _ = self.output_layer(dummy_input, training=False)
            print(f"   ✅ Build triggered successfully, weights created")

            # cleanMark
            self._use_migrated_params = False
            self._migrated_params = None

            print(f"   ✅ Model rebuilt with migrated parameters")
        except Exception as e:
            print(f"   ❌ Failed to rebuild model: {e}")
            import traceback
            traceback.print_exc()
            # attemptrestoreoldcapacity
            self._max_reserved_features = old_capacity
            self.number_of_variables = old_capacity
            self._current_active_features = old_active
            raise RuntimeError(f"Expansion failed at model rebuild: {e}")

        # ============================================================
        # step6：importantnewcompile（optimizerreset）
        # ============================================================
        print(f"\n🔧 Recompiling model...")
        print(f"   ⚠️ Note: Optimizer state will be reset (this is necessary)")

        self._optimizer_compiled = False

        try:
            self.compile(
                loss=self.loss_name,
                metrics=self.metrics_name,
                mean=self.mean,
                std=self.std,
                class_weight=self.class_weights
            )
            self._optimizer_compiled = True
            print(f"   ✅ Model recompiled with new optimizer")
        except Exception as e:
            print(f"   ⚠️ Compile will be done on next fit(): {e}")

        # ============================================================
        # step7：recordexpansion history
        # ============================================================
        expansion_record = {
            'timestamp': time.time(),
            'old_capacity': old_capacity,
            'new_capacity': new_capacity,
            'old_active': old_active,
            'new_active': required_features,
            'growth': growth,
        }
        self._expansion_history.append(expansion_record)

        # ============================================================
        # done！
        # ============================================================
        print(f"\n{'=' * 70}")
        print(f"✅ EXPANSION COMPLETED SUCCESSFULLY")
        print(f"{'=' * 70}")
        print(f"   Capacity: {old_capacity} → {new_capacity} (+{growth})")
        print(f"   Active features: {old_active} → {required_features}")
        print(f"   Available room: {new_capacity - required_features}")
        print(f"   Expansion count: {len(self._expansion_history)}")
        print(f"{'=' * 70}\n")

        print(f"ℹ️  Model expansion complete. Training will continue with new capacity.")
        print(f"ℹ️  Optimizer state has been reset (expected behavior).")
        if new_capacity - required_features < 5:
            print(f"⚠️  Warning: Only {new_capacity - required_features} slots remaining.")
            print(f"   Consider increasing max_features_reserve_ratio for more headroom.")

    # ============================================================================
    # 【steptwomodify4】savemodelstatusfunction
    # ============================================================================
    def _save_model_state(self):
        """
        savewhencurrentmodelparametersstatus

        Returnsdictionary，containhaveneedsmigrateparameters。
        thistheseparameterswillingexpansionaftermigratetonew、largermodelin。

        Returns:
            dict: containmodelparametersdictionary
                - 'split_index_array': numpyarray
                - 'split_values': numpyarray
                - 'leaf_classes_array': numpyarray
                - 'active_features': currently activefeature count
                - otherelementdata
        """
        if not hasattr(self, 'output_layer') or self.output_layer is None:
            raise RuntimeError("Cannot save state: output_layer not initialized")

        # extractedparameters（convert tonumpyarray，thislikeitsthenindependentatTensorFlowcomputefigure）
        state = {
            'split_index_array': self.output_layer.split_index_array.numpy().copy(),
            'split_values': self.output_layer.split_values.numpy().copy(),
            'leaf_classes_array': self.output_layer.leaf_classes_array.numpy().copy(),

            # savemaskwhencurrentstatus（althoughwillimportantnewcreate，butrecordbelowhavehelpatvalidation）
            'feature_mask': self.output_layer.feature_mask.numpy().copy(),

            # elementdata
            'active_features': self._current_active_features,
            'max_features': self._max_reserved_features,

            # optional：saveconfigcopy
            'config': deepcopy(self.config) if hasattr(self, 'config') else None,
        }

        # printsaveparametersshape（useatdebug）
        print(f"   Parameters saved:")
        print(f"     - split_index_array: {state['split_index_array'].shape}")
        print(f"     - split_values: {state['split_values'].shape}")
        print(f"     - leaf_classes_array: {state['leaf_classes_array'].shape}")
        print(f"     - active_features: {state['active_features']}")

        return state

    # ============================================================================
    # 【Step 2 Added】readymigrateafterparametersmatrix
    # ============================================================================
    def _prepare_migrated_parameters(self, old_state, old_capacity, new_capacity, new_active_features):
        """
        ingnumpyinreadymigrateaftercompleteparametersmatrix

        thismethodinspired bycustominitializeapproach：
        - ingnumpyindonehaveparametersmigrateoperateas
        - readygoodcomplete、correctshapeparametersmatrix
        - Returnsgivebuild_model()，useatdirectlyinitializenewGradTreeBlock

        parameters:
            old_state: _save_model_state()Returnsstatusdictionary
            old_capacity: oldfeaturecapacity
            new_capacity: newfeaturecapacity
            new_active_features: newactivefeature count

        Returns:
            dict: containmigrateafterparametersmatrix
                - 'split_index_array': newshapematrix，currentold_capacitycolumnsyesoldparameters，afteryesrandominitialize
                - 'split_values': sameabove
                - 'leaf_classes_array': completecopy（tofeature countnooff）
                - 'feature_mask': newmask
        """
        old_split_index = old_state['split_index_array']
        old_split_values = old_state['split_values']
        old_leaf_classes = old_state['leaf_classes_array']

        # getshapeinfo
        n_estimators = old_split_index.shape[0]
        internal_nodes = old_split_index.shape[1]

        print(f"   Migrating parameters from capacity {old_capacity} to {new_capacity}")

        # ============================================================
        # 1. readysplit_index_arraymigratematrix
        # ============================================================
        # createnewmatrix，usesmallrandomvaluesinitialize（simulateKerasRandomNormalinitialize）
        new_split_index = np.random.randn(n_estimators, internal_nodes, new_capacity).astype(np.float32) * 0.05
        # copyoldparameterstocurrentold_capacitycolumns
        new_split_index[:, :, :old_capacity] = old_split_index

        print(f"     ✓ split_index_array prepared:")
        print(f"       Old shape: {old_split_index.shape} → New shape: {new_split_index.shape}")
        print(f"       Preserved columns: 0-{old_capacity - 1}, New columns: {old_capacity}-{new_capacity - 1}")

        # ============================================================
        # 2. readysplit_valuesmigratematrix
        # ============================================================
        new_split_values = np.random.randn(n_estimators, internal_nodes, new_capacity).astype(np.float32) * 0.05
        new_split_values[:, :, :old_capacity] = old_split_values

        print(f"     ✓ split_values prepared:")
        print(f"       Old shape: {old_split_values.shape} → New shape: {new_split_values.shape}")

        # ============================================================
        # 3. leaf_classes_arraydirectlycopy（tofeature countnooff）
        # ============================================================
        new_leaf_classes = old_leaf_classes.copy()

        print(f"     ✓ leaf_classes_array copied:")
        print(f"       Shape: {new_leaf_classes.shape} (unchanged)")

        # ============================================================
        # 4. readynewfeature mask
        # ============================================================
        new_mask = np.zeros(new_capacity, dtype=np.float32)
        new_mask[new_active_features:] = -1000  # dormantfeature

        print(f"     ✓ feature_mask created:")
        print(f"       Active: {new_active_features}, Dormant: {new_capacity - new_active_features}")

        # ============================================================
        # validationmigratecomplete
        # ============================================================
        # ensureoldparametersbecorrectretain
        if not np.allclose(new_split_index[:, :, :old_capacity], old_split_index):
            raise RuntimeError("Parameter migration verification failed: split_index not preserved")
        if not np.allclose(new_split_values[:, :, :old_capacity], old_split_values):
            raise RuntimeError("Parameter migration verification failed: split_values not preserved")

        print(f"     ✅ Verification passed: old parameters preserved")

        # Returnsmigrateafterparameters
        return {
            'split_index_array': new_split_index,
            'split_values': new_split_values,
            'leaf_classes_array': new_leaf_classes,
            'feature_mask': new_mask,
        }

    # ============================================================================
    # 【steptwomodify5】restoremodelstatusfunction
    # ============================================================================
    def _restore_model_state(self, state, old_capacity, new_capacity):
        """
        willsaveparametersrestoretonew（larger）modelin

        migratestrategy：
        - forattofeaturedimensionsmutuallyoffparameters（split_index, split_values）：
          copyoldparameterstonewmatrixcurrentold_capacitycolumns，aftercolumnskeepinitializesmallrandomvalues
        - forattofeaturenooffparameters（leaf_classes）：
          directlycompletecopy

        parameters:
            state: _save_model_state()Returnsstatusdictionary
            old_capacity: oldfeaturecapacity
            new_capacity: newfeaturecapacity
        """
        if not hasattr(self, 'output_layer') or self.output_layer is None:
            raise RuntimeError("Cannot restore state: output_layer not initialized")

        # ============================================================
        # 1. migrate split_index_array
        # ============================================================
        old_split_index = state['split_index_array']
        new_split_index = self.output_layer.split_index_array.numpy()

        # validationshapecompatible
        expected_old_shape = (old_split_index.shape[0], old_split_index.shape[1], old_capacity)
        expected_new_shape = (new_split_index.shape[0], new_split_index.shape[1], new_capacity)

        if old_split_index.shape != expected_old_shape:
            raise ValueError(f"Old split_index shape mismatch: {old_split_index.shape} vs {expected_old_shape}")

        if new_split_index.shape != expected_new_shape:
            raise ValueError(f"New split_index shape mismatch: {new_split_index.shape} vs {expected_new_shape}")

        # copyoldparameterstonewmatrixcurrentold_capacitycolumns
        new_split_index[:, :, :old_capacity] = old_split_index
        # aftercolumns（old_capacity:new_capacity）keepinitializewhensmallrandomvalues

        # assignvaluesbackTensorFlow Variable
        self.output_layer.split_index_array.assign(new_split_index)

        print(f"   ✓ split_index_array migrated:")
        print(f"     Old shape: {old_split_index.shape}")
        print(f"     New shape: {new_split_index.shape}")
        print(f"     Copied columns: 0-{old_capacity - 1}")
        print(f"     New columns: {old_capacity}-{new_capacity - 1} (kept as initialized)")

        # ============================================================
        # 2. migrate split_values（samelikelogic）
        # ============================================================
        old_split_values = state['split_values']
        new_split_values = self.output_layer.split_values.numpy()

        # copyoldparameters
        new_split_values[:, :, :old_capacity] = old_split_values

        # assignvaluesback
        self.output_layer.split_values.assign(new_split_values)

        print(f"   ✓ split_values migrated:")
        print(f"     Old shape: {old_split_values.shape}")
        print(f"     New shape: {new_split_values.shape}")

        # ============================================================
        # 3. migrate leaf_classes_array
        # ============================================================
        # leaf_classestofeature countnooff，directlycompletecopy
        old_leaf_classes = state['leaf_classes_array']

        # validationshape（shouldcompletelysame）
        if old_leaf_classes.shape != self.output_layer.leaf_classes_array.shape:
            raise ValueError(
                f"leaf_classes shape changed unexpectedly: "
                f"{old_leaf_classes.shape} vs {self.output_layer.leaf_classes_array.shape}"
            )

        # directlycopy
        self.output_layer.leaf_classes_array.assign(old_leaf_classes)

        print(f"   ✓ leaf_classes_array copied:")
        print(f"     Shape: {old_leaf_classes.shape} (unchanged)")

        # ============================================================
        # validationmigrateresult
        # ============================================================
        print(f"\n   🔍 Verification:")

        # checkoldparametersyesnobecorrectcopy
        new_split_index_check = self.output_layer.split_index_array.numpy()
        if not np.allclose(new_split_index_check[:, :, :old_capacity], old_split_index):
            raise RuntimeError("Migration verification failed: split_index not correctly copied")
        print(f"     ✅ split_index_array: old parameters preserved")

        new_split_values_check = self.output_layer.split_values.numpy()
        if not np.allclose(new_split_values_check[:, :, :old_capacity], old_split_values):
            raise RuntimeError("Migration verification failed: split_values not correctly copied")
        print(f"     ✅ split_values: old parameters preserved")

        print(f"     ✅ All parameters migrated correctly")

    # ============================================================================
    # 【steptwomodify7】auxiliaryhelpmethod - getexpansion history
    # ============================================================================
    def get_expansion_history(self):
        """
        getdynamicexpansionhistory tracking

        Returns:
            list: expansioneventlist，eacheventyesdictionarycontain：
                - timestamp: expansionoccurtimestamp
                - old_capacity: expansioncurrentcapacity
                - new_capacity: expansionaftercapacity
                - old_active: expansioncurrentactivefeature count
                - new_active: expansionafteractivefeature count
                - growth: capacitygrowthamount

        example:
            >>> history = model.get_expansion_history()
            >>> for i, event in enumerate(history):
            ...     print(f"Expansion {i+1}:")
            ...     print(f"  {event['old_capacity']} -> {event['new_capacity']}")
            ...     print(f"  Growth: +{event['growth']}")
        """
        return self._expansion_history.copy()

    def print_expansion_summary(self):
        """
        printexpansion historyextractneedinfo（convenientatdebug）
        """
        if not self._expansion_history:
            print("No expansions have occurred yet.")
            return

        print(f"\n{'=' * 70}")
        print(f"Expansion History Summary")
        print(f"{'=' * 70}")
        print(f"Total expansions: {len(self._expansion_history)}")
        print()

        for i, event in enumerate(self._expansion_history):
            from datetime import datetime
            dt = datetime.fromtimestamp(event['timestamp'])
            print(f"Expansion #{i + 1}:")
            print(f"  Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Capacity: {event['old_capacity']} → {event['new_capacity']} (+{event['growth']})")
            print(f"  Active: {event['old_active']} → {event['new_active']}")
            print()

        # whencurrentstatus
        print(f"Current state:")
        print(f"  Capacity: {self._max_reserved_features}")
        print(f"  Active features: {self._current_active_features}")
        print(f"  Available room: {self._max_reserved_features - self._current_active_features}")
        print(f"{'=' * 70}\n")

    def is_model_built(self):
        """checkmodelyesnoalreadybuild"""
        return self._model_built

    def get_model_info(self):
        """getmodelstatusinfo"""
        if self._model_built:
            preprocessing_mode = "Standard" if hasattr(self, 'encoder_loo') else "Skipped"
            preprocessors_fitted = getattr(self, '_preprocessors_fitted', False)
            return {
                'built': True,
                'depth': self.depth,
                'n_variables': getattr(self, 'number_of_variables', 'Unknown'),
                'n_classes': getattr(self, 'number_of_classes', 'Unknown'),
                'preprocessing_mode': preprocessing_mode,
                'preprocessors_fitted': preprocessors_fitted
            }
        else:
            return {'built': False}

    def get_parameter_snapshot(self, include_leaf_weights=False):
        """
        extractedwhencurrentmodelparameterssnapshot，useatablationexperimentanalysis

        thismethodyestoablationexperimentsetcomputecorecoreanalysisinterface。itallowIsingtrainingprocessin
        anymomentextractedmodelinternalparametersmatrix（Imatrix、Tmatrixetc），fromwhileanalysislearned
        surrogatemechanismyeshow toworkas。

        setcomputeprincipleconcept：

        ingtraditional machine learningexperimentin，Isusuallyonlyoffcoremodelfinalperformance（accuracy、lossetc）。
        butingablationexperimentin，Isneedsunderstandmodelinternalmechanism。thisthenlike a doctornot onlyneedknowpath
        patient temperature (external symptoms)，alsoneedcanlookXoptical（internalstructure）like。

        thismethodthenyesIs"Xoptical"，itletIscanenoughlooktomodelinternalparametersdistribution：
        - ImatrixtellsIseachnodetendencyatchoosewhichfeature
        - TmatrixtellsIseachfeaturethresholdvaluesyeswhat
        - Feature masktellsIswhichfeaturepositionyesglobally active

        throughingtrainingdifferentstagecallthismethodandsavesnapshot，Iscanobserveobserveparametersevolution，
        such aswhensomefeaturemissingwhen，mutuallyofffeatureIvaluesyesnoincrease（thiswillproveclearsurrogatelearning）。

        parametersinstruction：
            include_leaf_weights: booleanvalues，yesnocontainleaf nodeweight
                defaultFalse，becauseleaf nodeweight（leaf_classes_array）usuallyverylarge，
                but alsoforatunderstandsurrogatemechanismnotyesmustneed。
                ifsettoTrue，willcontaincompleteleaf nodepredictionweight。

        Returnsvalues：
            snapshot_dict: containhaveparametersdictionary
                {
                    'split_index_array': numpyarray，shape [n_estimators, n_nodes, n_features]
                        thisthenyesImatrix，eachelementmeansingsomenodechoosesomefeaturelogitvalues

                    'split_values': numpyarray，shape [n_estimators, n_nodes, n_features]
                        thisthenyesTmatrix，each element isforshouldfeaturethresholdvalues

                    'leaf_classes_array': numpyarray (optional)
                        leaf nodepredictionweight，onlyinginclude_leaf_weights=Truewhencontain

                    'feature_mask_info': dictionary
                        {
                            'current_active_features': currently activefeature count
                            'max_reserved_features': reservedmaximumfeaturecapacity
                            'active_feature_indices': activefeatureindexlist [0, 1, ..., n-1]
                        }

                    'model_config': dictionary
                        {
                            'depth': treedeepdegree
                            'n_estimators': estimatorcount
                            'n_internal_nodes': internalnodecount
                            'n_leaf_nodes': leaf nodecount
                        }

                    'timestamp': extractedtimestamp（useattracking）
                }

        useuseexample：
            >>> # ingtrainingloopindefinephasesavesnapshot
            >>> snapshots = []
            >>> for batch_idx in range(n_batches):
            ...     model.fit(X_batch, y_batch)
            ...     if batch_idx % 100 == 0:  # each100batchsavetimes
            ...         snapshot = model.get_parameter_snapshot()
            ...         snapshots.append(snapshot)

            >>> # analysisfeaturechooseevolution
            >>> for i, snap in enumerate(snapshots):
            ...     I_matrix = snap['split_index_array']
            ...     # analysisfirstestimatorrootnodefeaturechoosedistribution
            ...     root_feature_logits = I_matrix[0, 0, :]
            ...     print(f"Batch {i*100}: Top features = {np.argsort(root_feature_logits)[-5:]}")

        Notematters：
            1. thismethodwillwillTensorFlowtensorconvert toNumPyarray，for futureanalysis
            2. Returnsyeswhencurrentparametersdeepcopy，notwillshadowaffecttrainingprocess
            3. ifmodelalsonohavebuild，willthrowabnormal
            4. Returnsarraymayverylarge（especiallyyesensemblecase），Notememoryuseuse
        """
        import time

        # checkmodelyesnoalreadythroughbuild
        if not self._model_built:
            raise RuntimeError(
                "Cannot extract parameters: model has not been built yet. "
                "Please call fit() at least once before using get_parameter_snapshot()."
            )

        # checkoutput_layer（namelyGradTreeBlock）yesnoexist
        if not hasattr(self, 'output_layer') or self.output_layer is None:
            raise RuntimeError(
                "Cannot extract parameters: output_layer not found. "
                "This should not happen if model is built. Please check model initialization."
            )

        print(f"\n{'=' * 70}")
        print(f"📸 Extracting Parameter Snapshot")
        print(f"{'=' * 70}")

        # ========================================================================
        # first step：extractedImatrix（split_index_array）
        # ========================================================================

        try:
            # GradTreeBlockwillparametersastoweightstorage，Iscanthroughattributesnamedirectlyvisitproblem
            split_index_array_tensor = self.output_layer.split_index_array
            # convert tonumpyarray，for futureanalysis
            split_index_array = split_index_array_tensor.numpy()
            print(f"✅ Extracted split_index_array (I matrix)")
            print(f"   Shape: {split_index_array.shape}")
            print(f"   Range: [{split_index_array.min():.3f}, {split_index_array.max():.3f}]")
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to extract split_index_array: {e}. "
                "The GradTreeBlock might not have been properly initialized."
            )

        # ========================================================================
        # twostep：extractedTmatrix（split_values）
        # ========================================================================

        try:
            split_values_tensor = self.output_layer.split_values
            split_values = split_values_tensor.numpy()
            print(f"✅ Extracted split_values (T matrix)")
            print(f"   Shape: {split_values.shape}")
            print(f"   Range: [{split_values.min():.3f}, {split_values.max():.3f}]")
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to extract split_values: {e}. "
                "The GradTreeBlock might not have been properly initialized."
            )

        # ========================================================================
        # step 3：extractedleaf nodeweight（optional）
        # ========================================================================

        leaf_classes_array = None
        if include_leaf_weights:
            try:
                leaf_classes_array_tensor = self.output_layer.leaf_classes_array
                leaf_classes_array = leaf_classes_array_tensor.numpy()
                print(f"✅ Extracted leaf_classes_array")
                print(f"   Shape: {leaf_classes_array.shape}")
            except AttributeError as e:
                print(f"⚠️  Warning: Could not extract leaf_classes_array: {e}")
                # thisnotyescritical error，Iscancontinue

        # ========================================================================
        # fourstep：extractedfeature maskinfo
        # ========================================================================

        # Feature maskinfostorageingGradTreemainclassin
        active_features = self._current_active_features
        max_features = self._max_reserved_features

        # generateactive feature indiceslist
        # thistheseyeswhencurrent"globally active"featureposition
        active_feature_indices = list(range(active_features))

        feature_mask_info = {
            'current_active_features': active_features,
            'max_reserved_features': max_features,
            'active_feature_indices': active_feature_indices,
            'inactive_slots': max_features - active_features
        }

        print(f"✅ Extracted feature mask info")
        print(f"   Active features: {active_features}")
        print(f"   Reserved capacity: {max_features}")
        print(f"   Inactive slots: {max_features - active_features}")

        # ========================================================================
        # step 5：extractedmodelconfiginfo
        # ========================================================================

        # fromGradTreeBlockextractedstructureinfo
        depth = self.output_layer.depth
        n_estimators = self.output_layer.n_estimators
        n_internal_nodes = self.output_layer.internal_node_num_
        n_leaf_nodes = self.output_layer.leaf_node_num_

        model_config = {
            'depth': depth,
            'n_estimators': n_estimators,
            'n_internal_nodes': n_internal_nodes,
            'n_leaf_nodes': n_leaf_nodes
        }

        print(f"✅ Extracted model config")
        print(f"   Depth: {depth}")
        print(f"   Estimators: {n_estimators}")
        print(f"   Internal nodes: {n_internal_nodes}")

        # ========================================================================
        # sixstep：assemblesnapshotdictionary
        # ========================================================================

        snapshot = {
            'split_index_array': split_index_array,
            'split_values': split_values,
            'feature_mask_info': feature_mask_info,
            'model_config': model_config,
            'timestamp': time.time()
        }

        # ifcontainleaf nodeweight，addtosnapshotin
        if include_leaf_weights and leaf_classes_array is not None:
            snapshot['leaf_classes_array'] = leaf_classes_array

        # ========================================================================
        # step 7：computeandreporttellsnapshotstatisticsinfo
        # ========================================================================

        # computesnapshotmemoryoccupyuse（roughpassestimation）
        snapshot_size_mb = (
                                   split_index_array.nbytes +
                                   split_values.nbytes +
                                   (leaf_classes_array.nbytes if leaf_classes_array is not None else 0)
                           ) / (1024 * 1024)

        print(f"\n📊 Snapshot Statistics:")
        print(f"   Estimated size: {snapshot_size_mb:.2f} MB")
        print(f"   Total parameters: {split_index_array.size + split_values.size}")

        print(f"\n{'=' * 70}")
        print(f"✅ Parameter snapshot extracted successfully")
        print(f"{'=' * 70}\n")

        return snapshot

    def analyze_feature_selection(self, snapshot=None, node_indices=None,
                                  estimator_idx=0, top_k=5):
        """
        analysisgivedefinesnapshotinfeaturechoosepattern

        thisyesconvenientanalysismethod，helphelpyoufastunderstandmodelingeachnodechoosewhichfeature。
        itespeciallysuitableuseatvalidationlearned surrogatemechanism：youcanuseitcheckwhensomefeature
        missingwhen，modelyesnoenhancemutuallyofffeatureweight。

        parametersinstruction：
            snapshot: get_parameter_snapshot()Returnssnapshotdictionary
                iftoNone，willautomaticallycallget_parameter_snapshot()getwhencurrentsnapshot

            node_indices: needanalysisnodeindexlist
                iftoNone，defaultanalysiscurrentseveralnode（rootandfirstlayer）

            estimator_idx: needanalysisestimatorindex（ifuseuseensemble）
                defaultto0，namelyfirstestimator

            top_k: eachnodeshowtop-kfeature
                defaultto5

        Returnsvalues：
            analysis_dict: analysisresultdictionary
                {
                    node_idx: {
                        'top_features': [(feat_idx, logit_value), ...],
                        'feature_stats': featurelogitvaluesstatisticsinfo,
                        'all_logits': completelogitvector
                    }
                }

        useuseexample：
            >>> # analysiswhencurrentmodelfeaturechoose
            >>> analysis = model.analyze_feature_selection(top_k=3)
            >>>
            >>> # analysissavesnapshot
            >>> snapshot = snapshots[5]  # 5savesnapshot
            >>> analysis = model.analyze_feature_selection(
            ...     snapshot=snapshot,
            ...     node_indices=[0, 1, 2],  # rootandfirstlayertwonode
            ...     top_k=5
            ... )
        """
        # ifnohaveprovideprovidesnapshot，getwhencurrentsnapshot
        if snapshot is None:
            print("No snapshot provided, extracting current parameters...")
            snapshot = self.get_parameter_snapshot()

        # extractedImatrix
        I_matrix = snapshot['split_index_array']
        n_estimators, n_nodes, n_features = I_matrix.shape

        # validationestimatorindex
        if estimator_idx >= n_estimators:
            raise ValueError(
                f"estimator_idx {estimator_idx} out of range. "
                f"Model has {n_estimators} estimators (indices 0-{n_estimators - 1})."
            )

        # ifnohavespecifynode，defaultanalysiscurrentseveralimportantnode
        if node_indices is None:
            # analysisrootnodeandfirstlayernode
            depth = snapshot['model_config']['depth']
            if depth >= 2:
                # Root (idx=0) + firstlayertwonode (idx=1, 2)
                node_indices = [0, 1, 2]
            else:
                # deepdegreetoosmall，onlyanalysisroot
                node_indices = [0]

        print(f"\n{'=' * 70}")
        print(f"🔍 Feature Selection Analysis")
        print(f"{'=' * 70}")
        print(f"Estimator: {estimator_idx}")
        print(f"Analyzing {len(node_indices)} nodes")
        print(f"Top-{top_k} features per node")

        analysis_results = {}

        for node_idx in node_indices:
            if node_idx >= n_nodes:
                print(f"⚠️  Warning: Node {node_idx} out of range, skipping")
                continue

            # extractedthisnodefeaturelogits
            node_logits = I_matrix[estimator_idx, node_idx, :]

            # find outtop-kfeature
            # argsortReturnsfromsmalltolargeindex，Isneedmaximumk，sotake[-top_k:]
            top_k_indices = np.argsort(node_logits)[-top_k:][::-1]  # reversetofromlargetosmall
            top_k_values = node_logits[top_k_indices]

            # create(feature_idx, logit_value)matchforlist
            top_features = list(zip(top_k_indices.tolist(), top_k_values.tolist()))

            # computestatisticsinfo
            feature_stats = {
                'mean': float(np.mean(node_logits)),
                'std': float(np.std(node_logits)),
                'min': float(np.min(node_logits)),
                'max': float(np.max(node_logits)),
                'range': float(np.max(node_logits) - np.min(node_logits))
            }

            # storageresult
            analysis_results[node_idx] = {
                'top_features': top_features,
                'feature_stats': feature_stats,
                'all_logits': node_logits  # savecompletelogitvectorfor progressanalysis
            }

            # printresult
            print(f"\n📍 Node {node_idx}:")
            print(f"   Top-{top_k} features:")
            for rank, (feat_idx, logit_val) in enumerate(top_features, 1):
                print(f"      #{rank}: Feature {feat_idx} (logit={logit_val:.3f})")
            print(f"   Distribution: mean={feature_stats['mean']:.3f}, "
                  f"std={feature_stats['std']:.3f}, "
                  f"range=[{feature_stats['min']:.3f}, {feature_stats['max']:.3f}]")

        print(f"\n{'=' * 70}")
        print(f"✅ Analysis complete")
        print(f"{'=' * 70}\n")

        return analysis_results

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            'depth': trial.suggest_int("depth", 3, 10),

            'learning_rate_index': trial.suggest_float("learning_rate_index", 0.0001, 0.25),
            'learning_rate_values': trial.suggest_float("learning_rate_values", 0.0001, 0.25),
            'learning_rate_leaf': trial.suggest_float("learning_rate_leaf", 0.0001, 0.25),

            'cosine_decay_steps': trial.suggest_categorical("cosine_decay_steps", [0, 100, 1000]),
        }

        try:
            if args['objective'] != 'regression':
                params['focal_loss'] = trial.suggest_categorical("focal_loss", [True, False])
                params['temperature'] = trial.suggest_categorical("temperature", [0, 0.25])
        except:
            if cls.objective != 'regression':
                params['focal_loss'] = trial.suggest_categorical("focal_loss", [True, False])
                params['temperature'] = trial.suggest_categorical("temperature", [0, 0.25])
        return params

    @classmethod
    def set_trial_parameters(cls, trial):
        return cls.define_trial_parameters(trial, {})

    def get_optimizer(self):
        """
        getconfigoptimizerinstance

        Note：thisnotyes@property，whileyescommonmethod，withavoidtoKeras 3.xinternalattributesconflict

        important：tocompatibleTensorFlow Addons（TFA），whenuseuseTFAoptimizer（likeSWA）when，
        Isneedsuseuselegacyversionbasebaseoptimizer。thisyesbecauseTFAalsonohavecompletelysuitablematchKeras 3.xnewAPI。
        """
        if self.optimizer_name == 'AdamW':
            # AdamWselfTFA，useuselegacy optimizerastobasebase
            optimizer = tfa.optimizers.AdamW(learning_rate=self.learning_rate_leaf,
                                             weight_decay=0.025,
                                             clipnorm=1.)
        elif self.optimizer_name == 'Adam':
            # ifonlyuseuseAdam，canuseusenewversionAPI
            # buttoconsistent，thisinalsouseuselegacyversion
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate_leaf,
                                                        clipnorm=1.)
        elif self.optimizer_name == 'SWA':
            # SWAselfTFA，mustuseuselegacy optimizer
            # criticalfixcurrently：useuse tf.keras.optimizers.legacy.Adam
            base_optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=self.learning_rate_leaf,
                clipnorm=1.
            )
            optimizer = tfa.optimizers.SWA(
                base_optimizer,
                start_averaging=0,
                average_period=10
            )
        return optimizer


# belowyes GradTreeBlock class
# Note：GradTreeBlock classkeepnotchange，becausesteptwomodifymainlying GradTree mainclassin

class GradTreeBlock(tf.keras.layers.Layer):
    def __init__(self,
                 name='GradTreeBlock',
                 trainable=True,
                 dtype='float',
                 **kwargs):
        super(GradTreeBlock, self).__init__()

        for arg_key, arg_value in kwargs.items():
            setattr(self, arg_key, arg_value)

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        tf.keras.utils.set_random_seed(self.random_seed)

    def build(self, input_shape):

        tf.keras.utils.set_random_seed(self.random_seed)

        self.data_shape = self.number_of_variables
        self.number_of_variables = input_shape[-1]
        self.selected_variables = self.number_of_variables

        self.path_identifier_list = []
        self.internal_node_index_list = []
        for leaf_index in tf.unstack(tf.constant([i for i in range(self.leaf_node_num_)])):
            for current_depth in tf.unstack(tf.constant([i for i in range(1, self.depth + 1)])):
                path_identifier = tf.cast(
                    tf.math.floormod(tf.math.floor(leaf_index / (tf.math.pow(2, (self.depth - current_depth)))), 2),
                    tf.float32)
                internal_node_index = tf.cast(tf.cast(tf.math.pow(2, (current_depth - 1)), tf.float32) + tf.cast(
                    tf.math.floor(leaf_index / (tf.math.pow(2, (self.depth - (current_depth - 1))))), tf.float32) - 1.0,
                                              tf.int64)
                self.path_identifier_list.append(path_identifier)
                self.internal_node_index_list.append(internal_node_index)
        self.path_identifier_list = tf.reshape(tf.stack(self.path_identifier_list), (-1, self.depth))
        self.internal_node_index_list = tf.reshape(tf.cast(tf.stack(self.internal_node_index_list), tf.int64),
                                                   (-1, self.depth))

        leaf_classes_array_shape = [self.n_estimators,
                                    self.leaf_node_num_, ] if self.objective == 'binary' or self.objective == 'regression' else [
            self.n_estimators, self.leaf_node_num_, self.number_of_classes]

        # 【modify】useusereservedfeature spacewhilenotyesactualactualfeature count
        # originaluseuse：self.selected_variables（actualactualfeature count）
        # Nowuseuse：self.max_features（reservedfeaturecapacity）

        # getdynamic feature spaceconfig
        self.active_features = getattr(self, 'active_features', self.number_of_variables)
        self.max_features = getattr(self, 'max_features', self.number_of_variables)

        print(f"📦 Creating parameter matrices with reserved space:")
        print(f"   - Active features: {self.active_features}")
        print(f"   - Reserved capacity: {self.max_features}")

        # ========================================================================
        # 【Step 2 Added】checkyesnouse migratedparams for initialization（Expansion scenario）
        # ========================================================================
        use_migrated = getattr(self, 'use_migrated_params', False)

        if use_migrated:
            # Expansion scenario：use migratedparametersdirectlyinitialize
            migrated_params = getattr(self, 'migrated_params', None)
            if migrated_params is None:
                raise RuntimeError("use_migrated_params=True but migrated_params not provided")

            # callcustominitializemethod（inspired by_initialize_with_custom_matricesapproach）
            self.initialize_with_migrated_parameters(migrated_params)
            # initializedone，directlyReturns
            return

        # Normal scenario：useuserandominitialize
        self.split_values = self.add_weight(name="split_values",
                                            initializer=tf.keras.initializers.get({'class_name': self.initializer,
                                                                                   'config': {
                                                                                       'seed': self.random_seed + 2}}),
                                            trainable=True,
                                            shape=[self.n_estimators, self.internal_node_num_, self.max_features])

        self.split_index_array = self.add_weight(name="split_index_array",
                                                 initializer=tf.keras.initializers.get({'class_name': self.initializer,
                                                                                        'config': {
                                                                                            'seed': self.random_seed + 3}}),
                                                 trainable=True,
                                                 shape=[self.n_estimators, self.internal_node_num_, self.max_features])
        self.leaf_classes_array = self.add_weight(name="leaf_classes_array",
                                                  initializer=tf.keras.initializers.get({'class_name': self.initializer,
                                                                                         'config': {
                                                                                             'seed': self.random_seed + 4}}),
                                                  trainable=True,
                                                  shape=leaf_classes_array_shape)

        # 【Added】createfeatureactivemask（controlsystemwhichfeaturepositionyesactiveleap）
        # currentactive_featurespositionto0（active），itsremaindertolargenegative（dormant）
        initial_mask = self._create_feature_mask(self.active_features, self.max_features)
        self.feature_mask = tf.Variable(
            initial_value=initial_mask,
            trainable=False,  # masknotparametertotraining
            dtype=tf.float32,
            name="feature_mask"
        )

        print(f"   - Feature mask created: {self.active_features} active, "
              f"{self.max_features - self.active_features} dormant")

    def call(self, inputs, training, output_weights=False):
        """
        【Step 3 modify】supportinstance-level maskcurrenttowardpropagation
        【XLAcompatibleFix】useusepureTensorFlowoperateas，avoidPythoncontrolsystemflow

        automaticallyprocessing：
        1. detectioninputinNaN（missingvalues）
        2. generateinstance_mask [batch_size, max_features]
        3. shouldusetwolayercurrentlyexchangemask：feature-level + instance-level
        4. implementationlearned surrogate splitseffect

        parameters:
            inputs: [batch_size, max_features]，maycontainNaN
            training: trainingpatternflag
            output_weights: yesnooutputweight
        """

        # ========================================================================
        # 【Step 3 Added + XLAFix】automaticallygenerate Instance-Level Mask
        # ========================================================================
        # detectionNaNposition（Returnsboolean mask）
        is_nan_mask = tf.math.is_nan(inputs)
        # shape: [batch_size, max_features], dtype: bool

        # generateinstance_mask：NaNpositionto-1000.0，otherpositionto0.0
        # useusetf.wherewhilenotyesPython if，ensureXLAcompatible
        instance_mask = tf.where(is_nan_mask, -1000.0, 0.0)
        # shape: [batch_size, max_features], dtype: float32

        # willinputsinNaNreplaceto0.0（avoidaftercontinuecomputepropagationNaN）
        inputs = tf.where(is_nan_mask, 0.0, inputs)
        # shape: [batch_size, max_features], dtype: float32

        # ========================================================================
        # originalhavelogic：expansioninputdimensions
        # ========================================================================
        X_estimator = tf.expand_dims(inputs, 1)
        # shape: [batch_size, 1, max_features]

        # ========================================================================
        # 【Step 3 modify】shouldusetwolayercurrentlyexchangeMask
        # ========================================================================
        # Layer 1: Feature-Level Mask（global）
        feature_mask_expanded = tf.reshape(self.feature_mask, [1, 1, -1])
        split_index_array_masked = self.split_index_array + feature_mask_expanded

        # Layer 2: Instance-Level Mask（per-sample）
        instance_mask_expanded = tf.reshape(
            instance_mask,
            [tf.shape(instance_mask)[0], 1, 1, -1]
        )
        split_index_array_masked = split_index_array_masked[None, :, :, :] + \
                                   instance_mask_expanded
        # shape: [batch_size, n_estimators, internal_nodes, max_features]

        # ========================================================================
        # 【Step 3 keep】Entmax + Hardmax（ingnewdimensionsaboveoperateas）
        # ========================================================================
        # Entmaxingfinallydimension（max_features）abovenormalization
        # foreach [batch, estimator, node] position，independentingfeaturesabovedoentmax
        split_index_array = entmax15(split_index_array_masked)
        # shape: [batch_size, n_estimators, internal_nodes, max_features]

        # useuseST-Operatortoone-hotencodingfeatureindex
        # each [batch, estimator, node] positionchoosefeature
        adjust_constant = tf.stop_gradient(
            split_index_array - tfa.seq2seq.hardmax(split_index_array)
        )
        split_index_array = split_index_array - adjust_constant
        # shape: [batch_size, n_estimators, internal_nodes, max_features]
        # each [b, e, i] positionyesone-hotvector（ingndimensionsabove）

        # ========================================================================
        # 【Step 3 modify】Gatheroperateas - adjustaxis
        # ========================================================================
        # generateuseataftercontinuecomputetensor：
        # internal_node_index_liststorageeachitempaththroughinternalnodeindex
        # shape: [leaf_nodes, depth]

        # original：split_index_arrayyes [e, i, n]，ingaxis=1（idimensions）gather
        # Now：split_index_arrayyes [b, e, i, n]，ingaxis=2（idimensions）gather
        split_index_array_selected = tf.gather(
            split_index_array,
            self.internal_node_index_list,
            axis=2  # 【modify】fromaxis=1changetoaxis=2
        )
        # shape: [batch_size, n_estimators, leaf_nodes, depth, max_features]

        # split_valuesnotneedsbatchdimensions，keeporiginallike
        split_values_selected = tf.gather(
            self.split_values,
            self.internal_node_index_list,
            axis=1  # 【keep】stillnaturallyyesaxis=1
        )
        # shape: [n_estimators, leaf_nodes, depth, max_features]

        # ========================================================================
        # 【Step 3 modify】Einsumcompute - adjustnotation
        # ========================================================================
        # split_index_array_selectedNowyesone-hotencoding，througheinsumchooseforshouldvalues

        # --- computes1_sum：eachnodethreshold ---
        # original："eldn,eldn->eld"
        # Now：split_index_array_selectedmanybatchdimensions
        s1_sum = tf.einsum(
            "eldn,beldn->beld",  # 【modify】right sideincreasebdimensions
            split_values_selected,  # [e, l, d, n]
            split_index_array_selected  # [b, e, l, d, n]
        )
        # shape: [batch_size, n_estimators, leaf_nodes, depth]

        # --- computes2_sum：eachsampleingeachnodefeaturevalues ---
        # original："ben,eldn->beld"
        # Now：split_index_array_selectedmanybatchdimensions
        s2_sum = tf.einsum(
            "ben,beldn->beld",  # 【modify】right sideincreasebdimensions
            X_estimator,  # [b, e, n]
            split_index_array_selected  # [b, e, l, d, n]
        )
        # shape: [batch_size, n_estimators, leaf_nodes, depth]

        # ========================================================================
        # 【Step 3 keep】aftercontinuecomputecompletelynotchange
        # ========================================================================
        # computesplitresult (outputshape: [b, e, l, d])
        node_result = (tf.nn.softsign(s1_sum - s2_sum) + 1) / 2

        # useuseST operatortohard decision
        node_result_corrected = node_result - tf.stop_gradient(
            node_result - tf.round(node_result)
        )

        # throughmultiplicationreducepath，toeachitempathresult (outputshape: [b, e, l])
        p = tf.reduce_prod(
            ((1 - self.path_identifier_list) * node_result_corrected +
             self.path_identifier_list * (1 - node_result_corrected)),
            axis=3
        )

        # toeachestimatororiginalprediction
        # optionalingweightedcurrentconvert toprobabilitydistribution
        if self.objective == 'regression':
            layer_output = tf.einsum('el,bel->be', self.leaf_classes_array, p)
        elif self.objective == 'binary':
            if self.from_logits:
                layer_output = tf.einsum('el,bel->be', self.leaf_classes_array, p)
            else:
                layer_output = tf.math.sigmoid(
                    tf.einsum('el,bel->be', self.leaf_classes_array, p)
                )
        elif self.objective == 'classification':
            if self.from_logits:
                layer_output = tf.einsum('elc,bel->bec', self.leaf_classes_array, p)
            else:
                layer_output = tf.keras.activations.softmax(
                    tf.einsum('elc,bel->bec', self.leaf_classes_array, p)
                )

        return layer_output

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _create_feature_mask(self, active_count, total_count):
        """
        createfeaturemask：
        - currentactive_countposition：0（active）
        - afterposition：-1000（dormant，ingentmaxafterprobabilityclose to0）

        Returnsshape：[total_count]
        """
        mask = np.zeros(total_count, dtype=np.float32)
        mask[active_count:] = -1000  # largenegative，letentmaxoutputclose to0
        return mask

    def update_active_features(self, new_active_count):
        """
        updateactivefeature count（throughadjustmask）

        thismethodingnewfeatureaddwhenbecall，only needsupdatemask，
        notneedsrebuildwholemodelorparametersmatrix
        """
        if new_active_count > self.max_features:
            raise ValueError(
                f"Cannot activate {new_active_count} features. "
                f"Maximum capacity is {self.max_features}"
            )

        # updatemask：newactivepositionfromlargenegativechangeto0
        new_mask = self._create_feature_mask(new_active_count, self.max_features)
        self.feature_mask.assign(new_mask)

        # updateactive_featurescount
        old_active = self.active_features
        self.active_features = new_active_count

        print(f"   🎯 GradTreeBlock mask updated: {old_active} -> {new_active_count}")

    def initialize_with_migrated_parameters(self, migrated_params):
        """
        【Step 2 Added】usemigrateafterparametersdirectlyinitializeweight

        thismethodinspired bycustominitializeapproach，ingExpansion scenarioin：
        - notetcKerasautomaticallycreateweight
        - directlyusemigrateafterparametersvaluescreateweight
        - avoid"firstcreateemptyweight，againcopyparameters"two-step operationas

        parameters:
            migrated_params: dictionary，contain：
                - 'split_index_array': migrateaftercompletematrix（alreadycontainoldparameters+newcolumns）
                - 'split_values': migrateaftercompletematrix
                - 'leaf_classes_array': leaf nodeparameters（completecopy）
                - 'feature_mask': newmask

        setcomputeapproach：
        thismethodclasssimilarat_initialize_with_custom_matrices，butuseatExpansion scenario。
        ituseuseTensorFlowadd_weight()andconstant_initializer()，
        steptopositionusecorrectvaluescreateweight。
        """
        print("   🔧 GradTreeBlock: Initializing with migrated parameters")

        # use migratedafterparametersdirectlycreateweight
        # critical：useusetf.constant_initializer()，thislikeweightingcreatewhenthenhavecorrectvalues

        self.split_values = self.add_weight(
            name="split_values",
            shape=migrated_params['split_values'].shape,
            initializer=tf.constant_initializer(migrated_params['split_values'].astype(np.float32)),
            trainable=True
        )

        self.split_index_array = self.add_weight(
            name="split_index_array",
            shape=migrated_params['split_index_array'].shape,
            initializer=tf.constant_initializer(migrated_params['split_index_array'].astype(np.float32)),
            trainable=True
        )

        self.leaf_classes_array = self.add_weight(
            name="leaf_classes_array",
            shape=migrated_params['leaf_classes_array'].shape,
            initializer=tf.constant_initializer(migrated_params['leaf_classes_array'].astype(np.float32)),
            trainable=True
        )

        # createfeature mask（use migratedaftermaskvalues）
        self.feature_mask = tf.Variable(
            initial_value=migrated_params['feature_mask'].astype(np.float32),
            trainable=False,
            dtype=tf.float32,
            name="feature_mask"
        )

        print(f"   ✅ Migrated parameters initialized as weights")
        print(f"      - split_index_array: {self.split_index_array.shape}")
        print(f"      - split_values: {self.split_values.shape}")
        print(f"      - leaf_classes_array: {self.leaf_classes_array.shape}")


# auxiliaryhelpfunction（fromoriginalfilecopy）
def entmax15(inputs, axis=-1):
    """
    Entmax 1.5 implementation, heavily inspired by
     * paper: https://arxiv.org/pdf/1905.05702.pdf
     * pytorch code: https://github.com/deep-spin/entmax
    :param inputs: similar to softmax logits, but for entmax1.5
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    :return: entmax activations of same shape as inputs
    """

    @tf.custom_gradient
    def _entmax_inner(inputs):
        with tf.name_scope('entmax'):
            inputs = inputs / 2  # divide by 2 so as to solve actual entmax
            inputs -= tf.reduce_max(inputs, axis, keepdims=True)  # subtract max for stability

            threshold, _ = entmax_threshold_and_support(inputs, axis)
            outputs_sqrt = tf.nn.relu(inputs - threshold)
            outputs = tf.square(outputs_sqrt)

        def grad_fn(d_outputs):
            with tf.name_scope('entmax_grad'):
                d_inputs = d_outputs * outputs_sqrt
                q = tf.reduce_sum(d_inputs, axis=axis, keepdims=True)
                q = q / tf.reduce_sum(outputs_sqrt, axis=axis, keepdims=True)
                d_inputs -= q * outputs_sqrt
                return d_inputs

        return outputs, grad_fn

    return _entmax_inner(inputs)


@tf.custom_gradient
def sparse_entmax15_loss_with_logits(labels, logits):
    """
    Sparse entmax loss, compatible with tensorflow.
    :param labels: shape [batch_size], integer labels
    :param logits: shape [batch_size, num_classes], pre-activations
    :returns: loss, scalar
    """
    assert logits.shape.ndims == 2 and labels.shape.ndims == 1
    with tf.name_scope('sparse_entmax15_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - tf.one_hot(labels, depth=tf.shape(logits)[-1], axis=-1)
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)

    def grad_fn(grad_output):
        with tf.name_scope('sparse_entmax15_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


@tf.custom_gradient
def sparse_entmax15_loss_with_logits_v2(labels, logits):
    """
    Sparse entmax loss, compatible with tensorflow.
    :param labels: shape [batch_size, num_classes], one-hot encoded labels
    :param logits: shape [batch_size, num_classes], pre-activations
    :returns: loss, scalar
    """
    assert logits.shape.ndims == 2
    with tf.name_scope('sparse_entmax15_loss'):
        p_star = entmax15(logits, axis=-1)
        omega_entmax15 = (1 - (tf.reduce_sum(p_star * tf.sqrt(p_star), axis=-1))) / 0.75
        p_incr = p_star - labels
        loss = omega_entmax15 + tf.einsum("ij,ij->i", p_incr, logits)

    def grad_fn(grad_output):
        with tf.name_scope('sparse_entmax15_loss_grad'):
            return None, grad_output[..., None] * p_incr

    return loss, grad_fn


def top_k_over_axis(inputs, k, axis=-1, **kwargs):
    """ performs tf.nn.top_k over any chosen axis """
    with tf.name_scope('top_k_along_axis'):
        if axis == -1:
            return tf.nn.top_k(inputs, k, **kwargs)

        perm_order = list(range(inputs.shape.ndims))
        perm_order.append(perm_order.pop(axis))
        inv_order = [perm_order.index(i) for i in range(len(perm_order))]

        input_perm = tf.transpose(inputs, perm_order)
        input_perm_sorted, sort_indices_perm = tf.nn.top_k(
            input_perm, k=k, **kwargs)

        input_sorted = tf.transpose(input_perm_sorted, inv_order)
        sort_indices = tf.transpose(sort_indices_perm, inv_order)
    return input_sorted, sort_indices


def _make_ix_like(inputs, axis=-1):
    """ creates indices 0, ... , input[axis] unsqueezed to input dimensios """
    assert inputs.shape.ndims is not None
    rho = tf.cast(tf.range(1, tf.shape(inputs)[axis] + 1), dtype=inputs.dtype)
    view = [1] * inputs.shape.ndims
    view[axis] = -1
    return tf.reshape(rho, view)


def gather_over_axis(values, indices, gather_axis):
    """
    replicates the behavior of torch.gather for tf<=1.8;
    for newer versions use tf.gather with batch_dims
    :param values: tensor [d0, ..., dn]
    :param indices: int64 tensor of same shape as values except for gather_axis
    :param gather_axis: performs gather along this axis
    :returns: gathered values, same shape as values except for gather_axis
        If gather_axis == 2
        gathered_values[i, j, k, ...] = values[i, j, indices[i, j, k, ...], ...]
        see torch.gather for more detils
    """
    assert indices.shape.ndims is not None
    assert indices.shape.ndims == values.shape.ndims

    ndims = indices.shape.ndims
    gather_axis = gather_axis % ndims
    shape = tf.shape(indices)

    selectors = []
    for axis_i in range(ndims):
        if axis_i == gather_axis:
            selectors.append(indices)
        else:
            index_i = tf.range(tf.cast(shape[axis_i], dtype=indices.dtype), dtype=indices.dtype)
            index_i = tf.reshape(index_i, [-1 if i == axis_i else 1 for i in range(ndims)])
            index_i = tf.tile(index_i, [shape[i] if i != axis_i else 1 for i in range(ndims)])
            selectors.append(index_i)

    return tf.gather_nd(values, tf.stack(selectors, axis=-1))


def entmax_threshold_and_support(inputs, axis=-1):
    """
    Computes clipping threshold for entmax1.5 over specified axis
    NOTE this implementation uses the same heuristic as
    the original code: https://tinyurl.com/pytorch-entmax-line-203
    :param inputs: (entmax1.5 inputs - max) / 2
    :param axis: entmax1.5 outputs will sum to 1 over this axis
    """

    with tf.name_scope('entmax_threshold_and_support'):
        num_outcomes = tf.shape(inputs)[axis]
        inputs_sorted, _ = top_k_over_axis(inputs, k=num_outcomes, axis=axis, sorted=True)

        rho = _make_ix_like(inputs, axis=axis)

        mean = tf.cumsum(inputs_sorted, axis=axis) / rho

        mean_sq = tf.cumsum(tf.square(inputs_sorted), axis=axis) / rho
        delta = (1 - rho * (mean_sq - tf.square(mean))) / rho

        delta_nz = tf.nn.relu(delta)
        tau = mean - tf.sqrt(delta_nz)

        support_size = tf.reduce_sum(tf.cast(tf.less_equal(tau, inputs_sorted), tf.int64), axis=axis, keepdims=True)

        tau_star = gather_over_axis(tau, support_size - 1, axis)
    return tau_star, support_size