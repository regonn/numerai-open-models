import gc
import json
import os
import random
import sys
import traceback

import cloudpickle
import lightgbm as lgb
import numerapi
import numpy as np
import pandas as pd
import polars as pl
import requests
from dotenv import load_dotenv
from google.cloud import storage
from google.oauth2.service_account import Credentials
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
NUMERAI_PUBLIC_ID = os.environ["NUMERAI_PUBLIC_ID"]
NUMERAI_SECRET_KEY = os.environ["NUMERAI_SECRET_KEY"]
NUMERAI_MODEL_ID = "open_model_11"
napi = numerapi.NumerAPI(NUMERAI_PUBLIC_ID, NUMERAI_SECRET_KEY)


def post_discord(message):
    """Send message to Discord webhook"""
    try:
        requests.post(DISCORD_WEBHOOK_URL, {"content": message})
    except Exception as e:
        print(f"Failed to send Discord notification: {e}")


def get_error_details():
    """Get detailed error information including type, message, and location"""
    exc_type, exc_value, exc_traceback = sys.exc_info()

    if exc_type is None:
        return "No error information available"

    # Get error type and message
    error_type = exc_type.__name__
    error_message = str(exc_value)

    # Get the most recent traceback frame (where the error occurred)
    if exc_traceback:
        tb_list = traceback.extract_tb(exc_traceback)
        if tb_list:
            # Get the last frame (most recent)
            last_frame = tb_list[-1]
            error_location = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"
        else:
            error_location = "Unknown location"
    else:
        error_location = "Unknown location"

    # Get full traceback for debugging
    full_traceback = traceback.format_exc()

    return {"type": error_type, "message": error_message, "location": error_location, "full_traceback": full_traceback}


# Main execution with error handling
try:
    # Send start notification
    start_message = f"🚀 **{NUMERAI_MODEL_ID}** training process started"
    post_discord(start_message)
    print(start_message)

    # GCP credentials setup
    cred = Credentials.from_service_account_info(
        {
            "type": "service_account",
            "project_id": os.environ["GCP_PROJECT_ID"],
            "private_key_id": os.environ["GCS_PRIVATE_KEY_ID"],
            "private_key": os.environ["GCS_PRIVATE_KEY"].replace("\\n", "\n"),
            "client_email": os.environ["GCS_CLIENT_MAIL"],
            "client_id": os.environ["GCS_CLIENT_ID"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.environ["GCS_CLIENT_X509_CERT_URL"],
        }
    )
    client = storage.Client(credentials=cred)
    bucket = client.get_bucket(os.environ["GCS_BUCKET_NAME"])

    # Download datasets
    post_discord(f"📥 Downloading datasets for {NUMERAI_MODEL_ID}...")
    napi.download_dataset("v5.0/train.parquet")
    napi.download_dataset("v5.0/validation.parquet")
    napi.download_dataset("v5.0/live.parquet")
    napi.download_dataset("v5.0/features.json")
    napi.download_dataset("v5.0/live_example_preds.parquet")

    model_file_name = f"./predict_{NUMERAI_MODEL_ID}.pkl"
    feature_metadata = json.load(open("v5.0/features.json"))
    features = feature_metadata["feature_sets"]["medium"]

    # If feature importance file is not available, use some features with "group" in name as risk factors
    neutralization_features = [f for f in features if "group" in f][:20]
    if len(neutralization_features) < 5:
        # If no group features, use first 20 features for neutralization
        neutralization_features = features[:20]

    target = "target"

    # Train data - use polars lazy mode for memory efficiency
    post_discord(f"📊 Loading training data for {NUMERAI_MODEL_ID}...")
    train_lazy = pl.scan_parquet("v5.0/train.parquet").select(["era"] + features + [target])
    train_lazy = train_lazy.filter(pl.col(target).is_not_null())
    # Materialize to pandas only when needed
    train = train_lazy.collect().to_pandas()
    train.to_csv("train.csv", index=False)

    # Validation data - use polars lazy mode
    validation_lazy = pl.scan_parquet("v5.0/validation.parquet").select(["era", "data_type"] + features + [target])
    validation_lazy = validation_lazy.filter((pl.col("data_type") == "validation") & pl.col(target).is_not_null())
    # Materialize to pandas
    validation = validation_lazy.collect().to_pandas()

    # Get example predictions for checking format - smaller file, ok to use pandas directly
    example_preds = pd.read_parquet("v5.0/live_example_preds.parquet")

    # Feature engineering - add feature interactions and statistical features
    def engineer_features(df):
        """Engineer features for either a pandas DataFrame or from a polars DataFrame
        Returns a pandas DataFrame for compatibility with existing model training code
        """
        # Check if input is a polars DataFrame or LazyFrame
        is_polars = isinstance(df, (pl.DataFrame, pl.LazyFrame))

        if is_polars:
            # For polars, we'll work with lazy operations as much as possible
            # Convert to LazyFrame if not already
            if isinstance(df, pl.DataFrame):
                lf = df.lazy()
            else:
                lf = df

            # Only perform era-based feature engineering if 'era' column exists
            # Use schema() to get column names for LazyFrame
            try:
                df_columns = df.schema.keys() if hasattr(df, "schema") else df.columns
                if "era" in df_columns:
                    try:
                        # Group features by era to capture temporal dynamics - using polars syntax
                        era_stats = lf.group_by("era").agg(
                            [pl.col(feat).mean().alias(f"era_mean_{feat}") for feat in features]
                        )
                        # Join back to main dataframe using correct Polars syntax
                        lf = lf.join(era_stats, on="era", how="left")
                    except Exception as e:
                        print(f"Warning: Could not create era-based features: {e}")
                        # Continue without era-based features
            except Exception as e:
                print(f"Warning: Could not check for era column: {e}")

            # Create interaction features between high importance features
            # Use only top 5 features to avoid explosion of dimensions
            top_5_features = neutralization_features[:5]
            for i in range(len(top_5_features)):
                for j in range(i + 1, len(top_5_features)):
                    feat_i, feat_j = top_5_features[i], top_5_features[j]
                    # Multiplication interaction
                    lf = lf.with_columns((pl.col(feat_i) * pl.col(feat_j)).alias(f"{feat_i}_mul_{feat_j}"))
                    # Division interaction (with safety for zeros)
                    lf = lf.with_columns(
                        (
                            pl.col(feat_i)
                            / pl.when(pl.col(feat_j) == 0).then(1e-8).otherwise(pl.col(feat_j)).fill_null(1e-8)
                        ).alias(f"{feat_i}_div_{feat_j}")
                    )

            # Add polynomial features for top features
            for feat in top_5_features:
                lf = lf.with_columns((pl.col(feat) ** 2).alias(f"{feat}_squared"))

            # Collect and convert to pandas for compatibility with rest of code
            return lf.collect().to_pandas()
        else:
            # pandas implementation (unchanged)
            # Create a copy to avoid modifying the original dataframe
            df_engineered = df.copy()

            # Only perform era-based feature engineering if 'era' column exists
            if "era" in df.columns:
                try:
                    # Group features by era to capture temporal dynamics
                    era_stats = df.groupby("era")[features].mean()
                    era_stats.columns = [f"era_mean_{col}" for col in era_stats.columns]
                    df_engineered = df_engineered.join(era_stats, on="era")
                except Exception as e:
                    print(f"Warning: Could not create era-based features: {e}")
                    # Continue without era-based features

            # Create interaction features between high importance features
            # Use only top 5 features to avoid explosion of dimensions
            top_5_features = neutralization_features[:5]
            for i in range(len(top_5_features)):
                for j in range(i + 1, len(top_5_features)):
                    feat_i, feat_j = top_5_features[i], top_5_features[j]
                    # Multiplication interaction
                    df_engineered[f"{feat_i}_mul_{feat_j}"] = df_engineered[feat_i] * df_engineered[feat_j]
                    # Division interaction (with safety for zeros) - use numpy where for better performance
                    denominator = df_engineered[feat_j].copy()
                    denominator = denominator.replace([0, np.inf, -np.inf], 1e-8)
                    df_engineered[f"{feat_i}_div_{feat_j}"] = df_engineered[feat_i] / denominator

            # Add polynomial features for top features
            for feat in top_5_features:
                df_engineered[f"{feat}_squared"] = df_engineered[feat] ** 2

            return df_engineered

    # Apply feature engineering
    post_discord(f"🔧 Applying feature engineering for {NUMERAI_MODEL_ID}...")
    print("Applying feature engineering...")
    train_engineered = engineer_features(train)
    validation_engineered = engineer_features(validation)

    # Get the new feature list after engineering
    engineered_features = [
        col
        for col in train_engineered.columns
        if col not in ["era", "data_type", target] and not pd.isna(train_engineered[col]).any()
    ]

    # Save the feature engineering state for prediction
    feature_engineering_state = {
        "engineered_features": engineered_features,
        "neutralization_features": neutralization_features,
        "features": features,
        "target": target,
    }

    # Make it globally accessible for the predict function
    globals()["feature_engineering_state"] = feature_engineering_state

    # Memory optimization: Feature selection to reduce dimensionality
    print(f"Total engineered features: {len(engineered_features)}")
    if len(engineered_features) > 500:
        print(f"Too many features ({len(engineered_features)}), reducing to top 500 for memory efficiency...")

        # Create a small model to get feature importances
        sample_model = lgb.LGBMRegressor(
            n_estimators=50, max_depth=2, subsample=0.2, colsample_bytree=0.1, random_state=42, n_jobs=1
        )

        # Use a smaller subset of data to estimate feature importances
        sample_size = min(5000, len(train_engineered))
        sample_indices = np.random.choice(len(train_engineered), sample_size, replace=False)
        sample_data = train_engineered.iloc[sample_indices]

        # Fit the model to get feature importances
        sample_model.fit(
            sample_data[engineered_features],
            sample_data[target],
        )

        # Get feature importances
        importances = sample_model.feature_importances_

        # Create a pandas Series for easier sorting
        feature_importances = pd.Series(importances, index=engineered_features)

        # Select top 500 features
        top_features = feature_importances.sort_values(ascending=False).head(500).index.tolist()

        print(f"Selected {len(top_features)} top features for training")
        engineered_features = top_features
        # Update the feature engineering state
        feature_engineering_state["engineered_features"] = engineered_features
        globals()["feature_engineering_state"] = feature_engineering_state
    else:
        print("Using all engineered features for training")
        # Ensure the global state is updated
        globals()["feature_engineering_state"] = feature_engineering_state

    # Force garbage collection
    gc.collect()

    # Feature neutralization function
    def neutralize(df, columns, by, proportion=1.0):
        """Neutralize a dataframe of predictions against a set of features

        Parameters
        ----------
        df: DataFrame with the predictions
        columns: list of str, columns to neutralize
        by: list of str, columns to neutralize against
        proportion: float, amount of neutralization to apply (1.0 = full neutralization)

        Returns
        -------
        DataFrame with neutralized predictions

        """
        if proportion == 0.0:
            return df.copy()

        # Convert to numpy for performance
        scores = df[columns].values
        factors = df[by].values

        # Standardize the factors
        factors = StandardScaler().fit_transform(factors)

        # Perform neutralization (regression residuals)
        scores_neutralized = np.zeros_like(scores)
        for i in range(scores.shape[1]):
            # Perform linear regression
            model = LinearRegression(fit_intercept=False).fit(factors, scores[:, i])
            # Calculate the residuals
            scores_neutralized[:, i] = scores[:, i] - proportion * model.predict(factors)

        # Convert back to DataFrame
        df_neutralized = pd.DataFrame(scores_neutralized, index=df.index, columns=columns)

        return df_neutralized

    # Improved model parameters optimized for memory efficiency
    model_params = {
        "n_estimators": 1000,
        "learning_rate": 0.003,
        "max_depth": 3,  # 浅い木で過学習を防止
        "num_leaves": 10,  # 少ない葉の数でメモリ使用量を削減
        "colsample_bytree": 0.3,  # より少ない特徴量を使用
        "subsample": 0.3,  # より少ないサンプルを使用
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "min_child_samples": 50,
        "min_child_weight": 1.0,
        "bagging_freq": 1,
        "feature_fraction": 0.3,  # より少ない特徴量を使用
        "random_state": 42,
        "n_jobs": 1,  # シングルスレッドでメモリ使用量を削減
        "verbose": -1,
        "boosting_type": "gbdt",
        "min_gain_to_split": 0.01,
        "min_data_in_bin": 5,
        "max_bin": 50,  # ビンの数を減らしてメモリ使用量を削減
    }

    post_discord(f"🤖 Starting model training with cross-validation for {NUMERAI_MODEL_ID}...")
    print("Training model with cross-validation...")

    # Extract unique eras for time-based validation
    unique_eras = sorted(train_engineered["era"].unique())
    n_eras = len(unique_eras)
    n_folds = 5  # Number of folds for cross-validation

    # Initialize list to store models from each fold
    models = []

    # Track validation scores across folds
    val_scores = []

    # Time-series cross-validation - use recent eras for validation
    for fold in range(n_folds):
        post_discord(f"🔄 Training fold {fold + 1}/{n_folds} for {NUMERAI_MODEL_ID}")
        print(f"Training fold {fold + 1}/{n_folds}")

        # Determine cutoff points for this fold
        # We'll use time-based splits to avoid data leakage
        validation_eras = unique_eras[-(fold + 1) * 3 :] if fold < n_folds - 1 else unique_eras[-3:]
        train_eras = [era for era in unique_eras if era not in validation_eras]

        # Split data
        train_fold = train_engineered[train_engineered["era"].isin(train_eras)]
        val_fold = train_engineered[train_engineered["era"].isin(validation_eras)]

        # Memory cleanup
        gc.collect()

        # Memory optimization: Convert to 32-bit floats to reduce memory usage
        X_train = train_fold[engineered_features].astype(np.float32)
        y_train = train_fold[target].astype(np.float32)
        X_val = val_fold[engineered_features].astype(np.float32)
        y_val = val_fold[target].astype(np.float32)

        # Free up memory from original dataframes
        del train_fold, val_fold
        gc.collect()

        # Initialize model for this fold
        model = lgb.LGBMRegressor(**model_params)

        # Train with patient early stopping
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["mae"],  # Simplified metrics
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=50),
            ],
            init_model=None,
        )

        # Free up memory
        del X_train, y_train, X_val, y_val
        gc.collect()

        # Save best iteration for this fold
        print(f"Fold {fold + 1} - Best iteration: {model.best_iteration_}")
        val_scores.append(model.best_score_["valid_0"]["l1"])

        # Add model to our collection
        models.append(model)

    # Calculate and print average validation score
    avg_val_score = sum(val_scores) / len(val_scores)
    print(f"Average validation MAE across folds: {avg_val_score:.6f}")
    post_discord(f"📊 Cross-validation completed for {NUMERAI_MODEL_ID}. Average MAE: {avg_val_score:.6f}")

    # Train a final model with more iterations and without early stopping
    post_discord(f"🎯 Training final model for {NUMERAI_MODEL_ID}...")
    print("Training final model on all data...")
    # For the final model, we'll use a slightly different configuration
    final_model_params = model_params.copy()
    final_model_params["n_estimators"] = 300  # 少ないイテレーション回数で早く収束
    final_model = lgb.LGBMRegressor(**final_model_params)

    # Memory optimization: Convert to 32-bit floats
    X_train_final = train_engineered[engineered_features].astype(np.float32)
    y_train_final = train_engineered[target].astype(np.float32)
    X_val_final = validation_engineered[engineered_features].astype(np.float32)
    y_val_final = validation_engineered[target].astype(np.float32)

    # Free memory
    del train_engineered, validation_engineered
    gc.collect()

    # Train without early stopping on all data
    final_model.fit(
        X_train_final,
        y_train_final,
        eval_set=[(X_val_final, y_val_final)],
        eval_metric=["mae"],
        callbacks=[
            lgb.log_evaluation(period=50),
        ],
    )

    # Free memory
    del X_train_final, y_train_final, X_val_final, y_val_final
    gc.collect()

    # For prediction, we'll use both the cross-validated models and the final model
    # This ensemble approach should improve robustness

    # Predict - use polars lazy mode to load and process live data
    live_features_lazy = pl.scan_parquet("v5.0/live.parquet").select(features)
    # Apply feature engineering to live data (engineer_features handles the lazy-to-pandas conversion)
    live_features_engineered = engineer_features(live_features_lazy)

    def predict(live_features_pd: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the trained model.
        Uses the same feature engineering that was applied during training.
        """
        # Get the feature engineering state from training (captured in closure)
        if "feature_engineering_state" not in globals():
            print("Warning: feature_engineering_state not found. Using fallback approach.")
            # Fallback: use original features only
            live_features_for_prediction = live_features_pd[features]
        else:
            # Use the same feature engineering state as during training
            state = feature_engineering_state
            engineered_features = state["engineered_features"]
            neutralization_features = state["neutralization_features"]

            # Apply the same feature engineering as during training
            try:
                # Use polars lazy mode for feature engineering
                live_features_pl = pl.from_pandas(live_features_pd).lazy()
                live_features_eng = engineer_features(live_features_pl)
            except Exception as e:
                # Fallback to regular pandas if polars conversion fails
                print(f"Warning: Polars conversion failed: {e}. Using pandas instead.")
                live_features_eng = engineer_features(live_features_pd)

            # Memory cleanup
            gc.collect()

            # Make sure all required engineered features exist
            # Add any missing engineered features with zeros
            for feat in engineered_features:
                if feat not in live_features_eng.columns:
                    print(f"Warning: Feature {feat} missing in live data. Adding with zeros.")
                    live_features_eng[feat] = 0.0

            # Ensure features are in the same order as they were during training
            live_features_for_prediction = live_features_eng[engineered_features]

        # Generate predictions from each of our cross-validation models in chunks if necessary
        batch_size = 100000  # Process in batches to reduce memory usage
        if len(live_features_for_prediction) > batch_size:
            print(f"Processing predictions in batches of {batch_size} rows to save memory")

            # Initialize arrays for results
            final_ensemble_predictions = np.zeros(len(live_features_for_prediction))

            # Process in batches
            for i in range(0, len(live_features_for_prediction), batch_size):
                end_idx = min(i + batch_size, len(live_features_for_prediction))
                batch = live_features_for_prediction.iloc[i:end_idx]

                # Get predictions from each model
                batch_predictions = []
                for fold_model in models:
                    fold_preds = fold_model.predict(batch)
                    batch_predictions.append(fold_preds)

                # Add final model predictions
                batch_predictions.append(final_model.predict(batch))

                # Average predictions for this batch
                batch_ensemble = np.mean(batch_predictions, axis=0)

                # Store in the final array
                final_ensemble_predictions[i:end_idx] = batch_ensemble

                # Clean up memory
                del batch, batch_predictions
                gc.collect()

            ensemble_predictions = final_ensemble_predictions
        else:
            # For smaller datasets, process all at once
            cv_predictions = []
            for fold_model in models:
                fold_preds = fold_model.predict(live_features_for_prediction)
                cv_predictions.append(fold_preds)

            # Get predictions from final model
            final_preds = final_model.predict(live_features_for_prediction)

            # Create an ensemble by averaging all model predictions
            all_predictions = cv_predictions + [final_preds]
            ensemble_predictions = np.mean(all_predictions, axis=0)

        # Convert to DataFrame with correct format
        predictions_df = pd.DataFrame(index=live_features_pd.index)
        predictions_df["prediction"] = ensemble_predictions

        # Clean up to free memory before neutralization
        if "feature_engineering_state" in globals():
            del live_features_eng, live_features_for_prediction
        gc.collect()

        # Apply feature neutralization with increased proportion
        # This helps reduce exposure to common risk factors
        if "feature_engineering_state" in globals():
            # Check if neutralization features exist in live data
            available_neutralization_features = [f for f in neutralization_features if f in live_features_pd.columns]
            if len(available_neutralization_features) < len(neutralization_features):
                print(
                    f"Warning: Only {len(available_neutralization_features)}/{len(neutralization_features)} neutralization features available."
                )

            if len(available_neutralization_features) > 0:
                neutralized_predictions = neutralize(
                    df=pd.concat([live_features_pd[available_neutralization_features], predictions_df], axis=1),
                    columns=["prediction"],
                    by=available_neutralization_features,
                    proportion=0.6,  # Increased neutralization (60%)
                )
            else:
                print("Warning: No neutralization features available. Skipping neutralization.")
                neutralized_predictions = predictions_df
        else:
            # Fallback: no neutralization
            neutralized_predictions = predictions_df

        # Clean up to free memory
        del predictions_df
        gc.collect()

        # Scale predictions to match example predictions distribution
        preds_mean = neutralized_predictions["prediction"].mean()
        preds_std = neutralized_predictions["prediction"].std()
        example_mean = example_preds["prediction"].mean()
        example_std = example_preds["prediction"].std()

        # Standardize then rescale to match example distribution
        neutralized_predictions["prediction"] = (
            neutralized_predictions["prediction"] - preds_mean
        ) / preds_std * example_std + example_mean

        # Apply rank-based transformation to ensure robustness
        # This method is more stable than sigmoid for financial data
        def rank_transform(x, min_val=0.001, max_val=0.999):
            # Convert to ranks (0 to 1)
            ranks = pd.Series(x).rank(pct=True)
            # Scale to desired range
            scaled = ranks * (max_val - min_val) + min_val
            return scaled.values

        # Apply rank transformation
        rank_preds = rank_transform(neutralized_predictions["prediction"])

        # Clean up to free memory
        del neutralized_predictions
        gc.collect()

        # Final predictions
        final_predictions = pd.DataFrame(index=live_features_pd.index)
        final_predictions["prediction"] = rank_preds

        # Ensure predictions are strictly between 0 and 1
        # Clip any potential extreme values just to be safe
        final_predictions["prediction"] = final_predictions["prediction"].clip(0.001, 0.999)

        # Ensure the predictions match the expected format
        return final_predictions

    # Create a self-contained predict function that includes all necessary information
    def create_predict_function():
        """Create a self-contained predict function with all necessary state"""
        # Capture the current state in the closure
        captured_state = feature_engineering_state.copy()
        captured_models = models.copy()
        captured_final_model = final_model
        captured_example_preds = example_preds.copy()

        def self_contained_predict(live_features_pd: pd.DataFrame) -> pd.DataFrame:
            """Self-contained predict function that works on any server"""
            # Use captured state instead of global variables
            engineered_features = captured_state["engineered_features"]
            neutralization_features = captured_state["neutralization_features"]
            original_features = captured_state["features"]

            # Apply the same feature engineering as during training
            try:
                # Use polars lazy mode for feature engineering
                live_features_pl = pl.from_pandas(live_features_pd).lazy()
                live_features_eng = engineer_features(live_features_pl)
            except Exception as e:
                # Fallback to regular pandas if polars conversion fails
                print(f"Warning: Polars conversion failed: {e}. Using pandas instead.")
                live_features_eng = engineer_features(live_features_pd)

            # Memory cleanup
            gc.collect()

            # Make sure all required engineered features exist
            # Add any missing engineered features with zeros
            for feat in engineered_features:
                if feat not in live_features_eng.columns:
                    print(f"Warning: Feature {feat} missing in live data. Adding with zeros.")
                    live_features_eng[feat] = 0.0

            # Ensure features are in the same order as they were during training
            live_features_for_prediction = live_features_eng[engineered_features]

            # Generate predictions from each of our cross-validation models
            cv_predictions = []
            for fold_model in captured_models:
                fold_preds = fold_model.predict(live_features_for_prediction)
                cv_predictions.append(fold_preds)

            # Get predictions from final model
            final_preds = captured_final_model.predict(live_features_for_prediction)

            # Create an ensemble by averaging all model predictions
            all_predictions = cv_predictions + [final_preds]
            ensemble_predictions = np.mean(all_predictions, axis=0)

            # Convert to DataFrame with correct format
            predictions_df = pd.DataFrame(index=live_features_pd.index)
            predictions_df["prediction"] = ensemble_predictions

            # Clean up to free memory before neutralization
            del live_features_eng, live_features_for_prediction
            gc.collect()

            # Apply feature neutralization
            available_neutralization_features = [f for f in neutralization_features if f in live_features_pd.columns]
            if len(available_neutralization_features) > 0:
                neutralized_predictions = neutralize(
                    df=pd.concat([live_features_pd[available_neutralization_features], predictions_df], axis=1),
                    columns=["prediction"],
                    by=available_neutralization_features,
                    proportion=0.6,
                )
            else:
                neutralized_predictions = predictions_df

            # Clean up to free memory
            del predictions_df
            gc.collect()

            # Scale predictions to match example predictions distribution
            preds_mean = neutralized_predictions["prediction"].mean()
            preds_std = neutralized_predictions["prediction"].std()
            example_mean = captured_example_preds["prediction"].mean()
            example_std = captured_example_preds["prediction"].std()

            # Standardize then rescale to match example distribution
            neutralized_predictions["prediction"] = (
                neutralized_predictions["prediction"] - preds_mean
            ) / preds_std * example_std + example_mean

            # Apply rank-based transformation
            def rank_transform(x, min_val=0.001, max_val=0.999):
                ranks = pd.Series(x).rank(pct=True)
                scaled = ranks * (max_val - min_val) + min_val
                return scaled.values

            # Apply rank transformation
            rank_preds = rank_transform(neutralized_predictions["prediction"])

            # Clean up to free memory
            del neutralized_predictions
            gc.collect()

            # Final predictions
            final_predictions = pd.DataFrame(index=live_features_pd.index)
            final_predictions["prediction"] = rank_preds

            # Ensure predictions are strictly between 0 and 1
            final_predictions["prediction"] = final_predictions["prediction"].clip(0.001, 0.999)

            return final_predictions

        return self_contained_predict

    # Use the self-contained predict function
    predict = create_predict_function()

    post_discord(f"💾 Saving model for {NUMERAI_MODEL_ID}...")
    p = cloudpickle.dumps(predict)
    with open(model_file_name, "wb") as f:
        f.write(p)

    print(f"Model saved to {model_file_name}")

    # Upload to Cloud Storage
    post_discord(f"☁️ Uploading model to GCS for {NUMERAI_MODEL_ID}...")
    gcs_file_name = f"predict_{NUMERAI_MODEL_ID}.pkl"
    blob = bucket.blob(gcs_file_name)
    blob.upload_from_filename(model_file_name)
    print(f"Model uploaded to GCS: {gcs_file_name}")

    # Send success notification
    success_message = f"✅ **{NUMERAI_MODEL_ID}** training completed successfully!\n📊 Average CV MAE: {avg_val_score:.6f}\n💾 Model saved and uploaded to GCS"
    post_discord(success_message)
    print(success_message)

except Exception:
    # Get detailed error information
    error_details = get_error_details()

    # Create detailed error message
    error_message = f"""❌ **{NUMERAI_MODEL_ID}** training failed!

**Error Type:** {error_details["type"]}
**Error Message:** {error_details["message"]}
**Error Location:** {error_details["location"]}

**Full Traceback:**
```
{error_details["full_traceback"]}
```"""

    # Send error notification to Discord
    post_discord(error_message)

    # Also print to console
    print(f"Error occurred: {error_details['type']}: {error_details['message']}")
    print(f"Location: {error_details['location']}")
    print(f"Full traceback:\n{error_details['full_traceback']}")

    # Re-raise the exception to maintain the original error handling
    raise
