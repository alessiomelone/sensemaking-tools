# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This module contains an implementation of the Community Notes matrix factorization
# algorithm, based on the paper "Notes on Notes: How Community Notes' Algorithm Works"
# by Christianْل Bayer, Thomasْل Métais, and J. Williams (2023).

import tensorflow as tf
from dataclasses import dataclass
from typing import List, Union, Tuple, Sequence # Sequence for trainable_variables

# Regarding TensorFlow CPU/GPU selection:
# In Python, TensorFlow automatically attempts to use a GPU if one is available
# and correctly configured (CUDA, cuDNN installed).
# Users can control GPU visibility using tf.config.set_visible_devices([], 'GPU')
# to force CPU, or by setting the CUDA_VISIBLE_DEVICES environment variable.
# The dynamic backend loading from TFJS (loadTFJS function) is not directly
# replicated here as Python's TensorFlow setup handles this differently.
print(f"TensorFlow version: {tf.__version__}")
physical_devices_gpu = tf.config.list_physical_devices('GPU')
if physical_devices_gpu:
    print(f"TensorFlow is using GPU: {physical_devices_gpu}")
else:
    print("TensorFlow is using CPU.")


@dataclass
class Rating:
    user_id: int # Changed from userId to user_id for Pythonic convention
    note_id: int # Changed from noteId to note_id
    rating: float # Ratings can be floats (e.g. normalized)


async def community_notes_matrix_factorization(
    ratings: List[Rating],
    num_factors: int = 1,
    epochs: int = 400,
    learning_rate: Union[float, List[float]] = [0.05, 0.01, 0.002, 0.0004],
    lambda_i_reg: float = 0.15, # Renamed from lambdaI for Python convention
    lambda_f_reg: float = 0.03, # Renamed from lambdaF
) -> List[float]:
    """
    Given ratings, return helpfulness scores and other model parameters for the given set of ratings.
    Args:
        ratings: A collection of Rating values.
        num_factors: The factor dimensionality.
        epochs: Number of training iterations to run per learningRate.
        learning_rate: Either a single learning rate value, or an array of values for a learning rate schedule.
        lambda_i_reg: Intercept term regularization parameter.
        lambda_f_reg: Factor term regularization parameter.
    Returns:
        Helpfulness scores (note intercepts).
    """
    if not ratings:
        return []

    num_users = max(r.user_id for r in ratings) + 1
    num_notes = max(r.note_id for r in ratings) + 1

    # Initialize parameters randomly. Using tf.Variable allows us to update them during training.
    mu = tf.Variable(0.0, dtype=tf.float32, name="mu")
    user_intercepts = tf.Variable(tf.random.normal([num_users, 1]), name="user_intercepts")
    note_intercepts = tf.Variable(tf.random.normal([num_notes, 1]), name="note_intercepts")
    user_factors = tf.Variable(tf.random.normal([num_users, num_factors]), name="user_factors")
    note_factors = tf.Variable(tf.random.normal([num_notes, num_factors]), name="note_factors")

    trainable_variables: Sequence[tf.Variable] = [
        mu, user_intercepts, note_intercepts, user_factors, note_factors
    ]

    learning_rates = [learning_rate] if isinstance(learning_rate, (int, float)) else learning_rate

    # Convert ratings to tensors for efficient computation.
    user_ids_tensor = tf.constant([r.user_id for r in ratings], dtype=tf.int32)
    note_ids_tensor = tf.constant([r.note_id for r in ratings], dtype=tf.int32)
    # For the math to work out, this should be a column vector, hence expand dims
    rating_values_tensor = tf.expand_dims(tf.constant([r.rating for r in ratings], dtype=tf.float32), axis=1)
    
    lambda_i_tensor = tf.constant(lambda_i_reg, dtype=tf.float32)
    lambda_f_tensor = tf.constant(lambda_f_reg, dtype=tf.float32)

    # @tf.function # Optionally compile for performance
    def calculate_loss() -> tf.Tensor:
        # Gather relevant intercepts and factors for the observed ratings
        gathered_user_intercepts = tf.gather(user_intercepts, user_ids_tensor)
        gathered_note_intercepts = tf.gather(note_intercepts, note_ids_tensor)
        gathered_user_factors = tf.gather(user_factors, user_ids_tensor)
        gathered_note_factors = tf.gather(note_factors, note_ids_tensor)
        
        mu_squared = tf.square(mu)

        # Dot product of user and note factors for each rating
        # tf.reduce_sum(tf.multiply(A, B), axis=1) is equivalent to row-wise dot product
        factor_dot_product = tf.expand_dims(
            tf.reduce_sum(tf.multiply(gathered_user_factors, gathered_note_factors), axis=1),
            axis=1
        )

        # Compute predicted ratings: r_hat = mu + user_intercept + note_intercept + user_factor * note_factor
        predicted_ratings = mu + gathered_user_intercepts + gathered_note_intercepts + factor_dot_product

        # Compute squared errors
        squared_error = tf.square(rating_values_tensor - predicted_ratings)

        # Intercept regularization terms
        # lambda_i * (sum(user_intercepts^2) + sum(note_intercepts^2) + mu^2)
        # The paper implies regularization over all parameters, not just the ones in the batch.
        # However, the TFJS code regularizes the gathered (batch) terms. Replicating TFJS:
        # intercept_regularization = lambda_i_tensor * (
        #     tf.reduce_sum(tf.square(gathered_user_intercepts)) + # This is different from paper but matches code
        #     tf.reduce_sum(tf.square(gathered_note_intercepts)) + # This is different from paper but matches code
        #     mu_squared # mu_squared is already a scalar, sum not needed if we mean global mu
        # )
        # Corrected interpretation based on typical MF regularization (sum over all parameters):
        intercept_reg_loss = lambda_i_tensor * (
            tf.reduce_sum(tf.square(user_intercepts)) + 
            tf.reduce_sum(tf.square(note_intercepts)) + 
            mu_squared
        )


        # Factor regularization terms
        # lambda_f * (sum(||user_factors||^2) + sum(||note_factors||^2))
        # The TFJS code `euclideanNorm(1).square()` seems to be ||factor_vector||^2 for each user/note in batch.
        # factor_regularization = lambda_f_tensor * (
        #     tf.reduce_sum(tf.square(tf.norm(gathered_user_factors, axis=1))) + # Norms of factors in batch
        #     tf.reduce_sum(tf.square(tf.norm(gathered_note_factors, axis=1)))   # Norms of factors in batch
        # )
        # Corrected interpretation for sum over all factor vectors:
        factor_reg_loss = lambda_f_tensor * (
            tf.reduce_sum(tf.square(tf.norm(user_factors, axis=1))) +
            tf.reduce_sum(tf.square(tf.norm(note_factors, axis=1)))
        )
        
        # Total loss: sum of squared errors for observed ratings + regularization terms for all params
        # The original TFJS code sums squaredError, interceptRegularization, and factorRegularization
        # where interceptRegularization and factorRegularization were calculated based on the *gathered* terms.
        # This means it was regularizing each term *as it appeared in a rating*.
        # This is unusual. Standard regularization is on the parameters themselves, not their occurrences.
        # Let's stick to the TFJS implementation's apparent logic for direct translation first.
        
        # Re-implementing TFJS-like regularization (on gathered terms):
        tfjs_style_intercept_reg = lambda_i_tensor * (
             tf.square(gathered_user_intercepts) + tf.square(gathered_note_intercepts) # These are element-wise per rating
        )
        # mu_squared is global, add it once per rating or once globally? TFJS adds it to ratingUInts.square() etc.
        # This implies mu_squared is added for each rating in the sum.
        tfjs_style_intercept_reg_mu_part = lambda_i_tensor * mu_squared # This will be broadcasted

        # TFJS: ratingUFactors.euclideanNorm(1).square() -> sum of squares of elements in each factor vector, then square that sum?
        # No, euclideanNorm(1) would be sum(abs(x_i)). Then square. This is L1 norm squared.
        # TFJS `euclideanNorm` without keepdims=true on a matrix (numSamples, numFactors) with axis=1
        # calculates norm for each row. So it's sum_j(x_ij^2) for each i. This is squared L2 norm.
        # So `tf.norm(gathered_user_factors, axis=1)` gives L2 norm. Then square it.
        # So `tf.reduce_sum(tf.square(gathered_user_factors), axis=1, keepdims=True)` is sum_j(x_ij^2)
        
        # This is sum(v_i^2) for each factor vector v.
        sum_sq_user_factors_per_rating = tf.reduce_sum(tf.square(gathered_user_factors), axis=1, keepdims=True)
        sum_sq_note_factors_per_rating = tf.reduce_sum(tf.square(gathered_note_factors), axis=1, keepdims=True)

        tfjs_style_factor_reg = lambda_f_tensor * (
            sum_sq_user_factors_per_rating + sum_sq_note_factors_per_rating
        )
        
        # Summing up all components for each rating, then sum over all ratings
        loss_per_rating = squared_error + tfjs_style_intercept_reg + tfjs_style_factor_reg + tfjs_style_intercept_reg_mu_part
        total_loss = tf.reduce_sum(loss_per_rating)
        return total_loss

    # Training loop
    for rate_idx, lr_val in enumerate(learning_rates):
        print(f"Setting learning rate to: {lr_val} (Stage {rate_idx+1}/{len(learning_rates)})")
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_val)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                current_loss = calculate_loss()
            
            gradients = tape.gradient(current_loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))
            
            if (epoch + 1) % 10 == 0:
                # .numpy() is synchronous. No await needed.
                loss_value = current_loss.numpy()
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_value:.4f}")
    
    # Extract note intercepts (helpfulness scores).
    # .numpy() converts tensor to NumPy array. Convert to list for return type.
    helpfulness_scores = note_intercepts.numpy().flatten().tolist()
    return helpfulness_scores

# Example Usage (optional, for testing)
async def main_example():
    print("Running Matrix Factorization Example...")
    sample_ratings = [
        Rating(user_id=0, note_id=0, rating=1.0),
        Rating(user_id=0, note_id=1, rating=-1.0),
        Rating(user_id=1, note_id=0, rating=1.0),
        Rating(user_id=1, note_id=1, rating=1.0),
        Rating(user_id=2, note_id=0, rating=-1.0),
        Rating(user_id=2, note_id=1, rating=1.0),
        Rating(user_id=0, note_id=2, rating=0.0),
        Rating(user_id=1, note_id=2, rating=1.0),
    ]
    
    # Ensure TF is eager for easier debugging if needed, though default in TF2
    # tf.config.run_functions_eagerly(True)

    helpfulness = await community_notes_matrix_factorization(
        sample_ratings,
        num_factors=1,
        epochs=50, # Reduced for quick example
        learning_rate=[0.01, 0.005], # Reduced schedule
        lambda_i_reg=0.1,
        lambda_f_reg=0.02
    )
    print("Helpfulness Scores (Note Intercepts):")
    for i, score in enumerate(helpfulness):
        print(f"Note {i}: {score:.4f}")

if __name__ == '__main__':
    # To run the async main_example:
    import asyncio
    asyncio.run(main_example())
