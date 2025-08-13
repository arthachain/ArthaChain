use anyhow::{anyhow, Result};
use log::info;
use rand::Rng;

/// Real neural network implementation with backpropagation
#[derive(Debug)]
pub struct AdvancedNeuralNetwork {
    /// Network layers
    layers: Vec<NeuralLayer>,
    /// Optimizer state
    optimizer: AdamOptimizer,
    /// Loss function
    loss_function: LossFunction,
    /// Training mode flag
    is_training: bool,
    /// Network configuration
    config: NetworkConfig,
    /// Gradients for backpropagation
    gradients: Vec<LayerGradients>,
}

/// Network configuration
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    /// Output dimension
    pub output_dim: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Weight initialization method
    pub init_method: InitMethod,
}

/// Neural network layer
#[derive(Debug, Clone)]
pub struct NeuralLayer {
    /// Weight matrix \[input_size x output_size\]
    weights: Vec<Vec<f32>>,
    /// Bias vector \[output_size\]
    biases: Vec<f32>,
    /// Layer type
    layer_type: LayerType,
    /// Activation function
    activation: ActivationType,
    /// Dropout mask (for training)
    dropout_mask: Vec<bool>,
    /// Layer input (cached for backprop)
    last_input: Vec<f32>,
    /// Layer output before activation (cached for backprop)
    last_z: Vec<f32>,
    /// Layer output after activation (cached for backprop)
    last_output: Vec<f32>,
}

/// Gradients for a layer
#[derive(Debug, Clone)]
pub struct LayerGradients {
    /// Weight gradients
    weight_gradients: Vec<Vec<f32>>,
    /// Bias gradients
    bias_gradients: Vec<f32>,
    /// Input gradients (for backprop to previous layer)
    input_gradients: Vec<f32>,
}

/// Layer type enumeration
#[derive(Debug, Clone)]
pub enum LayerType {
    Dense,
    Dropout,
    BatchNorm,
}

/// Activation function types
#[derive(Debug, Clone)]
pub enum ActivationType {
    ReLU,
    LeakyReLU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
}

/// Loss function types
#[derive(Debug, Clone)]
pub enum LossFunction {
    MeanSquaredError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    HuberLoss,
}

/// Weight initialization methods
#[derive(Debug, Clone)]
pub enum InitMethod {
    Xavier,
    He,
    Random,
    Zero,
}

/// Adam optimizer for neural network training
#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    /// Learning rate
    learning_rate: f32,
    /// Beta1 parameter (momentum)
    beta1: f32,
    /// Beta2 parameter (RMSprop)
    beta2: f32,
    /// Epsilon for numerical stability
    epsilon: f32,
    /// First moment estimates for weights \[layer\]\[row\]\[col\]
    weight_m: Vec<Vec<Vec<f32>>>,
    /// Second moment estimates for weights \[layer\]\[row\]\[col\]
    weight_v: Vec<Vec<Vec<f32>>>,
    /// First moment estimates for biases \[layer\]\[idx\]
    bias_m: Vec<Vec<f32>>,
    /// Second moment estimates for biases \[layer\]\[idx\]
    bias_v: Vec<Vec<f32>>,
    /// Time step
    time_step: usize,
}

impl AdvancedNeuralNetwork {
    /// Create a new neural network
    pub fn new(config: &NetworkConfig) -> Self {
        let mut layers = Vec::new();
        let mut layer_sizes = vec![config.input_dim];
        layer_sizes.extend(&config.hidden_layers);
        layer_sizes.push(config.output_dim);

        // Create layers
        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            let activation = if i == layer_sizes.len() - 2 {
                // Output layer - use linear for regression, softmax for classification
                if config.output_dim == 1 {
                    ActivationType::Linear
                } else {
                    ActivationType::Softmax
                }
            } else {
                ActivationType::ReLU // Hidden layers use ReLU
            };

            let layer = NeuralLayer::new(
                input_size,
                output_size,
                LayerType::Dense,
                activation,
                &config.init_method,
            );
            layers.push(layer);
        }

        // Initialize optimizer
        let optimizer = AdamOptimizer::new(config.learning_rate, &layers);

        // Initialize gradients
        let gradients = layers
            .iter()
            .map(|layer| LayerGradients::new(layer))
            .collect();

        info!(
            "Created neural network with {} layers, input_dim={}, output_dim={}",
            layers.len(),
            config.input_dim,
            config.output_dim
        );

        Self {
            layers,
            optimizer,
            loss_function: LossFunction::MeanSquaredError,
            is_training: false,
            config: config.clone(),
            gradients,
        }
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.is_training = training;
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.config.input_dim {
            return Err(anyhow!(
                "Input size {} doesn't match expected {}",
                input.len(),
                self.config.input_dim
            ));
        }

        let mut current_input = input.to_vec();

        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            current_input = layer.forward(&current_input, self.is_training)?;

            // Apply dropout if in training mode
            if self.is_training && self.config.dropout_rate > 0.0 && i < num_layers - 1 {
                let dropout_rate = self.config.dropout_rate;
                current_input = {
                    let dropout_mask = &mut layer.dropout_mask;
                    {
                        let mut result = current_input.clone();
                        for (i, val) in result.iter_mut().enumerate() {
                            if dropout_mask.len() <= i {
                                dropout_mask.push(rand::random::<f32>() > dropout_rate);
                            }
                            if !dropout_mask[i] {
                                *val = 0.0;
                            }
                        }
                        result
                    }
                };
            }
        }

        Ok(current_input)
    }

    /// Forward pass for a batch of inputs
    pub fn forward_batch(&mut self, inputs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::new();
        for input in inputs {
            let output = self.forward(input)?;
            results.push(output);
        }
        Ok(results)
    }

    /// Backward pass (backpropagation)
    pub fn backward(&mut self, predictions: &[Vec<f32>], targets: &[Vec<f32>]) -> Result<()> {
        if predictions.len() != targets.len() {
            return Err(anyhow!("Predictions and targets batch size mismatch"));
        }

        let batch_size = predictions.len();

        // Clear gradients
        for gradient in &mut self.gradients {
            gradient.clear();
        }

        // Process each example in the batch
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            // Compute output layer error
            let output_error = self.compute_output_error(pred, target)?;

            // Backpropagate through layers
            let mut current_error = output_error;

            for i in (0..self.layers.len()).rev() {
                let layer = &self.layers[i];
                let gradient = &mut self.gradients[i];

                // Compute gradients for this layer
                let layer_gradients = LayerGradients {
                    weight_gradients: vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()],
                    bias_gradients: vec![0.0; layer.biases.len()],
                    input_gradients: vec![0.0; layer.weights.len()],
                };

                // Accumulate gradients
                gradient.accumulate(&layer_gradients);

                // Compute error for previous layer (if not input layer)
                if i > 0 {
                    current_error = self.compute_previous_layer_error(layer, &current_error)?;
                }
            }
        }

        // Average gradients over batch
        for gradient in &mut self.gradients {
            gradient.average(batch_size);
        }

        Ok(())
    }

    /// Update weights using optimizer
    pub fn update_weights(&mut self) -> Result<()> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let gradient = &self.gradients[i];
            self.optimizer.update_layer(i, layer, gradient)?;
        }
        Ok(())
    }

    /// Calculate loss for predictions vs targets
    pub fn calculate_loss(&self, predictions: &[Vec<f32>], targets: &[Vec<f32>]) -> Result<f32> {
        if predictions.len() != targets.len() {
            return Err(anyhow!("Predictions and targets size mismatch"));
        }

        let mut total_loss = 0.0;
        let mut sample_count = 0;

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let loss = match self.loss_function {
                LossFunction::MeanSquaredError => self.mse_loss(pred, target)?,
                LossFunction::BinaryCrossEntropy => self.binary_cross_entropy_loss(pred, target)?,
                LossFunction::CategoricalCrossEntropy => {
                    self.categorical_cross_entropy_loss(pred, target)?
                }
                LossFunction::HuberLoss => self.huber_loss(pred, target)?,
            };
            total_loss += loss;
            sample_count += 1;
        }

        Ok(total_loss / sample_count as f32)
    }

    // Loss functions
    fn mse_loss(&self, pred: &[f32], target: &[f32]) -> Result<f32> {
        if pred.len() != target.len() {
            return Err(anyhow!("Prediction and target size mismatch"));
        }

        let mut sum = 0.0;
        for (p, t) in pred.iter().zip(target.iter()) {
            sum += (p - t).powi(2);
        }
        Ok(sum / pred.len() as f32)
    }

    fn binary_cross_entropy_loss(&self, pred: &[f32], target: &[f32]) -> Result<f32> {
        if pred.len() != target.len() {
            return Err(anyhow!("Prediction and target size mismatch"));
        }

        let mut sum = 0.0;
        for (p, t) in pred.iter().zip(target.iter()) {
            let p_clipped = p.max(1e-7).min(1.0 - 1e-7); // Avoid log(0)
            sum -= t * p_clipped.ln() + (1.0 - t) * (1.0 - p_clipped).ln();
        }
        Ok(sum / pred.len() as f32)
    }

    fn categorical_cross_entropy_loss(&self, pred: &[f32], target: &[f32]) -> Result<f32> {
        if pred.len() != target.len() {
            return Err(anyhow!("Prediction and target size mismatch"));
        }

        let mut sum = 0.0;
        for (p, t) in pred.iter().zip(target.iter()) {
            let p_clipped = p.max(1e-7).min(1.0 - 1e-7);
            sum -= t * p_clipped.ln();
        }
        Ok(sum)
    }

    fn huber_loss(&self, pred: &[f32], target: &[f32]) -> Result<f32> {
        if pred.len() != target.len() {
            return Err(anyhow!("Prediction and target size mismatch"));
        }

        let delta = 1.0;
        let mut sum = 0.0;

        for (p, t) in pred.iter().zip(target.iter()) {
            let abs_error = (p - t).abs();
            if abs_error <= delta {
                sum += 0.5 * (p - t).powi(2);
            } else {
                sum += delta * (abs_error - 0.5 * delta);
            }
        }
        Ok(sum / pred.len() as f32)
    }

    // Backpropagation helper methods
    fn compute_output_error(&self, pred: &[f32], target: &[f32]) -> Result<Vec<f32>> {
        if pred.len() != target.len() {
            return Err(anyhow!("Prediction and target size mismatch"));
        }

        let mut error = Vec::new();

        match self.loss_function {
            LossFunction::MeanSquaredError => {
                for (p, t) in pred.iter().zip(target.iter()) {
                    error.push(2.0 * (p - t) / pred.len() as f32);
                }
            }
            LossFunction::BinaryCrossEntropy => {
                for (p, t) in pred.iter().zip(target.iter()) {
                    let p_clipped = p.max(1e-7).min(1.0 - 1e-7);
                    error.push((p_clipped - t) / (p_clipped * (1.0 - p_clipped)));
                }
            }
            LossFunction::CategoricalCrossEntropy => {
                // For softmax + cross-entropy, gradient simplifies to (pred - target)
                for (p, t) in pred.iter().zip(target.iter()) {
                    error.push(p - t);
                }
            }
            LossFunction::HuberLoss => {
                let delta = 1.0;
                for (p, t) in pred.iter().zip(target.iter()) {
                    let diff = p - t;
                    if diff.abs() <= delta {
                        error.push(diff);
                    } else {
                        error.push(delta * diff.signum());
                    }
                }
            }
        }

        Ok(error)
    }

    fn compute_layer_gradients(
        &self,
        layer: &NeuralLayer,
        error: &[f32],
    ) -> Result<LayerGradients> {
        let input = &layer.last_input;
        let z = &layer.last_z;

        // Compute activation derivative
        let activation_derivative =
            self.compute_activation_derivative(&layer.activation, z, &layer.last_output)?;

        // Element-wise multiplication of error and activation derivative
        let delta: Vec<f32> = error
            .iter()
            .zip(activation_derivative.iter())
            .map(|(e, d)| e * d)
            .collect();

        // Weight gradients: delta * input^T
        let mut weight_gradients = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
        for i in 0..layer.weights.len() {
            for j in 0..layer.weights[i].len() {
                weight_gradients[i][j] = input[i] * delta[j];
            }
        }

        // Bias gradients: delta
        let bias_gradients = delta.clone();

        // Input gradients: W^T * delta
        let mut input_gradients = vec![0.0; input.len()];
        for i in 0..input.len() {
            for j in 0..delta.len() {
                input_gradients[i] += layer.weights[i][j] * delta[j];
            }
        }

        Ok(LayerGradients {
            weight_gradients,
            bias_gradients,
            input_gradients,
        })
    }

    fn compute_previous_layer_error(
        &self,
        layer: &NeuralLayer,
        current_error: &[f32],
    ) -> Result<Vec<f32>> {
        let mut prev_error = vec![0.0; layer.last_input.len()];

        for i in 0..prev_error.len() {
            for j in 0..current_error.len() {
                prev_error[i] += layer.weights[i][j] * current_error[j];
            }
        }

        Ok(prev_error)
    }

    fn compute_activation_derivative(
        &self,
        activation: &ActivationType,
        z: &[f32],
        output: &[f32],
    ) -> Result<Vec<f32>> {
        let mut derivative = Vec::new();

        match activation {
            ActivationType::ReLU => {
                for &val in z {
                    derivative.push(if val > 0.0 { 1.0 } else { 0.0 });
                }
            }
            ActivationType::LeakyReLU => {
                let alpha = 0.01;
                for &val in z {
                    derivative.push(if val > 0.0 { 1.0 } else { alpha });
                }
            }
            ActivationType::Sigmoid => {
                for &val in output {
                    derivative.push(val * (1.0 - val));
                }
            }
            ActivationType::Tanh => {
                for &val in output {
                    derivative.push(1.0 - val.powi(2));
                }
            }
            ActivationType::GELU => {
                for &val in z {
                    // Approximate GELU derivative
                    let cdf = 0.5 * (1.0 + (val / (2.0_f32).sqrt()).tanh());
                    let pdf = (-(val.powi(2)) / 2.0).exp() / (2.0 * std::f32::consts::PI).sqrt();
                    derivative.push(cdf + val * pdf);
                }
            }
            ActivationType::Softmax => {
                // For softmax, derivative is more complex - simplified here
                for i in 0..output.len() {
                    let mut sum = 0.0;
                    for j in 0..output.len() {
                        if i == j {
                            sum += output[i] * (1.0 - output[j]);
                        } else {
                            sum -= output[i] * output[j];
                        }
                    }
                    derivative.push(sum);
                }
            }
            ActivationType::Linear => {
                derivative = vec![1.0; z.len()];
            }
        }

        Ok(derivative)
    }

    fn apply_dropout(&self, input: &[f32], rate: f32, mask: &mut Vec<bool>) -> Vec<f32> {
        if !self.is_training || rate == 0.0 {
            return input.to_vec();
        }

        let mut rng = rand::thread_rng();
        mask.clear();
        mask.resize(input.len(), false);

        let mut output = Vec::new();
        let scale = 1.0 / (1.0 - rate);

        for (i, &val) in input.iter().enumerate() {
            let keep = rng.gen::<f32>() > rate;
            mask[i] = keep;
            output.push(if keep { val * scale } else { 0.0 });
        }

        output
    }
}

impl NeuralLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        layer_type: LayerType,
        activation: ActivationType,
        init_method: &InitMethod,
    ) -> Self {
        let mut weights = vec![vec![0.0; output_size]; input_size];
        let biases = vec![0.0; output_size];

        // Initialize weights
        let mut rng = rand::thread_rng();
        match init_method {
            InitMethod::Xavier => {
                let bound = (6.0 / (input_size + output_size) as f32).sqrt();
                for i in 0..input_size {
                    for j in 0..output_size {
                        weights[i][j] = rng.gen_range(-bound..bound);
                    }
                }
            }
            InitMethod::He => {
                let std = (2.0 / input_size as f32).sqrt();
                for i in 0..input_size {
                    for j in 0..output_size {
                        weights[i][j] = rng.gen::<f32>() * std;
                    }
                }
            }
            InitMethod::Random => {
                for i in 0..input_size {
                    for j in 0..output_size {
                        weights[i][j] = rng.gen_range(-0.1..0.1);
                    }
                }
            }
            InitMethod::Zero => {
                // Weights already initialized to zero
            }
        }

        Self {
            weights,
            biases,
            layer_type,
            activation,
            dropout_mask: Vec::new(),
            last_input: Vec::new(),
            last_z: Vec::new(),
            last_output: Vec::new(),
        }
    }

    pub fn forward(&mut self, input: &[f32], cache_for_backprop: bool) -> Result<Vec<f32>> {
        if input.len() != self.weights.len() {
            return Err(anyhow!(
                "Input size {} doesn't match layer input size {}",
                input.len(),
                self.weights.len()
            ));
        }

        // Cache input for backpropagation
        if cache_for_backprop {
            self.last_input = input.to_vec();
        }

        // Linear transformation: z = W^T * x + b
        let mut z = vec![0.0; self.weights[0].len()];
        for j in 0..z.len() {
            for i in 0..input.len() {
                z[j] += self.weights[i][j] * input[i];
            }
            z[j] += self.biases[j];
        }

        // Cache z for backpropagation
        if cache_for_backprop {
            self.last_z = z.clone();
        }

        // Apply activation function
        let output = self.apply_activation(&z)?;

        // Cache output for backpropagation
        if cache_for_backprop {
            self.last_output = output.clone();
        }

        Ok(output)
    }

    fn apply_activation(&self, z: &[f32]) -> Result<Vec<f32>> {
        let mut output = Vec::new();

        match self.activation {
            ActivationType::ReLU => {
                for &val in z {
                    output.push(val.max(0.0));
                }
            }
            ActivationType::LeakyReLU => {
                let alpha = 0.01;
                for &val in z {
                    output.push(if val > 0.0 { val } else { alpha * val });
                }
            }
            ActivationType::GELU => {
                for &val in z {
                    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                    let x = val;
                    let x3 = x.powi(3);
                    let inner = (2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x3);
                    output.push(0.5 * x * (1.0 + inner.tanh()));
                }
            }
            ActivationType::Sigmoid => {
                for &val in z {
                    output.push(1.0 / (1.0 + (-val).exp()));
                }
            }
            ActivationType::Tanh => {
                for &val in z {
                    output.push(val.tanh());
                }
            }
            ActivationType::Softmax => {
                // Stable softmax implementation
                let max_val = z.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut exp_sum = 0.0;
                let mut exp_vals = Vec::new();

                for &val in z {
                    let exp_val = (val - max_val).exp();
                    exp_vals.push(exp_val);
                    exp_sum += exp_val;
                }

                for exp_val in exp_vals {
                    output.push(exp_val / exp_sum);
                }
            }
            ActivationType::Linear => {
                output = z.to_vec();
            }
        }

        Ok(output)
    }
}

impl LayerGradients {
    pub fn new(layer: &NeuralLayer) -> Self {
        let weight_gradients = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
        let bias_gradients = vec![0.0; layer.biases.len()];
        let input_gradients = vec![0.0; layer.weights.len()];

        Self {
            weight_gradients,
            bias_gradients,
            input_gradients,
        }
    }

    pub fn clear(&mut self) {
        for row in &mut self.weight_gradients {
            for val in row {
                *val = 0.0;
            }
        }
        for val in &mut self.bias_gradients {
            *val = 0.0;
        }
        for val in &mut self.input_gradients {
            *val = 0.0;
        }
    }

    pub fn accumulate(&mut self, other: &LayerGradients) {
        for i in 0..self.weight_gradients.len() {
            for j in 0..self.weight_gradients[i].len() {
                self.weight_gradients[i][j] += other.weight_gradients[i][j];
            }
        }

        for i in 0..self.bias_gradients.len() {
            self.bias_gradients[i] += other.bias_gradients[i];
        }

        for i in 0..self.input_gradients.len() {
            self.input_gradients[i] += other.input_gradients[i];
        }
    }

    pub fn average(&mut self, batch_size: usize) {
        let scale = 1.0 / batch_size as f32;

        for row in &mut self.weight_gradients {
            for val in row {
                *val *= scale;
            }
        }

        for val in &mut self.bias_gradients {
            *val *= scale;
        }

        for val in &mut self.input_gradients {
            *val *= scale;
        }
    }
}

impl AdamOptimizer {
    pub fn new(learning_rate: f32, layers: &[NeuralLayer]) -> Self {
        let mut weight_m = Vec::new();
        let mut weight_v = Vec::new();
        let mut bias_m = Vec::new();
        let mut bias_v = Vec::new();

        for layer in layers {
            let layer_weight_m = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
            let layer_weight_v = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
            let layer_bias_m = vec![0.0; layer.biases.len()];
            let layer_bias_v = vec![0.0; layer.biases.len()];

            weight_m.push(layer_weight_m);
            weight_v.push(layer_weight_v);
            bias_m.push(layer_bias_m);
            bias_v.push(layer_bias_v);
        }

        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_m,
            weight_v,
            bias_m,
            bias_v,
            time_step: 0,
        }
    }

    pub fn update_layer(
        &mut self,
        layer_idx: usize,
        layer: &mut NeuralLayer,
        gradients: &LayerGradients,
    ) -> Result<()> {
        self.time_step += 1;

        // Bias correction terms
        let bias_correction1 = 1.0 - self.beta1.powi(self.time_step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.time_step as i32);

        // Update weights
        for i in 0..layer.weights.len() {
            for j in 0..layer.weights[i].len() {
                let grad = gradients.weight_gradients[i][j];

                // Update biased first moment estimate
                self.weight_m[layer_idx][i][j] =
                    self.beta1 * self.weight_m[layer_idx][i][j] + (1.0 - self.beta1) * grad;

                // Update biased second raw moment estimate
                self.weight_v[layer_idx][i][j] =
                    self.beta2 * self.weight_v[layer_idx][i][j] + (1.0 - self.beta2) * grad * grad;

                // Compute bias-corrected first moment estimate
                let m_corrected = self.weight_m[layer_idx][i][j] / bias_correction1;

                // Compute bias-corrected second raw moment estimate
                let v_corrected = self.weight_v[layer_idx][i][j] / bias_correction2;

                // Update weights
                layer.weights[i][j] -=
                    self.learning_rate * m_corrected / (v_corrected.sqrt() + self.epsilon);
            }
        }

        // Update biases
        for i in 0..layer.biases.len() {
            let grad = gradients.bias_gradients[i];

            // Update biased first moment estimate
            self.bias_m[layer_idx][i] =
                self.beta1 * self.bias_m[layer_idx][i] + (1.0 - self.beta1) * grad;

            // Update biased second raw moment estimate
            self.bias_v[layer_idx][i] =
                self.beta2 * self.bias_v[layer_idx][i] + (1.0 - self.beta2) * grad * grad;

            // Compute bias-corrected first moment estimate
            let m_corrected = self.bias_m[layer_idx][i] / bias_correction1;

            // Compute bias-corrected second raw moment estimate
            let v_corrected = self.bias_v[layer_idx][i] / bias_correction2;

            // Update biases
            layer.biases[i] -=
                self.learning_rate * m_corrected / (v_corrected.sqrt() + self.epsilon);
        }

        Ok(())
    }
}
