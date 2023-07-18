use super::{super::matrix::Matrix, NetTrait};
use super::super::data_set::DataSet;
use std::fmt::{ Debug, Formatter };

const LEARN_FACTOR: f32 = 0.05;
// const REGULATION_FACTOR: f32 = 1e-11;
// const LEARN_REG_BATCH_FACTOR: f32 = 1.0;
const TARGET_ACCURACY: f32 = 0.98;
const FLOATING_WEIGHT: f32 = 0.1;
const VALIDATION_SIZE: usize = 500;

#[derive(Default)]
pub struct NN {
    // General stuff
    layer_lengths: Vec<usize>,
    layer_count: usize,
    last_layer: usize,

    // The magic
    layers: Vec<Matrix>,
    transposed_layers: Vec<Matrix>,
    weights: Vec<Matrix>,
    transposed_weights: Vec<Matrix>,
    bias: Vec<Matrix>,
    errors: Vec<Matrix>,
    pre_activation: Vec<Matrix>,

    // For adjusting weights and gradients
    avg_weight: Vec<Matrix>,
    avg_bias: Vec<Matrix>,
    bias_gradient: Vec<Matrix>,
    weight_gradient: Vec<Matrix>,

    // For Validation
    target_vector: Matrix,
    current_cost: Matrix,
    floating_average: f32,

    // Training Data
    training_data: DataSet,
    validation_data: DataSet,
}

impl Debug for NN {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        let mut fmt_str = String::new();
        fmt_str.push_str(&format!("Layer Count: {}\n", self.layer_count));
        fmt_str.push_str(&format!("Layer lengths: {:?}\n", self.layer_lengths));
        for i in 0..self.layer_count {
            fmt_str.push_str(&format!("\nLayer {}:\n", i));
            let m = &self.layers[i];
            fmt_str.push_str(&format!("Layer: {:?}\n", m));
            let m = &self.transposed_layers[i];
            fmt_str.push_str(&format!("Transposed Layer: {:?}\n", m));
            let m = &self.errors[i];
            fmt_str.push_str(&format!("Errors: {:?}\n", m));
            let m = &self.weights[i];
            fmt_str.push_str(&format!("Weights: {:?}\n", m));
            let m = &self.transposed_weights[i];
            fmt_str.push_str(&format!("Transposed Weights: {:?}\n", m));
            let m = &self.bias[i];
            fmt_str.push_str(&format!("Bias: {:?}\n", m));
            let m = &self.avg_bias[i];
            fmt_str.push_str(&format!("Avg Bias: {:?}\n", m));
            let m = &self.bias_gradient[i];
            fmt_str.push_str(&format!("Gradiant Bias: {:?}\n", m));
            let m = &self.avg_weight[i];
            fmt_str.push_str(&format!("Avg Weight: {:?}\n", m));
            let m = &self.weight_gradient[i];
            fmt_str.push_str(&format!("Gradiant Weight: {:?}\n", m));
        }
        write!(f, "Stats:\n{}", fmt_str)
    }
}

impl NetTrait for NN {
    fn initialize_layers(&mut self, layer_sizes: Vec<usize>) {
        self.layer_lengths = layer_sizes.clone();
        self.layer_count = self.layer_lengths.len();
        self.last_layer = self.layer_count - 1;

        self.layers = vec![Matrix::new(0, 0); self.layer_count];
        self.transposed_layers = vec![Matrix::new(0, 0); self.layer_count];
        self.errors = vec![Matrix::new(0, 0); self.layer_count];
        self.weights = vec![Matrix::new(0, 0); self.layer_count];
        self.transposed_weights = vec![Matrix::new(0, 0); self.layer_count];
        self.bias = vec![Matrix::new(0, 0); self.layer_count];
        self.pre_activation = vec![Matrix::new(0, 0); self.layer_count];
    
        self.avg_weight = vec![Matrix::new(0, 0); self.layer_count];
        self.avg_bias = vec![Matrix::new(0, 0); self.layer_count];
        self.bias_gradient = vec![Matrix::new(0, 0); self.layer_count];
        self.weight_gradient = vec![Matrix::new(0, 0); self.layer_count];
    
        self.target_vector = Matrix::new(self.layer_lengths[self.last_layer], 1);
        self.current_cost = Matrix::new(self.layer_lengths[self.last_layer], 1);
        self.floating_average = 0.0;
    }

    fn initialize_training_data(&mut self, data: &DataSet) {
        self.training_data = data.clone();
        self.initialize_validation_data(data);
    }

    fn initialize_validation_data(&mut self, data: &DataSet) {
        self.validation_data = data.clone();
    }

    fn is_finished(&mut self) -> bool {
        self.get_average_accuracy() >= TARGET_ACCURACY
    }

    fn get_average_accuracy(&mut self) -> f32 {
        self.floating_average *= FLOATING_WEIGHT;
        self.floating_average += (1.0 - FLOATING_WEIGHT) * self.generate_validation();
        self.floating_average
    }

    fn learn(&mut self) {
        // Feedforward
        self.calculate_layers();

        self.calculate_cost();

        let activated = self.derivative_activation(self.last_layer);
        self.errors[self.last_layer] = self.current_cost.hadamard_product(&activated).unwrap();

        // Backpropagate
        for i in (1..(self.layer_count - 1)).rev() {
            self.transposed_weights[i + 1] = self.weights[i + 1].transpose();
            self.errors[i] = self.transposed_weights[i + 1].multiply(&self.errors[i + 1]).unwrap();
            let activated = self.derivative_activation(i);
            self.errors[i] = self.errors[i].hadamard_product(&activated).unwrap();
        }

        // Gradiant Weights
        for i in 1..self.layer_count {
            self.transposed_layers[i - 1] = self.layers[i - 1].transpose();
            self.weight_gradient[i] = self.errors[i].dyadic_product(&self.transposed_layers[i - 1]).unwrap();
        }

        for i in 0..self.layer_count {
            self.bias_gradient[i] = self.errors[i].clone();
        }

        for i in 0..self.layer_count {
            self.avg_bias[i] = self.avg_bias[i].add(&self.bias_gradient[i]).unwrap();
            self.avg_weight[i] = self.avg_weight[i].add(&self.weight_gradient[i]).unwrap();
        }
    }

    fn guess(&mut self, input: &Vec<f32>) -> usize {
        self.set_input(input);

        self.calculate_layers();

        let size = self.layers[self.last_layer].len();
        if size == 1 {
            self.layers[self.last_layer].get_at_index(0).round() as usize
        } else {
            let mut sum = 0.0;
            for i in 0..size {
                sum += self.layers[self.last_layer].get_at_index(i).exp();
            }
            for i in 0..size {
                let new_val = self.layers[self.last_layer].get_at_index(i).exp();
                self.layers[self.last_layer].set_at_index(i, new_val / sum);
            }
            let mut index = usize::MAX;
            let mut highest_prob = 0.0;
            for i in 0..size {
                let g = self.layers[self.last_layer].get_at_index(i);
                if g > highest_prob {
                    highest_prob = g;
                    index = i;
                }
            }
            index
        }
    }
    
    fn train(&mut self) {
        let index = self.training_data.get_random_index();
        let input = self.training_data.get_input(index).as_vec();
        let output = self.training_data.get_output(index);
        // println!("{:?} {:?}", input, output);
        self.internal_train(&input, &output.as_vec());
    }

    fn adapt_weights(&mut self) {
        // println!("{:?}", self);
        for i in (0..=self.last_layer).rev() {
            self.avg_bias[i].multiply_scalar(LEARN_FACTOR);
            self.bias[i] = self.bias[i].sub(&self.avg_bias[i]).unwrap();

            // self.weights[i].multiply_scalar(LEARN_REG_BATCH_FACTOR);
            self.avg_weight[i].multiply_scalar(LEARN_FACTOR);
            self.weights[i] = self.weights[i].sub(&self.avg_weight[i]).unwrap();
        }
        for i in 0..self.layer_count {
            self.avg_bias[i].fill(0.0);
            self.avg_weight[i].fill(0.0);
        }
    }

    fn set_target(&mut self, target: &Vec<f32>) {
        self.target_vector.fill_vec(target);
    }

    fn set_input(&mut self, input: &Vec<f32>) {
        self.layers[0].fill_vec(input);
    }

    fn calculate_cost(&mut self) {
        self.current_cost = self.layers[self.last_layer].sub(&self.target_vector).unwrap();
    }

    fn calculate_layers(&mut self) {
        for i in 1..self.layer_count {
            self.calculate_layer(i);
        }
    }

    fn print_guess(&mut self) {
        for _ in 0..10 {
            let index = self.validation_data.get_random_index();
            let input = self.validation_data.get_input(index).as_vec();
            let output = self.validation_data.get_output(index);
            let g = self.guess(&input);
            println!("Guess: {}, Solution: {}", g, output.get_solution());
        }
        println!("Average: {:.2}%", self.floating_average * 100.0);
    }
}

impl NN {
    pub fn new(layer_sizes: Vec<usize>) -> Result<Self, &'static str> {
        if layer_sizes.len() == 1 {
            Err("Network only has one layer.")
        } else if layer_sizes.len() < 1 {
            Err("Network has no layers.")
        } else {
            let mut result: NN = Default::default();
            result.initialize_layers(layer_sizes);
            result.initialize_network();
            // println!("Initialized a Neural Network!\n{:?}", result);
            Ok(result)
        }
    }

    fn initialize_network(&mut self) {
        for i in 0..self.layer_count {
            let len = self.layer_lengths[i];

            self.layers[i] = Matrix::new(len, 1);
            self.transposed_layers[i] = Matrix::new(1, len);
            self.errors[i] = Matrix::new(len, 1);
            self.pre_activation[i] = Matrix::new(len, 1);
        }

        self.weights[0] = Matrix::new(0, 0);
        self.transposed_weights[0] = Matrix::new(0, 0);
        self.bias[0] = Matrix::new(0, 0);
        self.errors[0] = Matrix::new(0, 0);

        for i in 1..self.layer_count {
            let prev_len = self.layer_lengths[i - 1];
            let curr_len = self.layer_lengths[i];

            self.weights[i] = Matrix::new(curr_len, prev_len);
            self.transposed_weights[i] = Matrix::new(prev_len, curr_len);

            let two_over_input_count = 2.0 / (prev_len as f32);
            self.weights[i].gaussian_fill(0.0, two_over_input_count);

            self.bias[i] = Matrix::new(curr_len, 1);
            self.bias[i].fill(0.1);
        }

        for i in 0..self.layer_count {
            let (b_row, b_col) = self.bias[i].get_dim();
            let (w_row, w_col) = self.weights[i].get_dim();

            self.avg_bias[i] = Matrix::new(b_row, b_col);
            self.avg_weight[i] = Matrix::new(w_row, w_col);
            self.weight_gradient[i] = Matrix::new(w_row, w_col);
        }
    }

    
    fn internal_train(&mut self, input: &Vec<f32>, output: &Vec<f32>) {
        self.set_target(output);
        self.set_input(input);
        self.learn();
    }
    
    fn generate_validation(&mut self) -> f32 {
        let mut correct = 0;
        let mut total = 0;
        for _ in 0..VALIDATION_SIZE {
            let index = self.validation_data.get_random_index();
            let input = self.validation_data.get_input(index).as_vec();
            let output = self.validation_data.get_output(index);
            let g = self.guess(&input);
            if g == output.get_solution() as usize {
                correct += 1;
            }
            total += 1;
        }
        (correct as f32) / (total as f32)
    }

    fn calculate_layer(&mut self, layer: usize) {
        self.pre_activation[layer] = self.weights[layer].multiply(&self.layers[layer - 1]).unwrap();
        self.pre_activation[layer] = self.pre_activation[layer].add(&self.bias[layer]).unwrap();
        self.calculate_activation(layer);
    }

    fn calculate_activation(&mut self, layer: usize) {
        let (r, c) = self.pre_activation[layer].get_dim();
        for x in 0..r {
            for y in 0..c {
                let v = self.pre_activation[layer].get_unchecked(x, y);
                let v = self.sigmoid(v);
                self.layers[layer].set_unchecked(x, y, v);
            }
        }
    }

    fn derivative_activation(&mut self, curr_layer: usize) -> Matrix {
        let layer = &self.pre_activation[curr_layer];
        let (r, c) = layer.get_dim();
        let mut result = Matrix::new(r, c);
        for x in 0..r {
            for y in 0..c {
                let v = self.sigmoid_derivative(layer.get_unchecked(x, y));
                result.set_unchecked(x, y, v);
            }
        }
        result
    }

    fn sigmoid(self: &Self, val: f32) -> f32 {
        1.0 /  (1.0 + (-val).exp())
    }

    fn sigmoid_derivative(self: &Self, val: f32) -> f32 {
        let t = self.sigmoid(val);
        t * (1.0 - t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const TEST_CASES: usize = 10;
    #[test]
    fn init_nn() {
        for _ in 0..TEST_CASES {
            let big_net = NN::new(vec![10, 15, 20, 25, 30]);
            assert!(big_net.is_ok());

            let big_net = big_net.unwrap();

            assert_eq!(big_net.weights[0].get_dim(), (0, 0));
            assert_eq!(big_net.weights[1].get_dim(), (15, 10));
            assert_eq!(big_net.weights[2].get_dim(), (20, 15));
            assert_eq!(big_net.weights[3].get_dim(), (25, 20));
            assert_eq!(big_net.weights[4].get_dim(), (30, 25));

            assert_eq!(big_net.bias[0].get_dim(), (0, 0));
            assert_eq!(big_net.bias[1].get_dim(), (15, 1));
            assert_eq!(big_net.bias[2].get_dim(), (20, 1));
            assert_eq!(big_net.bias[3].get_dim(), (25, 1));
            assert_eq!(big_net.bias[4].get_dim(), (30, 1));

            assert_eq!(big_net.target_vector.get_dim(), (30, 1));

            let n = NN::new(vec![10, 3]);
            assert!(n.is_ok());

            let one_layer = NN::new(vec![0]);
            assert_eq!(one_layer.err().unwrap(), "Network only has one layer.");

            let no_layer = NN::new(vec![]);
            assert_eq!(no_layer.err().unwrap(), "Network has no layers.");
        }
    }
}
