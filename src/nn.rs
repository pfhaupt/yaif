use super::matrix::Matrix;
use super::data_set::DataSet;
use std::fmt::{ Debug, Formatter };

const LEARN_FACTOR: f32 = 0.05;
const REGULATION_FACTOR: f32 = 1e-11;
const LEARN_REG_BATCH_FACTOR: f32 = 1.0;
const TARGET_ACCURACY: f32 = 0.95;
const FLOATING_WEIGHT: f32 = 0.0;
const VALIDATION_SIZE: usize = 100;

const BATCH_SIZE: usize = 50;
const TRAINING_DATA: usize = 1_000;
const EPOCH_SIZE: usize = TRAINING_DATA / BATCH_SIZE;
const EPOCH_COUNT: usize = 100;

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

    pub fn initialize_training_data(&mut self, data: &DataSet) {
        self.training_data = data.clone();
        self.initialize_validation_data(data);
    }

    pub fn initialize_validation_data(&mut self, data: &DataSet) {
        self.validation_data = data.clone();
    }

    pub fn train(&mut self) {
        let index = self.training_data.get_random_index();
        let input = self.training_data.get_input(index).as_vec();
        let output = self.training_data.get_output(index);
        // println!("{:?} {:?}", input, output);
        self.internal_train(&input, &output.as_vec());
    }

    fn internal_train(&mut self, input: &Vec<f32>, output: &Vec<f32>) {
        self.set_target(output);
        self.set_input(input);
        self.learn();
    }

    pub fn adapt_weights(&mut self) {
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

    pub fn is_finished(&mut self) -> bool {
        self.get_average_accuracy() >= TARGET_ACCURACY
    }

    fn get_average_accuracy(&mut self) -> f32 {
        self.floating_average *= FLOATING_WEIGHT;
        self.floating_average += (1.0 - FLOATING_WEIGHT) * self.generate_validation();
        self.floating_average
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

    pub fn run(&mut self, debug: bool) -> bool {
        for _ in 0..EPOCH_COUNT {
            for _ in 0..EPOCH_SIZE {
                for _ in 0..BATCH_SIZE {
                    self.train();
                }
                self.adapt_weights();
            }
            if debug {
                self.print_guess();
            }
            if self.is_finished() {
                return true;
            }
        }
        false
    }

    fn set_target(&mut self, target: &Vec<f32>) {
        self.target_vector.fill_vec(target);
    }

    fn set_input(&mut self, input: &Vec<f32>) {
        self.layers[0].fill_vec(input);
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

    pub fn print_guess(&mut self) {
        for _ in 0..10 {
            let index = self.validation_data.get_random_index();
            let input = self.validation_data.get_input(index).as_vec();
            let output = self.validation_data.get_output(index);
            let g = self.guess(&input);
            println!("Guess: {}, Solution: {}", g, output.get_solution());
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

    fn calculate_cost(&mut self) {
        self.current_cost = self.layers[self.last_layer].sub(&self.target_vector).unwrap();
    }

    fn calculate_layers(&mut self) {
        for i in 1..self.layer_count {
            self.calculate_layer(i);
        }
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
                result.set(x, y, self.sigmoid_derivative(layer.get(x, y).unwrap())).unwrap();
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
    use std::io::Read;
    const TEST_CASES: usize = 1_000;
    const ATTEMPTS: usize = 6;
    const TARGET: usize =  ATTEMPTS / 2;
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

    #[test]
    fn xor_test() {
        let mut success_ctr = 0;
        for _ in 0..ATTEMPTS {
            let mut nn = NN::new(vec![2, 2, 2]).unwrap();
            let inputs: Vec<Vec<f32>> = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
            let outputs: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![1.0, 0.0]];
            let data = DataSet::new(&inputs, &outputs);
            nn.initialize_training_data(&data);
            let finished = nn.run(false);
            if finished {
                success_ctr += 1;
            }
        }
        assert!(success_ctr >= TARGET, "Network must pass {} out of {} tests, got {}", TARGET, ATTEMPTS, success_ctr);
    }

    #[test]
    fn mnist_test() {        
        fn parse_input(which: &str) -> Vec<Vec<f32>> {
            let (count, file_size, file_path)= match which {
                "training" => (60_000, 47_040_016, "mnist/train-images.idx3-ubyte"),
                "validation"=> (10_000, 7_840_016, "mnist/t10k-images.idx3-ubyte"),
                _ => panic!("Unknown argument `{}`", which)
            };

            match std::fs::File::open(file_path) {
                Ok(mut file) => {
                    let mut buffer = vec![];
                    match file.read_to_end(&mut buffer) {
                        Ok(size) => {
                            assert_eq!(size, file_size, "Excepted MNIST image-size is {} bytes, but reading got {} bytes.", file_size, size);
                            let next_bytes = |offset: usize| -> [u8; 4] {
                                let mut bytes = [0; 4];
                                for i in 0..4 {
                                    bytes[i] = buffer[offset + i];
                                }
                                bytes
                            };
                            let bytes = next_bytes(0);
                            let _magic_number = u32::from_be_bytes(bytes) as usize;
                            assert!(bytes[0] == 0);
                            assert!(bytes[1] == 0);
                            assert!(bytes[2] == 8); // each Byte is a value
                            assert!(bytes[3] == 3); // a Vector of matrices (or a '3D Matrix')

                            let bytes = next_bytes(4);
                            let first_dim = u32::from_be_bytes(bytes) as usize;
                            assert_eq!(first_dim, count);
                            
                            let bytes = next_bytes(8);
                            let second_dim = u32::from_be_bytes(bytes) as usize;
                            assert_eq!(second_dim, 28);

                            let bytes = next_bytes(12);
                            let third_dim = u32::from_be_bytes(bytes) as usize;
                            assert_eq!(third_dim, 28);

                            let next_byte = |index: &mut usize| -> u8 { let v = buffer[*index]; *index += 1; v };
                            let mut input = vec![vec![0.0; second_dim * third_dim]; first_dim];
                            let mut index = 16;
                            for f in 0..first_dim {
                                for i in 0..(second_dim * third_dim) {
                                    input[f][i] = next_byte(&mut index) as f32 / 256.0;
                                }
                            }
                            input
                        },
                        Err(e) => {
                            panic!("{}", e);
                        }
                    }
                },
                Err(e) => {
                    panic!("{}", e);
                }
            }
        }

        fn parse_output(which: &str) -> Vec<Vec<f32>> {
            let (count, file_size, file_path)= match which {
                "training" => (60_000, 60_008, "mnist/train-labels.idx1-ubyte"),
                "validation"=> (10_000, 10_008, "mnist/t10k-labels.idx1-ubyte"),
                _ => panic!("Unknown argument `{}`", which)
            };
            match std::fs::File::open(file_path) {
                Ok(mut file) => {
                    let mut buffer = vec![];
                    match file.read_to_end(&mut buffer) {
                        Ok(size) => {
                            assert_eq!(size, file_size, "Excepted MNIST label-size is {} bytes, but reading got {} bytes.", file_size, size);
                            
                            let next_bytes = |offset: usize| -> [u8; 4] {
                                let mut bytes = [0; 4];
                                for i in 0..4 {
                                    bytes[i] = buffer[offset + i];
                                }
                                bytes
                            };
                            let bytes = next_bytes(0);
                            let _magic_number = u32::from_be_bytes(bytes) as usize;
                            assert!(bytes[0] == 0);
                            assert!(bytes[1] == 0);
                            assert!(bytes[2] == 8); // each Byte is a value
                            assert!(bytes[3] == 1); // A single vector (or a '1D Matrix')

                            let bytes = next_bytes(4);
                            let dimension = u32::from_be_bytes(bytes) as usize;
                            assert_eq!(dimension, count);

                            let next_byte = |index: &mut usize| -> u8 { let v = buffer[*index]; *index += 1; v };
                            let mut output = vec![vec![0.0; 10]; dimension];
                            let mut index = 8;
                            for f in 0..dimension {
                                let v = next_byte(&mut index) as usize;
                                output[f][v] = 1.0;
                            }
                            output
                        },
                        Err(e) => {
                            panic!("{}", e);
                        }
                    }
                },
                Err(e) => {
                    panic!("{}", e);
                }
            }
        }

        let layers = vec![28 * 28, 50, 50, 10];

        let inputs: Vec<Vec<f32>> = parse_input("training");
        let outputs: Vec<Vec<f32>> = parse_output("training");
        let train_set = DataSet::new(&inputs, &outputs);
    
        let inputs: Vec<Vec<f32>> = parse_input("validation");
        let outputs: Vec<Vec<f32>> = parse_output("validation");
        let validation_set = DataSet::new(&inputs, &outputs);
        
        let mut success_ctr = 0;
        for _ in 0..ATTEMPTS {
            match NN::new(layers.clone()) {
                Ok(mut nn) => {
                    nn.initialize_training_data(&train_set);
                    nn.initialize_validation_data(&validation_set);
                    if nn.run(false) {
                        success_ctr += 1;
                    }
                },
                Err(e) => {
                    panic!("Could not init Network! {}", e);
                }
            }
        }
        assert!(success_ctr >= TARGET, "Network must pass {} out of {} tests, got {}", TARGET, ATTEMPTS, success_ctr);
    }
}
