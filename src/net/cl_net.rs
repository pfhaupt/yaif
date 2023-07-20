use crate::{cl_kernel::ClStruct, data_set::DataSet, cl_buffer::ClBuffer};

use super::NetTrait;

use opencl3::error_codes::ClError;

const LEARN_FACTOR: f32 = 0.05;
// const REGULATION_FACTOR: f32 = 1e-11;
// const LEARN_REG_BATCH_FACTOR: f32 = 1.0;
const TARGET_ACCURACY: f32 = 0.98;
const FLOATING_WEIGHT: f32 = 0.1;
const VALIDATION_SIZE: usize = 500;

pub struct ClNet {
    // General stuff
    layer_lengths: Vec<usize>,
    layer_count: usize,
    last_layer: usize,

    // The magic, but this time with GPU buffers
    layers: Vec<ClBuffer>,
    transposed_layers: Vec<ClBuffer>,
    weights: Vec<ClBuffer>,
    transposed_weights: Vec<ClBuffer>,
    bias: Vec<ClBuffer>,
    transposed_bias: Vec<ClBuffer>,
    errors: Vec<ClBuffer>,
    error_buffer: Vec<ClBuffer>,
    pre_activation: Vec<ClBuffer>,
    der_activation: Vec<ClBuffer>,

    avg_weight: Vec<ClBuffer>,
    avg_bias: Vec<ClBuffer>,
    bias_gradient: Vec<ClBuffer>,
    weight_gradient: Vec<ClBuffer>,

    // For Validation
    target_vector: ClBuffer,
    current_cost: ClBuffer,
    floating_average: f32,

    // Training Data
    training_data: DataSet,
    validation_data: DataSet,

    // OpenCL handler
    cl_struct: ClStruct
}

impl Default for ClNet {
    fn default() -> Self {
        Self {
            layer_lengths: vec![],
            layer_count: 0,
            last_layer: 0,

            layers: vec![],
            transposed_layers: vec![],
            weights: vec![],
            transposed_weights: vec![],
            bias: vec![],
            transposed_bias: vec![],
            errors: vec![],
            error_buffer: vec![],
            pre_activation: vec![],
            der_activation: vec![],

            avg_weight: vec![],
            avg_bias: vec![],

            bias_gradient: vec![],
            weight_gradient: vec![],

            target_vector: ClBuffer::default(),
            current_cost: ClBuffer::default(),
            floating_average: 0.0,

            training_data: DataSet::default(),
            validation_data: DataSet::default(),

            cl_struct: ClStruct::new().unwrap()
        }
    }
}

impl NetTrait for ClNet {
    fn initialize_layers(&mut self, layer_sizes: Vec<usize>) {
        self.layer_lengths = layer_sizes.clone();
        self.layer_count = self.layer_lengths.len();
        self.last_layer = self.layer_count - 1;

        for i in 0..self.layer_count {
            let len = self.layer_lengths[i];
            self.layers.push(ClBuffer::new(&self.cl_struct, len, 1));
            self.transposed_layers.push(ClBuffer::new(&self.cl_struct, 1, len));

            self.errors.push(ClBuffer::new(&self.cl_struct, len, 1));
            self.error_buffer.push(ClBuffer::new(&self.cl_struct, 1, len));

            self.pre_activation.push(ClBuffer::new(&self.cl_struct, len, 1));
            self.der_activation.push(ClBuffer::new(&self.cl_struct, len, 1));

            self.bias_gradient.push(ClBuffer::new(&self.cl_struct, len, 1));
        }

        self.weights.push(ClBuffer::new(&self.cl_struct, 0, 0));
        self.transposed_weights.push(ClBuffer::new(&self.cl_struct, 0, 0));
        self.bias.push(ClBuffer::new(&self.cl_struct, 0, 0));
        self.errors.push(ClBuffer::new(&self.cl_struct, 0, 0));

        for i in 1..self.layer_count {
            let prev_len = self.layer_lengths[i - 1];
            let curr_len = self.layer_lengths[i];

            self.weights.push(ClBuffer::new(&self.cl_struct, curr_len, prev_len));
            self.transposed_weights.push(ClBuffer::new(&self.cl_struct, prev_len, curr_len));

            let two_over_input_count = 2.0 / (prev_len as f32);
            match self.fill_buffer_gauss(&self.weights[i], 0.0, two_over_input_count) {
                Ok(()) => {},
                Err(e) => panic!("Unrecoverable error when initializing Network layers: {}", e)
            }

            self.bias.push(ClBuffer::new(&self.cl_struct, curr_len, 1));
            self.transposed_bias.push(ClBuffer::new(&self.cl_struct, 1, curr_len));

            match self.fill_buffer_scalar(&self.bias[i], 0.1) {
                Ok(()) => {},
                Err(e) => panic!("Unrecoverable error when initializing Network layers: {}", e)
            }
        }

        for i in 0..self.layer_count {
            let (b_row, b_col) = self.bias[i].get_dims();
            let (w_row, w_col) = self.weights[i].get_dims();

            self.avg_bias.push(ClBuffer::new(&self.cl_struct, b_row, b_col));
            self.avg_weight.push(ClBuffer::new(&self.cl_struct, w_row, w_col));
            self.weight_gradient.push(ClBuffer::new(&self.cl_struct, w_row, w_col));
        }

        self.target_vector = ClBuffer::new(&self.cl_struct, self.layer_lengths[self.last_layer], 1);
        self.current_cost = ClBuffer::new(&self.cl_struct, self.layer_lengths[self.last_layer], 1);
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
        self.calculate_layers();

        self.calculate_cost();

        self.calculate_derivations();

        self.cl_struct.matrix_hadamard(&mut self.current_cost, &mut self.der_activation[self.last_layer], &mut self.errors[self.last_layer]).unwrap();

        // Backpropagation is now ready to go
        for i in (1..(self.layer_count - 1)).rev() {
            self.cl_struct.matrix_transpose(&mut self.weights[i + 1], &mut self.transposed_weights[i + 1]).unwrap();
            self.cl_struct.matrix_mult(&mut self.transposed_weights[i + 1], &mut self.errors[i + 1], &mut self.error_buffer[i]).unwrap();
            self.cl_struct.matrix_hadamard(&mut self.error_buffer[i], &mut self.der_activation[i], &mut self.errors[i]).unwrap();
        }

        // Gradiant Weights
        for i in 1..self.layer_count {
            self.cl_struct.matrix_transpose(&mut self.layers[i - 1], &mut self.transposed_layers[i - 1]).unwrap();
            self.cl_struct.matrix_dyadic(&mut self.errors[i], &mut self.transposed_layers[i - 1], &mut self.weight_gradient[i]).unwrap();
        }

        for i in 0..self.layer_count {
            self.cl_struct.copy_buffer(&mut self.errors[i], &mut self.bias_gradient[i]).unwrap();
        }

        for i in 0..self.layer_count {
            self.cl_struct.matrix_add_inline(&mut self.avg_bias[i], &mut self.bias_gradient[i]).unwrap();
            self.cl_struct.matrix_add_inline(&mut self.avg_weight[i], &mut self.weight_gradient[i]).unwrap();
        }
    }

    fn guess(&mut self, input: &Vec<f32>) -> usize {
        self.set_input(input);

        self.calculate_layers();
        
        let mut r = self.cl_struct.read_buffer(&self.layers[self.last_layer]).unwrap();
        if r.len() == 1 {
            r[0].round() as usize
        } else {
            let size = r.len();

            let mut sum = 0.0;
            for i in 0..size {
                sum += r[i].exp();
            }
            for i in 0..size {
                let new_val = r[i].exp();
                r[i] = new_val / sum;
            }
            let mut index = usize::MAX;
            let mut highest_prob = 0.0;
            for i in 0..size {
                let g = r[i];
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
        for i in (0..=self.last_layer).rev() {
            self.cl_struct.matrix_scalar_mult(&mut self.avg_bias[i], LEARN_FACTOR).unwrap();
            self.cl_struct.matrix_sub_inline(&mut self.bias[i], &mut self.avg_bias[i]).unwrap();

            self.cl_struct.matrix_scalar_mult(&mut self.avg_weight[i], LEARN_FACTOR).unwrap();
            self.cl_struct.matrix_sub_inline(&mut self.weights[i], &mut self.avg_weight[i]).unwrap();
        }
        for i in 0..self.layer_count {
            self.cl_struct.fill_scalar(&self.avg_bias[i], 0.0).unwrap();
            self.cl_struct.fill_scalar(&self.avg_weight[i], 0.0).unwrap();
        }
    }
    
    fn set_target(&mut self, target: &Vec<f32>) {
        self.fill_buffer_vec(&self.layers[self.last_layer], target.clone()).unwrap();
    }

    fn set_input(&mut self, input: &Vec<f32>) {
        self.fill_buffer_vec(&self.layers[0], input.clone()).unwrap();
    }

    fn calculate_cost(&mut self) {
        self.cl_struct.matrix_sub(&mut self.layers[self.last_layer], &mut self.target_vector, &mut self.current_cost).unwrap();
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

impl ClNet {
    pub fn new(cl_struct: ClStruct, layer_sizes: Vec<usize>) -> Result<Self, &'static str> {
        if layer_sizes.len() == 1 {
            Err("Network only has one layer.")
        } else if layer_sizes.len() < 1 {
            Err("Network has no layers.")
        } else {
            let mut result: ClNet = Default::default();
            result.set_cl_struct(cl_struct);
            result.load_kernels();
            result.initialize_layers(layer_sizes);
            // println!("Initialized a Neural Network!\n{:?}", result);
            Ok(result)
        }
    }

    fn internal_train(&mut self, input: &Vec<f32>, output: &Vec<f32>) {
        self.set_target(output);
        self.set_input(input);
        self.learn();
    }

    fn calculate_layer(&mut self, layer: usize) {
        self.cl_struct.matrix_mult(&mut self.weights[layer], &mut self.layers[layer - 1], &mut self.pre_activation[layer]).unwrap();
        
        self.cl_struct.matrix_add_inline(&mut self.pre_activation[layer], &mut self.bias[layer]).unwrap();
        self.calculate_activation(layer);
    }

    fn calculate_derivations(&mut self) {
        for i in 0..self.layer_count {
            self.cl_struct.der_sigmoid(&mut self.pre_activation[i], &mut self.der_activation[i]).unwrap();
        }
    }

    fn calculate_activation(&mut self, layer: usize) {
        self.cl_struct.sigmoid(&mut self.pre_activation[layer], &mut self.layers[layer]).unwrap()
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

    fn load_kernels(&mut self) {
        self.cl_struct.load_kernels();
    }

    fn set_cl_struct(&mut self, cl_struct: ClStruct) {
        self.cl_struct = cl_struct
    }

    fn fill_buffer_vec(&self, buffer: &ClBuffer, values: Vec<f32>) -> Result<(), ClError> {
        self.cl_struct.fill_vec(buffer, values)
    }

    fn fill_buffer_scalar(&self, buffer: &ClBuffer, val: f32) -> Result<(), ClError> {
        self.cl_struct.fill_scalar(buffer, val)
    }

    fn fill_buffer_gauss(&self, buffer: &ClBuffer, mean: f32, variance: f32) -> Result<(), ClError> {
        self.cl_struct.fill_gauss(buffer, mean, variance)
    }
}