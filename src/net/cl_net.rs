use crate::{cl_kernel::ClStruct, data_set::DataSet};

use opencl3::{memory::Buffer, error_codes::ClError};

use super::NetTrait;

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

    // Buffers are 1D, but matrix ops need 2D, so I need to store each dimension
    layer_dims: Vec<(usize, usize)>,
    weight_dims: Vec<(usize, usize)>,
    bias_dims: Vec<(usize, usize)>,
    error_dims: Vec<(usize, usize)>,
    active_dims: Vec<(usize, usize)>,

    // The magic, but this time with GPU buffers
    layers: Vec<Option<Buffer<f32>>>,
    transposed_layers: Vec<Option<Buffer<f32>>>,
    weights: Vec<Option<Buffer<f32>>>,
    transposed_weights: Vec<Option<Buffer<f32>>>,
    bias: Vec<Option<Buffer<f32>>>,
    transposed_bias: Vec<Option<Buffer<f32>>>,
    errors: Vec<Option<Buffer<f32>>>,
    error_buffer: Vec<Option<Buffer<f32>>>,
    pre_activation: Vec<Option<Buffer<f32>>>,
    der_activation: Vec<Option<Buffer<f32>>>,

    avg_weight: Vec<Option<Buffer<f32>>>,
    avg_bias: Vec<Option<Buffer<f32>>>,
    bias_gradient: Vec<Option<Buffer<f32>>>,
    weight_gradient: Vec<Option<Buffer<f32>>>,

    // For Validation
    target_vector: Option<Buffer<f32>>,
    current_cost: Option<Buffer<f32>>,
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
            
            layer_dims: vec![],
            weight_dims: vec![],
            bias_dims: vec![],
            error_dims: vec![],
            active_dims: vec![],

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

            target_vector: None,
            current_cost: None,
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
        
        self.layer_dims = vec![(0, 0); self.layer_count];
        self.weight_dims = vec![(0, 0); self.layer_count];
        self.bias_dims = vec![(0, 0); self.layer_count];
        self.error_dims = vec![(0, 0); self.layer_count];
        self.active_dims = vec![(0, 0); self.layer_count];

        let fill_none = |v: &mut Vec<Option<Buffer<f32>>>| {
            *v = vec![];
            for _ in 0..self.layer_count { v.push(None); }
        };

        fill_none(&mut self.layers);
        fill_none(&mut self.transposed_layers);
        fill_none(&mut self.weights);
        fill_none(&mut self.transposed_weights);
        fill_none(&mut self.bias);
        fill_none(&mut self.transposed_bias);
        fill_none(&mut self.errors);
        fill_none(&mut self.error_buffer);
        fill_none(&mut self.pre_activation);
        fill_none(&mut self.der_activation);
        fill_none(&mut self.avg_weight);
        fill_none(&mut self.avg_bias);
        fill_none(&mut self.bias_gradient);
        fill_none(&mut self.weight_gradient);

        self.target_vector = self.cl_struct.create_buffer(self.layer_lengths[self.last_layer], 1);
        self.current_cost = self.cl_struct.create_buffer(self.layer_lengths[self.last_layer], 1);
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

        let (m, n) = self.active_dims[self.last_layer];
        self.cl_struct.matrix_hadamard(&mut self.current_cost, &mut self.der_activation[self.last_layer], &mut self.errors[self.last_layer], m, n).unwrap();

        // Backpropagation is now ready to go

        for i in (1..(self.layer_count - 1)).rev() {
            let (m, n) = self.weight_dims[i + 1];
            self.cl_struct.matrix_transpose(&mut self.weights[i + 1], &mut self.transposed_weights[i + 1], m, n).unwrap();
            
            let (n, m) = self.weight_dims[i + 1]; // reversed order because transposed
            let (_, k) = self.error_dims[i];

            // self.cl_struct.matrix_mult(&mut self.transposed_weights[i + 1], &mut self.errors[i + 1], &mut elf.errors[i], m, n, k).unwrap();
            self.cl_struct.matrix_mult(&mut self.transposed_weights[i + 1], &mut self.errors[i + 1], &mut self.error_buffer[i], m, n, k).unwrap();
            
            let (m, n) = self.error_dims[i];
            self.cl_struct.matrix_hadamard(&mut self.error_buffer[i], &mut self.der_activation[i], &mut self.errors[i], m, n).unwrap();
        }

        // Gradiant Weights
        for i in 1..self.layer_count {
            let (m, n) = self.layer_dims[i - 1];
            self.cl_struct.matrix_transpose(&mut self.layers[i - 1], &mut self.transposed_layers[i - 1], m, n).unwrap();

            let (m, n) = self.weight_dims[i];
            self.cl_struct.matrix_dyadic(&mut self.errors[i], &mut self.transposed_layers[i - 1], &mut self.weight_gradient[i], m, n).unwrap();
        }

        for i in 0..self.layer_count {
            let (m, n) = self.error_dims[i];
            self.cl_struct.copy_buffer(&mut self.errors[i], &mut self.bias_gradient[i], m, n).unwrap();
        }

        for i in 0..self.layer_count {
            let (m, n) = self.bias_dims[i];
            self.cl_struct.matrix_add_inline(&mut self.avg_bias[i], &mut self.bias_gradient[i], m, n).unwrap();

            let (m, n) = self.weight_dims[i];
            self.cl_struct.matrix_add_inline(&mut self.avg_weight[i], &mut self.weight_gradient[i], m, n).unwrap();

        }
    }

    fn guess(&mut self, input: &Vec<f32>) -> usize {
        self.set_input(input);

        self.calculate_layers();

        let (m, n) = self.layer_dims[self.last_layer];
        
        let mut r = self.cl_struct.read_buffer(&self.layers[self.last_layer], m * n).unwrap();
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
            let (m, n) = self.bias_dims[i];
            self.cl_struct.matrix_scalar_mult(&mut self.avg_bias[i], LEARN_FACTOR, m, n).unwrap();
            self.cl_struct.matrix_sub_inline(&mut self.bias[i], &mut self.avg_bias[i], m, n).unwrap();

            let (m, n) = self.weight_dims[i];
            self.cl_struct.matrix_scalar_mult(&mut self.avg_weight[i], LEARN_FACTOR, m, n).unwrap();
            self.cl_struct.matrix_sub_inline(&mut self.weights[i], &mut self.avg_weight[i], m, n).unwrap();
        }
        for i in 0..self.layer_count {
            let (m, n) = self.bias_dims[i];
            self.cl_struct.fill_scalar(&self.avg_bias[i], 0.0, m, n).unwrap();

            let (m, n) = self.weight_dims[i];
            self.cl_struct.fill_scalar(&self.avg_weight[i], 0.0, m, n).unwrap();
        }
    }
    
    fn set_target(&mut self, target: &Vec<f32>) {
        self.fill_buffer_vec(&self.layers[self.last_layer], target.clone(), target.len(), 1).unwrap();
    }

    fn set_input(&mut self, input: &Vec<f32>) {
        self.fill_buffer_vec(&self.layers[0], input.clone(),input.len(), 1).unwrap();
    }

    fn calculate_cost(&mut self) {
        let (m, n) = self.layer_dims[self.last_layer];
        self.cl_struct.matrix_sub(&mut self.layers[self.last_layer], &mut self.target_vector, &mut self.current_cost, m, n).unwrap();
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
            result.initialize_network()?;
            // println!("Initialized a Neural Network!\n{:?}", result);
            Ok(result)
        }
    }
    
    fn initialize_network(&mut self) -> Result<(), ClError> {
        for i in 0..self.layer_count {
            let len = self.layer_lengths[i];

            self.layers[i] = self.cl_struct.create_buffer(len, 1);
            self.transposed_layers[i] = self.cl_struct.create_buffer(1, len);
            self.layer_dims[i] = (len, 1);

            self.errors[i] = self.cl_struct.create_buffer(len, 1);
            self.error_buffer[i] = self.cl_struct.create_buffer(len, 1);
            self.error_dims[i] = (len, 1);

            self.pre_activation[i] = self.cl_struct.create_buffer(len, 1);
            self.der_activation[i] = self.cl_struct.create_buffer(len, 1);
            self.active_dims[i] = (len, 1);

            self.bias_gradient[i] = self.cl_struct.create_buffer(len, 1);
        }

        self.weights[0] = self.cl_struct.create_buffer(0, 0);
        self.weight_dims[0] = (0, 0);
        self.transposed_weights[0] = self.cl_struct.create_buffer(0, 0);
        self.bias[0] = self.cl_struct.create_buffer(0, 0);
        self.bias_dims[0] = (0, 0);
        self.errors[0] = self.cl_struct.create_buffer(0, 0);
        self.error_dims[0] = (0, 0);

        for i in 1..self.layer_count {
            let prev_len = self.layer_lengths[i - 1];
            let curr_len = self.layer_lengths[i];

            self.weights[i] = self.cl_struct.create_buffer(curr_len, prev_len);
            self.transposed_weights[i] = self.cl_struct.create_buffer(prev_len, curr_len);
            self.weight_dims[i] = (curr_len, prev_len);

            let two_over_input_count = 2.0 / (prev_len as f32);
            self.fill_buffer_gauss(&self.weights[i], 0.0, two_over_input_count, curr_len, prev_len)?;

            self.bias[i] = self.cl_struct.create_buffer(curr_len, 1);
            self.transposed_bias[i] = self.cl_struct.create_buffer(1, curr_len);
            self.bias_dims[i] = (curr_len, 1);

            self.fill_buffer_scalar(&self.bias[i], 0.1, curr_len, 1)?;
        }

        for i in 0..self.layer_count {
            let (b_row, b_col) = self.bias_dims[i];
            let (w_row, w_col) = self.weight_dims[i];
            
            self.avg_bias[i] = self.cl_struct.create_buffer(b_row, b_col);
            self.avg_weight[i] = self.cl_struct.create_buffer(w_row, w_col);
            self.weight_gradient[i] = self.cl_struct.create_buffer(w_row, w_col);
        }
        Ok(())
    }

    fn internal_train(&mut self, input: &Vec<f32>, output: &Vec<f32>) {
        self.set_target(output);
        self.set_input(input);
        self.learn();
    }

    fn calculate_layer(&mut self, layer: usize) {
        let (m, n1) = self.weight_dims[layer];
        let (_, k) = self.layer_dims[layer - 1];
        let n = n1;
        self.cl_struct.matrix_mult(&mut self.weights[layer], &mut self.layers[layer - 1], &mut self.pre_activation[layer], m, n, k).unwrap();
        
        let (m, n) = self.bias_dims[layer];
        self.cl_struct.matrix_add_inline(&mut self.pre_activation[layer], &mut self.bias[layer], m, n).unwrap();
        self.calculate_activation(layer);
    }

    fn calculate_derivations(&mut self) {
        for i in 0..self.layer_count {
            let (m, n) = self.active_dims[i];
            self.cl_struct.der_sigmoid(&mut self.pre_activation[i], &mut self.der_activation[i], m, n).unwrap();
        }
    }

    fn calculate_activation(&mut self, layer: usize) {
        let (r, c) = self.layer_dims[layer];
        self.cl_struct.sigmoid(&mut self.pre_activation[layer], &mut self.layers[layer], r, c).unwrap()
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

    fn fill_buffer_vec(&self, buffer: &Option<Buffer<f32>>, values: Vec<f32>, m: usize, n: usize) -> Result<(), ClError> {
        self.cl_struct.fill_vec(buffer, values, m, n)
    }

    fn fill_buffer_scalar(&self, buffer: &Option<Buffer<f32>>, val: f32, m: usize, n: usize) -> Result<(), ClError> {
        self.cl_struct.fill_scalar(buffer, val, m, n)
    }

    fn fill_buffer_gauss(&self, buffer: &Option<Buffer<f32>>, mean: f32, variance: f32, m: usize, n: usize) -> Result<(), ClError> {
        self.cl_struct.fill_gauss(buffer, mean, variance, m, n)
    }
}